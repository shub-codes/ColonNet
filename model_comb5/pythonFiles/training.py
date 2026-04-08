"""
training.py  — ColonNet Training Pipeline
================================================================
Architecture
  Stage 1  Bounding box    — EfficientNet-B0 + YOLOv5n-style head
  Stage 2  Classification  — EfficientNet-B0 + lightweight cls head
  Stage 3  Segmentation    — Attention U-Net (base_filters=32)
  Stage 4  Combined model  — frozen Stage 2 + Stage 3

Key differences vs. Combo 2 (training.py)
  • Backbone: EfficientNetB0  instead of MobileNetV3Small
  • Bbox head: YOLOv5n-style FPN (C3-lite + DW-sep)  instead of SSD-Lite
  • HPO: Optuna (TPE, 20–25 trials)  instead of RandomSearch
  • Backbone layer name constant updated to match EfficientNetB0 sub-model

Fixes carried forward from Combo 2 (training.py)
  FIX-T1  combined_box_loss uses smooth-L1 only (no GIoU) to keep loss ≥ 0
  FIX-T2  Optuna search space caps LR at 3.16e-4 (log10 ≤ −3.5)
  FIX-T3  Stage 1 caps its LR at 1e-4 regardless of Optuna's best value
  FIX-T4  BoxIoUCallback aborts Stage 1 if mean IoU < 0.15 at epoch 10
  FIX-T5  Stage 2 uses conservative LR (1e-4) for the classification head
  FIX-T6  Stage 2 partially unfreezes the last ~20 % of backbone sub-layers
  FIX-T7  Stage 1 bleeding-only; Stage 2 bleeding + non-bleeding
"""

import gc
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ── Optuna ────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise ImportError(
        "Optuna is required for Combination 1.\n"
        "Install it with:  pip install optuna"
    ) from e

from utils.losses import focal_tversky, tversky
from utils.data_loaders import load_data, load_data_unet
from utils.base_models import build_model, Build_AttUnet_Model

# ─────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "SavedModels")
os.makedirs(MODELS_DIR, exist_ok=True)

def mp(name):
    return os.path.join(MODELS_DIR, name)

PATH_BOX    = mp("CheckPoint1.keras")
PATH_CLS    = mp("classNbox.keras")
PATH_SEG    = mp("segmentation.keras")
PATH_COLON  = mp("ColonNet.keras")
PATH_OPTUNA = mp("optuna_best_params.json")

# Sub-model name for the EfficientNetB0 backbone (used in Stage 2 fine-tuning)
BACKBONE_LAYER_NAME = "EfficientNetB0_FPN"

# ─────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────

def smooth_l1(y_true, y_pred):
    """Element-wise smooth-L1, averaged over the batch."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    loss   = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    return tf.reduce_mean(loss)


def combined_box_loss(y_true, y_pred):
    """
    Smooth-L1 masked to valid (area > 0) boxes — always ≥ 0.
    Non-bleeding samples have target [0,0,0,0] and are excluded so they
    don't pull the head toward the degenerate 'predict nothing' solution.
    (FIX-T1: no GIoU term — keeps loss non-negative for stable early stopping)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)

    diff           = tf.abs(y_true - y_pred)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)

    masked  = sl1_per_sample * valid_mask
    n_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(masked) / n_valid


LOSSES = {
    "c_final": tf.keras.losses.BinaryCrossentropy(),
    "b_final": combined_box_loss,
}
CUSTOM_BOX = {
    "smooth_l1":         smooth_l1,
    "combined_box_loss": combined_box_loss,
}

# ─────────────────────────────────────────────────────────
# SEGMENTATION METRICS
# ─────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

SEG_CUSTOM = {
    "focal_tversky": focal_tversky,
    "tversky":       tversky,
    "dice_coef":     dice_coef,
}

# ─────────────────────────────────────────────────────────
# BOX IoU CALLBACK  (FIX-T4)
# ─────────────────────────────────────────────────────────

class BoxIoUCallback(tf.keras.callbacks.Callback):
    """
    Prints mean IoU each epoch and aborts Stage 1 early if the box head
    degenerates to predicting the full image for every sample.
    (FIX-T4 — carried forward unchanged from Combo 2)
    """
    ABORT_THRESHOLD = 0.15
    ABORT_EPOCH     = 10

    def __init__(self, X_val, box_val):
        super().__init__()
        self.X_val   = X_val
        self.box_val = box_val

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        if isinstance(preds, (list, tuple)):
            box_pred = np.array(preds[1])
        else:
            box_pred = np.array(preds["b_final"])

        box_true = self.box_val
        box_pred = np.clip(box_pred, 0.0, 1.0)

        ix1   = np.maximum(box_true[:, 0], box_pred[:, 0])
        iy1   = np.maximum(box_true[:, 1], box_pred[:, 1])
        ix2   = np.minimum(box_true[:, 2], box_pred[:, 2])
        iy2   = np.minimum(box_true[:, 3], box_pred[:, 3])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        at    = (box_true[:, 2] - box_true[:, 0]) * (box_true[:, 3] - box_true[:, 1])
        ap    = (box_pred[:, 2] - box_pred[:, 0]) * (box_pred[:, 3] - box_pred[:, 1])
        union = at + ap - inter + 1e-7
        ious  = inter / union

        valid    = at > 0
        mean_iou = float(np.mean(ious[valid])) if valid.any() else 0.0
        print(f"  [BoxIoU] epoch {epoch+1:>3d}  val mean-IoU = {mean_iou:.4f}", end="")

        if mean_iou < self.ABORT_THRESHOLD and (epoch + 1) >= self.ABORT_EPOCH:
            print(f"\n  ⚠  BoxIoU below {self.ABORT_THRESHOLD} at epoch {epoch+1}.")
            print("  ⚠  Box head may have collapsed. "
                  "Delete CheckPoint1.keras and re-check data / loss.")
            self.model.stop_training = True
        else:
            print()


# ─────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────

class PrintMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        metrics = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()
                            if isinstance(v, (int, float)))
        print(f"Epoch {epoch + 1} — {metrics}")


def make_callbacks(ckpt_path, monitor="val_loss", mode="min",
                   es_patience=5, lr_patience=3, extra=None):
    cbs = [
        PrintMetrics(),
        ModelCheckpoint(ckpt_path, monitor=monitor, mode=mode,
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor=monitor, mode=mode, patience=es_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=lr_patience, min_lr=1e-6, verbose=1),
    ]
    if extra:
        cbs.extend(extra)
    return cbs


# ─────────────────────────────────────────────────────────
# OPTUNA HYPERPARAMETER SEARCH  (replaces RandomSearch)
# 20–25 trials of TPE with a log-uniform LR search space.
# Search space mirrors Combo 2's RandomSearch bounds exactly so results
# are directly comparable.
# ─────────────────────────────────────────────────────────

N_OPTUNA_TRIALS = 25   # set to 20 if compute is tight

if not os.path.exists(PATH_OPTUNA):
    print("Loading dataset for Optuna search …")
    _opt_imgs, _opt_boxes, _opt_labels = load_data(with_neg=False, aug=False)
    N   = min(512, len(_opt_imgs))
    idx = np.random.permutation(len(_opt_imgs))[:N]
    _opt_imgs   = _opt_imgs[idx]
    _opt_boxes  = _opt_boxes[idx]
    _opt_labels = _opt_labels[idx]
    print(f"Optuna subset shape: {_opt_imgs.shape}")


def optuna_objective(trial):
    """
    Optuna objective function for Stage 1 (bbox) hyperparameter search.

    Search space
    ─────────────────────────────────────────────────────
    lr          log-uniform in [1e-6, 3.16e-4]   (FIX-T2: same upper cap as Combo 2)
    dropout     uniform in [0.0, 0.6]
    dense_scale uniform in [0.5, 1.5]
    ─────────────────────────────────────────────────────
    Each trial trains for 10 epochs on a ≤512-sample subset and returns
    the final val combined_box_loss. Lower is better (Optuna direction="minimize").
    """
    lr          = trial.suggest_float("lr",          1e-6,  3.16e-4, log=True)
    dropout     = trial.suggest_float("dropout",     0.0,   0.6)
    dense_scale = trial.suggest_float("dense_scale", 0.5,   1.5)

    Xtr, Xval, btr, bval, ltr, lval = train_test_split(
        _opt_imgs, _opt_boxes, _opt_labels, test_size=0.2, random_state=42
    )
    tf.keras.backend.clear_session()
    model = build_model(
        dropout_cls=dropout, dropout_reg=dropout, dense_scale=dense_scale
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=LOSSES,
        loss_weights={"c_final": 0.0, "b_final": 1.0},
    )
    history = model.fit(
        Xtr, {"c_final": ltr, "b_final": btr},
        validation_data=(Xval, {"c_final": lval, "b_final": bval}),
        epochs=10, batch_size=8, verbose=0,
    )
    val_loss = float(history.history.get("val_b_final_loss", [1e6])[-1])
    del model
    gc.collect()
    return val_loss


if os.path.exists(PATH_OPTUNA):
    with open(PATH_OPTUNA) as f:
        _p = json.load(f)
    best_lr      = _p["lr"]
    best_dropout = _p["dropout"]
    best_scale   = _p["dense_scale"]
    print(f"Optuna params loaded → LR={best_lr:.4e}  "
          f"dropout={best_dropout:.3f}  scale={best_scale:.3f}")
else:
    print(f"Starting Optuna TPE search ({N_OPTUNA_TRIALS} trials) …")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best_params  = study.best_params
    best_lr      = best_params["lr"]
    best_dropout = best_params["dropout"]
    best_scale   = best_params["dense_scale"]

    with open(PATH_OPTUNA, "w") as f:
        json.dump({"lr": best_lr, "dropout": best_dropout,
                   "dense_scale": best_scale,
                   "optuna_best_value": study.best_value}, f, indent=2)
    print(f"Optuna done → LR={best_lr:.4e}  "
          f"dropout={best_dropout:.3f}  scale={best_scale:.3f}  "
          f"val_loss={study.best_value:.6f}")
    print(f"Saved → {PATH_OPTUNA}")

tf.keras.backend.clear_session()

# =====================================================
# STAGE 1 — BOUNDING BOX  (bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 1 — Bounding Box Regression (YOLOv5n head)")
print("=" * 55)

X_train, X_val, box_train, box_val, label_train, label_val = train_test_split(
    *load_data(with_neg=False, aug=False), test_size=0.2, random_state=42
)

if os.path.exists(PATH_BOX):
    print(f"  Loading existing weights: {PATH_BOX}")
    model = load_model(PATH_BOX, custom_objects=CUSTOM_BOX)
else:
    print("  No existing weights — building fresh model.")
    model = build_model(
        dropout_cls=best_dropout, dropout_reg=best_dropout, dense_scale=best_scale
    )

# Freeze classification head — bbox branch trains alone.
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = False

# FIX-T3: Cap Stage 1 LR at 1e-4.
STAGE1_LR = min(best_lr, 1e-4)
print(f"  Stage 1 LR = {STAGE1_LR:.2e}  "
      f"(Optuna best={best_lr:.2e}, capped at 1e-4)")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=STAGE1_LR, clipnorm=1.0),
    loss=LOSSES,
    loss_weights={"c_final": 0.0, "b_final": 1.0},
)

box_iou_cb = BoxIoUCallback(X_val, box_val)   # FIX-T4

model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=30, batch_size=8,
    callbacks=make_callbacks(PATH_BOX, monitor="val_b_final_loss",
                             es_patience=10, lr_patience=10,
                             extra=[box_iou_cb]),
    verbose=1,
)
print(f"Stage 1 complete → {PATH_BOX}")


# =====================================================
# STAGE 2 — CLASSIFICATION  (bleeding + non-bleeding)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 2 — Classification (EfficientNet-B0)")
print("=" * 55)

X_train, X_val, box_train, box_val, label_train, label_val = train_test_split(
    *load_data(with_neg=True, aug=False), test_size=0.2, random_state=42
)

if os.path.exists(PATH_CLS):
    print(f"  Loading existing Stage 2 weights: {PATH_CLS}")
    model = load_model(PATH_CLS, custom_objects=CUSTOM_BOX)
else:
    print(f"  Fine-tuning from Stage 1: {PATH_BOX}")
    model = load_model(PATH_BOX, custom_objects=CUSTOM_BOX)

# Step 1 — freeze everything
for layer in model.layers:
    layer.trainable = False

# Step 2 — unfreeze classification head (FIX-T5)
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = True

# Step 3 — partially unfreeze last ~20 % of EfficientNetB0 sub-layers (FIX-T6)
backbone = None
for layer in model.layers:
    if layer.name == BACKBONE_LAYER_NAME:
        backbone = layer
        break

if backbone is not None:
    sub_layers = [l for l in backbone.layers
                  if hasattr(l, "trainable") and len(l.weights) > 0]
    n_unfreeze = max(2, len(sub_layers) // 5)
    for sub in sub_layers[-n_unfreeze:]:
        sub.trainable = True
    print(f"  Backbone: unfroze last {n_unfreeze}/{len(sub_layers)} sub-layers")
else:
    print(f"  WARNING: backbone layer '{BACKBONE_LAYER_NAME}' not found — "
          f"only classification head will be trained.")

# FIX-T5: Conservative LR prevents the wild oscillation seen in Combo 2 old code
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, clipnorm=1.0),
    loss=LOSSES,
    loss_weights={"c_final": 1.0, "b_final": 0.0},
    metrics={"c_final": "accuracy"},
)
model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=30, batch_size=8,
    callbacks=make_callbacks(PATH_CLS, monitor="val_c_final_loss",
                             es_patience=10, lr_patience=10),
    verbose=1,
)
print(f"Stage 2 complete → {PATH_CLS}")


# =====================================================
# STAGE 3 — SEGMENTATION  (Attention U-Net, bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 3 — Segmentation (Attention U-Net)")
print("=" * 55)

X_train, X_val, y_train, y_val = train_test_split(
    *load_data_unet(), test_size=0.2, shuffle=True, random_state=42,
)
y_train = y_train.reshape(-1, 224, 224, 1)
y_val   = y_val.reshape(-1, 224, 224, 1)

if os.path.exists(PATH_SEG):
    print(f"  Loading existing weights: {PATH_SEG}")
    model = load_model(PATH_SEG, custom_objects=SEG_CUSTOM)
else:
    print("  No existing weights — building Attention U-Net.")
    model = Build_AttUnet_Model(num_filters=32)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=focal_tversky,
    metrics=[dice_coef, tf.keras.metrics.BinaryIoU(threshold=0.5)],
)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=8,
    callbacks=make_callbacks(PATH_SEG, monitor="val_dice_coef", mode="max",
                             es_patience=10, lr_patience=10),
    verbose=1,
)
print(f"Stage 3 complete → {PATH_SEG}")


# =====================================================
# STAGE 4 — COMBINED ColonNet (Combo 1)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 4 — Building Combined ColonNet (Combo 1)")
print("=" * 55)

if not (os.path.exists(PATH_CLS) and os.path.exists(PATH_SEG)):
    print("Stage 4 skipped — Stage 2 and/or Stage 3 not yet complete.")
else:
    upper = load_model(PATH_CLS, custom_objects=CUSTOM_BOX)
    lower = load_model(PATH_SEG, custom_objects=SEG_CUSTOM)

    for m in (upper, lower):
        for layer in m.layers:
            layer.trainable = False

    inp = Input(shape=(224, 224, 3), name="combined_input")
    cls_out, box_out = upper(inp)
    seg_out          = lower(inp)

    final = Model(
        inputs=inp,
        outputs={"c_final": cls_out, "b_final": box_out, "seg_output": seg_out},
        name="ColonNet",
    )
    final.save(PATH_COLON)
    print(f"Stage 4 complete → {PATH_COLON}")

print("\nRun complete ✅")
print(f"Models in: {MODELS_DIR}")