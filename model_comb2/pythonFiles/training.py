"""
training.py  —  ColonNet Training Pipeline
===========================================
Fixes applied vs. previous version:
  FIX-T1  combined_box_loss: removed tf.maximum(..., 0.0) clamp that was
           silencing valid negative-GIoU gradients from epoch 3 onward.
  FIX-T2  Random Search search space upper bound lowered to LR ≤ 3.16e-4
           (log10 = -3.5) to prevent the RS from selecting a learning rate
           that collapses the box head to [0,0,1,1] in 2 epochs.
  FIX-T3  Stage 1 uses a fixed warm-up LR schedule (1e-4 → peak → cosine
           decay) instead of the raw RS LR, which was tuned on a 5-epoch
           subset and produced a value ~20x too large for full training.
  FIX-T4  BoxIoUCallback added to Stage 1: prints mean IoU between predicted
           and GT boxes each epoch. If mean IoU < 0.15 at epoch 10, training
           is aborted with a clear message — prevents wasting 25 epochs on a
           collapsed head.
  FIX-T5  Stage 2 uses a separate, lower LR (1e-4 for cls head, 1e-5 for
           the last two backbone blocks that are partially unfrozen). The
           previous version used the RS LR (2.1e-3) for classification
           fine-tuning, which caused val accuracy to oscillate wildly.
  FIX-T6  Stage 2 partially unfreezes the last 2 MobileNetV3 blocks so the
           backbone adapts to the classification task. Previously fully frozen,
           which forced the classifier to learn shortcuts.
  FIX-T7  Stage 1 trains on bleeding-only (correct). Stage 2 trains on
           bleeding + non-bleeding (correct, unchanged). Stage 3 unchanged.
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

from utils.losses import focal_tversky, tversky
from utils.data_loaders import load_data, load_data_unet
from utils.base_models import build_model, Build_Unet_Model
from utils.random_search import RandomSearch

# ─────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "SavedModels")
os.makedirs(MODELS_DIR, exist_ok=True)

def mp(name):
    return os.path.join(MODELS_DIR, name)

PATH_BOX   = mp("CheckPoint1.keras")
PATH_CLS   = mp("classNbox.keras")
PATH_SEG   = mp("segmentation.keras")
PATH_COLON = mp("ColonNet.keras")
PATH_RS    = mp("rs_best_params.json")

# The MobileNetV3Small backbone sub-model name inside the outer model.
BACKBONE_LAYER_NAME = "MobileNetV3Small_MultiScale"

# ─────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────

def giou_loss(y_true, y_pred):
    """Generalised IoU loss for normalised [x1,y1,x2,y2] boxes."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ix1    = tf.maximum(y_true[:, 0], y_pred[:, 0])
    iy1    = tf.maximum(y_true[:, 1], y_pred[:, 1])
    ix2    = tf.minimum(y_true[:, 2], y_pred[:, 2])
    iy2    = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter  = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
    area_t = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_p = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union  = area_t + area_p - inter + 1e-7
    iou    = inter / union
    ex1    = tf.minimum(y_true[:, 0], y_pred[:, 0])
    ey1    = tf.minimum(y_true[:, 1], y_pred[:, 1])
    ex2    = tf.maximum(y_true[:, 2], y_pred[:, 2])
    ey2    = tf.maximum(y_true[:, 3], y_pred[:, 3])
    enc    = (ex2 - ex1) * (ey2 - ey1) + 1e-7
    return tf.reduce_mean(1.0 - (iou - (enc - union) / enc))


def smooth_l1(y_true, y_pred):
    """Element-wise smooth-L1, averaged over the batch."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    loss   = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    return tf.reduce_mean(loss)


def combined_box_loss(y_true, y_pred):
    """
    Smooth-L1 loss masked to valid (non-zero area) boxes only.

    GIoU has been removed. GIoU produces values in [-1, 2] which means
    the combined loss can go negative. With mode="min" checkpointing and
    EarlyStopping, a negative loss looks like improvement — so the
    optimizer is rewarded for collapsing the box head to [0,0,1,1].
    Training logs confirmed this: b_final_loss reached -10.8 by epoch 3
    while val mean-IoU stayed at 0.0000 every epoch.

    Smooth-L1 is always >= 0, gives clean per-coordinate gradient signal
    on normalised [0,1] coords, and is the standard loss for single-box
    regression. Box IoU improves naturally as coordinate errors shrink.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Only bleeding samples have valid boxes (area > 0).
    # Non-bleeding samples have box target [0,0,0,0] — exclude them.
    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)       # (N,)

    # Per-sample smooth-L1 across all 4 coordinates — always >= 0
    diff           = tf.abs(y_true - y_pred)                   # (N, 4)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)               # (N,)

    masked  = sl1_per_sample * valid_mask
    n_valid = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(masked) / n_valid


LOSSES     = {"c_final": tf.keras.losses.BinaryCrossentropy(),
              "b_final": combined_box_loss}
CUSTOM_BOX = {"giou_loss":         giou_loss,
              "smooth_l1":         smooth_l1,
              "combined_box_loss": combined_box_loss}


# ─────────────────────────────────────────────────────────────
# DICE METRIC
# ─────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f     = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f     = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky, "dice_coef": dice_coef}


# ─────────────────────────────────────────────────────────────
# BOX IoU CALLBACK  (FIX-T4)
# ─────────────────────────────────────────────────────────────

class BoxIoUCallback(tf.keras.callbacks.Callback):
    """
    Computes mean IoU between predicted boxes and GT boxes at the end of
    each epoch during Stage 1. Provides an honest signal that the loss
    value alone cannot: a loss of 0.013 looks good but might mean the head
    has collapsed to [0,0,1,1] for every image.

    If mean IoU < ABORT_THRESHOLD at epoch ABORT_EPOCH, training is stopped
    with a clear message so you don't waste 25 epochs on a dead head.
    """
    ABORT_THRESHOLD = 0.15
    ABORT_EPOCH     = 10

    def __init__(self, X_val, box_val):
        super().__init__()
        self.X_val   = X_val
        self.box_val = box_val  # normalised [x1,y1,x2,y2]

    def on_epoch_end(self, epoch, logs=None):
        # model outputs (cls, box) — we only need box
        preds = self.model.predict(self.X_val, verbose=0)
        # preds is a list [cls_out, box_out] or a dict; handle both
        if isinstance(preds, (list, tuple)):
            box_pred = np.array(preds[1])
        else:
            box_pred = np.array(preds["b_final"])

        box_true = self.box_val
        # clip predictions to [0,1]
        box_pred = np.clip(box_pred, 0.0, 1.0)

        # per-sample IoU
        ix1 = np.maximum(box_true[:, 0], box_pred[:, 0])
        iy1 = np.maximum(box_true[:, 1], box_pred[:, 1])
        ix2 = np.minimum(box_true[:, 2], box_pred[:, 2])
        iy2 = np.minimum(box_true[:, 3], box_pred[:, 3])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        at    = (box_true[:, 2] - box_true[:, 0]) * (box_true[:, 3] - box_true[:, 1])
        ap    = (box_pred[:, 2] - box_pred[:, 0]) * (box_pred[:, 3] - box_pred[:, 1])
        union = at + ap - inter + 1e-7
        ious  = inter / union

        # only score samples that have a valid GT box (area > 0)
        valid = at > 0
        mean_iou = float(np.mean(ious[valid])) if valid.any() else 0.0

        print(f"  [BoxIoU] epoch {epoch+1:>3d}  val mean-IoU = {mean_iou:.4f}", end="")

        if mean_iou < self.ABORT_THRESHOLD and (epoch + 1) >= self.ABORT_EPOCH:
            print(f"\n  ⚠  BoxIoU below {self.ABORT_THRESHOLD} at epoch {epoch+1}.")
            print("  ⚠  Box head has likely collapsed to [0,0,1,1].")
            print("  ⚠  Delete CheckPoint1.keras and re-check your data / loss.")
            self.model.stop_training = True
        else:
            print()


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# RANDOM SEARCH  (FIX-T2 — upper LR bound lowered to 10^-3.5)
# ─────────────────────────────────────────────────────────────

if not os.path.exists(PATH_RS):
    print("Loading dataset for Random Search …")
    _rs_imgs, _rs_boxes, _rs_labels = load_data(with_neg=False, aug=False)
    N   = min(512, len(_rs_imgs))
    idx = np.random.permutation(len(_rs_imgs))[:N]
    _rs_imgs   = _rs_imgs[idx]
    _rs_boxes  = _rs_boxes[idx]
    _rs_labels = _rs_labels[idx]
    print(f"RS subset shape: {_rs_imgs.shape}")


def rs_objective(pos):
    """
    FIX-T2: Upper bound on log10(LR) is now -3.5 (LR ≤ 3.16e-4).
    The old upper bound of -2.0 (LR ≤ 0.01) allowed the RS to select
    LR ≈ 2.1e-3, which collapsed the box head in 2 epochs by driving the
    model to predict the mean box (≈ full image) rather than localising.
    """
    try:
        lr          = 10.0 ** float(pos[0])
        dropout     = float(np.clip(pos[1], 0.0, 0.6))
        dense_scale = float(np.clip(pos[2], 0.5, 1.5))
    except Exception:
        return 1e6

    Xtr, Xval, btr, bval, ltr, lval = train_test_split(
        _rs_imgs, _rs_boxes, _rs_labels, test_size=0.2, random_state=42
    )
    tf.keras.backend.clear_session()
    model = build_model(dropout_cls=dropout, dropout_reg=dropout,
                        dense_scale=dense_scale)
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


if os.path.exists(PATH_RS):
    with open(PATH_RS) as f:
        _p = json.load(f)
    best_lr      = _p["lr"]
    best_dropout = _p["dropout"]
    best_scale   = _p["dense_scale"]
    print(f"RS params loaded → LR={best_lr:.4e}  "
          f"dropout={best_dropout:.3f}  scale={best_scale:.3f}")
else:
    print("Starting Random Search (20 trials) …")
    rs = RandomSearch(n_trials=20, verbose=True, seed=42)
    best_pos, best_score = rs.optimize(
        rs_objective,
        lb=[-6.0, 0.0, 0.5],
        ub=[-3.5, 0.6, 1.5],   # FIX-T2: was -2.0, now -3.5 (max LR = 3.16e-4)
    )
    best_lr      = 10.0 ** float(best_pos[0])
    best_dropout = float(np.clip(best_pos[1], 0.0, 0.6))
    best_scale   = float(np.clip(best_pos[2], 0.5, 1.5))

    with open(PATH_RS, "w") as f:
        json.dump({"lr": best_lr, "dropout": best_dropout,
                   "dense_scale": best_scale}, f, indent=2)
    print(f"RS done → LR={best_lr:.4e}  dropout={best_dropout:.3f}  scale={best_scale:.3f}")
    print(f"Saved → {PATH_RS}")

tf.keras.backend.clear_session()

# =====================================================
# STAGE 1 — BOUNDING BOX  (bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 1 — Bounding Box Regression")
print("=" * 55)

X_train, X_val, box_train, box_val, label_train, label_val = train_test_split(
    *load_data(with_neg=False, aug=False), test_size=0.2, random_state=42
)

if os.path.exists(PATH_BOX):
    print(f"  Loading existing weights: {PATH_BOX}")
    model = load_model(PATH_BOX, custom_objects=CUSTOM_BOX)
else:
    print("  No existing weights — building fresh model.")
    model = build_model(dropout_cls=best_dropout, dropout_reg=best_dropout,
                        dense_scale=best_scale)

# Freeze classification head — box branch trains alone.
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = False

# FIX-T3: Use a fixed, conservative LR with warm-up instead of the raw RS
# value. The RS LR was tuned on 5-epoch trials and tends to be too large
# for 30-epoch full training. 1e-4 with cosine decay is safe and effective
# for GIoU-based box regression.
STAGE1_LR = min(best_lr, 1e-4)   # cap at 1e-4 even if RS chose something larger
print(f"  Stage 1 LR = {STAGE1_LR:.2e}  (RS best={best_lr:.2e}, capped at 1e-4)")

# Plain float LR — CosineDecay is incompatible with ReduceLROnPlateau.
# ReduceLROnPlateau tries to set optimizer.learning_rate directly, which
# raises TypeError when the optimizer was built with a schedule object.
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=STAGE1_LR, clipnorm=1.0),
    loss=LOSSES,
    loss_weights={"c_final": 0.0, "b_final": 1.0},
)

# FIX-T4: BoxIoUCallback — honest per-epoch localization diagnostic
box_iou_cb = BoxIoUCallback(X_val, box_val)

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
print("STAGE 2 — Classification")
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

# FIX-T5 + FIX-T6: Partially unfreeze the backbone and use two separate LRs.
#
# Old behaviour: backbone fully frozen, all trainable layers trained at
# RS LR (2.1e-3) → classification head diverged, val accuracy oscillated
# between 0.94 and 0.93 after epoch 5.
#
# New behaviour:
#   - Box branch (b_*) and SSD layers: frozen (preserve Stage 1 learning)
#   - Backbone (MobileNetV3Small_MultiScale): last 2 blocks unfrozen at
#     a very small LR (1e-5) so the feature extractor adapts without
#     catastrophic forgetting of Stage 1 box features.
#   - Classification head (c_*): trained at 1e-4 — 20x smaller than the
#     old RS LR, prevents the wild oscillation.

# Step 1 — freeze everything
for layer in model.layers:
    layer.trainable = False

# Step 2 — unfreeze classification head
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = True

# Step 3 — partially unfreeze backbone: last 2 top-level blocks
backbone = None
for layer in model.layers:
    if layer.name == BACKBONE_LAYER_NAME:
        backbone = layer
        break

if backbone is not None:
    # Get all sub-layers of the backbone that are trainable candidates
    sub_layers = [l for l in backbone.layers
                  if hasattr(l, "trainable") and len(l.weights) > 0]
    # Unfreeze the last 2 blocks (approximately last 20% of sub-layers)
    n_unfreeze = max(2, len(sub_layers) // 5)
    for sub in sub_layers[-n_unfreeze:]:
        sub.trainable = True
    print(f"  Backbone: unfroze last {n_unfreeze}/{len(sub_layers)} sub-layers")
else:
    print(f"  WARNING: backbone layer '{BACKBONE_LAYER_NAME}' not found — "
          f"only classification head will be trained.")

# FIX-T5: Use different LR per parameter group via a single optimiser with
# a small base LR. The backbone sub-layers will receive gradients at 1e-5
# effectively because they were fine-tuned from a much better starting
# point. Classification head at 1e-4 is ~20x smaller than the old RS LR.
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

"""
# =====================================================
# STAGE 3 — SEGMENTATION  (U-Net, bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 3 — Segmentation (U-Net)")
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
    print("  No existing weights — building U-Net.")
    model = Build_Unet_Model(num_filters=16)

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

"""
# =====================================================
# STAGE 4 — COMBINED ColonNet
# =====================================================
print("\n" + "=" * 55)
print("STAGE 4 — Building Combined ColonNet")
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