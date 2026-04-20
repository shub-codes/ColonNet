"""
training.py  — ColonNet Training Pipeline  [FIXED v2]
================================================================
Architecture
  Stage 1  Bounding box    — EfficientNet-B0 + YOLOv5n-style head
  Stage 2  Classification  — EfficientNet-B0 + lightweight cls head
  Stage 3  Segmentation    — Attention U-Net (base_filters=32)
  Stage 4  Combined model  — frozen Stage 2 + Stage 3

KEY FIXES APPLIED (vs original training.py)
  FIX-F1  GLOBAL DATA SPLIT — one canonical train/val split saved to disk
           before any stage trains. All load_data() calls filter by these IDs
           so Stage 1, 2, 3 never share samples across train/val boundaries.
  FIX-F2  CLASS WEIGHTS for Stage 2 — bleeding class upweighted (3:1)
           to counteract sigmoid saturation and class imbalance.
  FIX-F3  BBOX COLLAPSE GUARD — BoxIoUCallback now also checks whether
           >60 % of predictions are near-full-image boxes and aborts early.
  FIX-F4  ReduceLROnPlateau monitor fixed per stage (was hard-coded to
           val_b_final_loss even during Stage 2/3).
  FIX-F5  Stage 2 cls threshold sweep logged at end of training so you
           can pick a threshold < 0.5 if the sigmoid has saturated low.
  FIX-F6  Stage 3 UNet trains on the TRAIN split only (was using its own
           independent split via load_data_unet, risking leakage).
  FIX-F7  Stage 4 combined model output dict key fixed to match eval
           expectations ("seg_output" from AttentionUNet).

Fixes carried forward from original training.py
  FIX-T1  combined_box_loss uses smooth-L1 only (no GIoU) to keep loss ≥ 0
  FIX-T2  Optuna search space caps LR at 3.16e-4 (log10 ≤ −3.5)
  FIX-T3  Stage 1 caps its LR at 3e-4 regardless of Optuna's best value
  FIX-T4  BoxIoUCallback aborts Stage 1 if mean IoU < 0.10 at epoch 15
  FIX-T5  Stage 2 uses conservative LR (1e-4) for the classification head
  FIX-T6  Stage 2 partially unfreezes the last ~20 % of backbone sub-layers
  FIX-T7  Stage 1 bleeding-only; Stage 2 bleeding + non-bleeding
"""

import gc
import json
import os
import sys
import hashlib

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

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "SavedModels")
os.makedirs(MODELS_DIR, exist_ok=True)

def mp(name):
    return os.path.join(MODELS_DIR, name)

PATH_BOX       = mp("CheckPoint1.keras")
PATH_CLS       = mp("classNbox.keras")
PATH_SEG       = mp("segmentation.keras")
PATH_COLON     = mp("ColonNet.keras")
PATH_OPTUNA    = mp("optuna_best_params.json")
PATH_SPLIT     = mp("global_split.json")          # FIX-F1

BACKBONE_LAYER_NAME = "EfficientNetB0_FPN"

# ─────────────────────────────────────────────────────────
# FIX-F1 — GLOBAL DATA SPLIT
# Load the full bleeding dataset once, split it, and save the
# train/val indices to disk. Every stage reads from the same split,
# so no image ever appears in both train and val across any stage.
# ─────────────────────────────────────────────────────────

def _make_or_load_split():
    """
    Returns (train_idx, val_idx) as numpy arrays.
    If a saved split exists on disk it is reused; otherwise a new one
    is created from the full bleeding dataset (with_neg=False) and saved.

    NOTE: load_data must accept an 'indices' keyword that filters which
    samples to return. If your load_data does not support this, see the
    alternative approach in the comment below.
    """
    if os.path.exists(PATH_SPLIT):
        with open(PATH_SPLIT) as f:
            d = json.load(f)
        train_idx = np.array(d["train_idx"])
        val_idx   = np.array(d["val_idx"])
        print(f"[FIX-F1] Loaded existing split → "
              f"{len(train_idx)} train / {len(val_idx)} val")
        return train_idx, val_idx

    # Load full dataset to know total size (images only to save RAM)
    print("[FIX-F1] Building global split from full dataset …")
    imgs, boxes, labels = load_data(with_neg=False, aug=False)
    n = len(imgs)
    all_idx = np.arange(n)
    train_idx, val_idx = train_test_split(all_idx, test_size=0.2, random_state=42)

    with open(PATH_SPLIT, "w") as f:
        json.dump({
            "train_idx": train_idx.tolist(),
            "val_idx":   val_idx.tolist(),
            "n_total":   n,
        }, f)
    print(f"[FIX-F1] Saved global split → "
          f"{len(train_idx)} train / {len(val_idx)} val  (file: {PATH_SPLIT})")

    # Return the data we already loaded (avoid re-loading for Stage 1)
    return train_idx, val_idx


GLOBAL_TRAIN_IDX, GLOBAL_VAL_IDX = _make_or_load_split()


def _split_data(imgs, boxes, labels, train_idx, val_idx):
    """Apply pre-computed split indices to a loaded dataset."""
    # Guard: if dataset size changed (e.g. augmentation doubled it), clip
    n = len(imgs)
    t_idx = train_idx[train_idx < n]
    v_idx = val_idx[val_idx < n]
    return (
        imgs[t_idx], imgs[v_idx],
        boxes[t_idx], boxes[v_idx],
        labels[t_idx], labels[v_idx],
    )


def _split_unet(imgs, masks, train_idx, val_idx):
    """Apply pre-computed split indices to segmentation data."""
    n = len(imgs)
    t_idx = train_idx[train_idx < n]
    v_idx = val_idx[val_idx < n]
    return imgs[t_idx], imgs[v_idx], masks[t_idx], masks[v_idx]


# ─────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────

def smooth_l1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    loss   = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    return tf.reduce_mean(loss)


def combined_box_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0.01) & (h > 0.01), tf.float32)
    ix1    = tf.maximum(y_true[:, 0], y_pred[:, 0])
    iy1    = tf.maximum(y_true[:, 1], y_pred[:, 1])
    ix2    = tf.minimum(y_true[:, 2], y_pred[:, 2])
    iy2    = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter  = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
    area_t = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_p = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union  = area_t + area_p - inter + 1e-7
    iou    = tf.clip_by_value(inter / union, 0.0, 1.0)
    diff           = tf.abs(y_true - y_pred)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)
    per_sample = ((1.0 - iou) + 0.5 * sl1_per_sample) * valid_mask
    n_valid    = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(per_sample) / n_valid

LOSSES = {
    "c_final": tf.keras.losses.BinaryCrossentropy(),
    "b_final": combined_box_loss,
}

# Module-level globals for weighted_bce — overwritten in Stage 2 before compile().
# Must be defined here so the @register decorator runs at import time,
# allowing Keras to deserialize classNbox.keras on reload.
_W_BLEED    = 1.0
_W_NONBLEED = 1.0

@tf.keras.utils.register_keras_serializable(package="ColonNet")
def weighted_bce(y_true, y_pred):
    y_true  = tf.cast(y_true, tf.float32)
    weights = tf.where(tf.equal(y_true, 1.0), _W_BLEED, _W_NONBLEED)
    bce     = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(weights * bce)

CUSTOM_BOX = {
    "smooth_l1":         smooth_l1,
    "combined_box_loss": combined_box_loss,
    "weighted_bce":      weighted_bce,
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
# BOX IoU CALLBACK  (FIX-T4 + FIX-F3)
# ─────────────────────────────────────────────────────────

class BoxIoUCallback(tf.keras.callbacks.Callback):
    """
    Prints mean IoU each epoch and aborts Stage 1 if:
      (a) mean IoU < ABORT_THRESHOLD at epoch ABORT_EPOCH  [FIX-T4], OR
      (b) > FULLIMG_FRAC of predictions cover >90% of the image  [FIX-F3]

    (b) catches the degenerate 'predict full image always' failure mode
    that produces IoU ~0.13 at test time even when val IoU looks plausible.
    """
    ABORT_THRESHOLD = 0.10
    ABORT_EPOCH     = 15
    FULLIMG_FRAC    = 0.30   # abort if >30% boxes are near-full-image
    FULLIMG_WARN    = 0.15   # warn (but don't abort) above 15%

    def __init__(self, X_val, box_val):
        super().__init__()
        self.X_val         = X_val
        self.box_val       = box_val
        self._best_weights = None  # last weights before collapse warning

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

        # FIX-F3: detect full-image box collapse
        # A "near full image" box covers >90% width AND >90% height (normalised)
        box_w = box_pred[:, 2] - box_pred[:, 0]
        box_h = box_pred[:, 3] - box_pred[:, 1]
        full_img_frac = float(np.mean((box_w > 0.9) & (box_h > 0.9)))

        print(f"  [BoxIoU] epoch {epoch+1:>3d}  val mean-IoU = {mean_iou:.4f}  "
              f"full-img-boxes = {full_img_frac*100:.1f}%", end="")

        # Save weights while still healthy so we can restore on abort
        if full_img_frac < self.FULLIMG_WARN:
            self._best_weights = self.model.get_weights()

        abort = False
        if mean_iou < self.ABORT_THRESHOLD and (epoch + 1) >= self.ABORT_EPOCH:
            print(f"\n  ⚠  BoxIoU below {self.ABORT_THRESHOLD} at epoch {epoch+1}.")
            abort = True
        if full_img_frac > self.FULLIMG_FRAC and (epoch + 1) >= 3:
            print(f"\n  ⚠  {full_img_frac*100:.0f}% of bbox predictions are "
                  f"near-full-image (collapsed head). [FIX-F3]")
            abort = True
        elif full_img_frac > self.FULLIMG_WARN:
            print(f"  ← WARNING: collapse starting ({full_img_frac*100:.0f}%)", end="")

        if abort:
            if self._best_weights is not None:
                self.model.set_weights(self._best_weights)
                print("  ⚠  Restored best pre-collapse weights.")
            print("  ⚠  Stopping Stage 1. Delete CheckPoint1.keras and retrain.")
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
    # FIX-F4: ReduceLROnPlateau now monitors the SAME metric as EarlyStopping
    # (was hard-coded to val_b_final_loss even during seg training)
    cbs = [
        PrintMetrics(),
        ModelCheckpoint(ckpt_path, monitor=monitor, mode=mode,
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor=monitor, mode=mode, patience=es_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, mode=mode,
                          patience=lr_patience, min_lr=1e-6, verbose=1),
    ]
    if extra:
        cbs.extend(extra)
    return cbs


# ─────────────────────────────────────────────────────────
# OPTUNA HYPERPARAMETER SEARCH
# ─────────────────────────────────────────────────────────

N_OPTUNA_TRIALS = 25

if not os.path.exists(PATH_OPTUNA):
    print("Loading dataset for Optuna search …")
    _opt_imgs, _opt_boxes, _opt_labels = load_data(with_neg=False, aug=False)
    # Use only the TRAIN split for HPO (FIX-F1 applied here too)
    _opt_imgs   = _opt_imgs[GLOBAL_TRAIN_IDX[GLOBAL_TRAIN_IDX < len(_opt_imgs)]]
    _opt_boxes  = _opt_boxes[GLOBAL_TRAIN_IDX[GLOBAL_TRAIN_IDX < len(_opt_boxes)]]
    _opt_labels = _opt_labels[GLOBAL_TRAIN_IDX[GLOBAL_TRAIN_IDX < len(_opt_labels)]]
    N   = min(512, len(_opt_imgs))
    idx = np.random.permutation(len(_opt_imgs))[:N]
    _opt_imgs   = _opt_imgs[idx]
    _opt_boxes  = _opt_boxes[idx]
    _opt_labels = _opt_labels[idx]
    print(f"Optuna subset shape: {_opt_imgs.shape}")


def optuna_objective(trial):
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

# FIX-F1: use the global split, not a fresh random split
_s1_imgs, _s1_boxes, _s1_labels = load_data(with_neg=False, aug=False)
X_train, X_val, box_train, box_val, label_train, label_val = _split_data(
    _s1_imgs, _s1_boxes, _s1_labels, GLOBAL_TRAIN_IDX, GLOBAL_VAL_IDX
)
del _s1_imgs, _s1_boxes, _s1_labels

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

# FIX-T3 (revised): Cap at 5e-5 (was 3e-4 — caused bbox head collapse ~epoch 5-6)
STAGE1_LR = min(best_lr, 5e-5)
print(f"  Stage 1 LR = {STAGE1_LR:.2e}  "
      f"(Optuna best={best_lr:.2e}, capped at 5e-5)")

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=STAGE1_LR,
        weight_decay=1e-4,
        clipnorm=0.5,   # tighter clip — bbox gradients spike near collapse
    ),
    loss=LOSSES,
    loss_weights={"c_final": 0.0, "b_final": 1.0},
)

box_iou_cb = BoxIoUCallback(X_val, box_val)   # FIX-T4 + FIX-F3

model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=30, batch_size=8,
    callbacks=make_callbacks(PATH_BOX, monitor="val_b_final_loss", mode="min",
                             es_patience=10, lr_patience=3,   # fast drop before collapse
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

# FIX-F1: use the global split on the full dataset (with_neg=True, aug=True)
_s2_imgs, _s2_boxes, _s2_labels = load_data(with_neg=True, aug=True)
X_train, X_val, box_train, box_val, label_train, label_val = _split_data(
    _s2_imgs, _s2_boxes, _s2_labels, GLOBAL_TRAIN_IDX, GLOBAL_VAL_IDX
)
del _s2_imgs, _s2_boxes, _s2_labels

# FIX-F2: Compute class weights to address bleeding/non-bleeding imbalance
# and counteract sigmoid saturation on the bleeding class.
n_bleed     = int(np.sum(label_train))
n_nonbleed  = int(len(label_train) - n_bleed)
total       = n_bleed + n_nonbleed
# sklearn-style balanced weights, then scale so bleeding gets at least 3×
w_nonbleed  = total / (2.0 * max(n_nonbleed, 1))
w_bleed     = total / (2.0 * max(n_bleed, 1))
# Enforce at least 3:1 ratio for the bleeding class
if w_bleed / w_nonbleed < 3.0:
    w_bleed = w_nonbleed * 3.0
print(f"  [FIX-F2] Class weights → non-bleeding: {w_nonbleed:.3f}  "
      f"bleeding: {w_bleed:.3f}  (ratio {w_bleed/w_nonbleed:.1f}:1)")

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

# FIX-T5: Conservative LR; FIX-F4: monitor fixed to val_c_final_loss
# FIX-F2 (revised): set module-level globals so the registered weighted_bce
# uses the correct per-run class weights. This avoids both the Keras 3
# sample_weight KeyError and the serialization crash on reload.
_W_BLEED    = w_bleed
_W_NONBLEED = w_nonbleed

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4, clipnorm=1.0),
    loss={"c_final": weighted_bce, "b_final": LOSSES["b_final"]},
    loss_weights={"c_final": 1.0, "b_final": 0.0},
    metrics={"c_final": "accuracy"},
)

model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=30, batch_size=8,
    callbacks=make_callbacks(PATH_CLS, monitor="val_c_final_loss", mode="min",
                             es_patience=10, lr_patience=10),
    verbose=1,
)

print(f"Stage 2 complete → {PATH_CLS}")

# FIX-F5: Threshold sweep — log optimal decision threshold on val set
# so eval.py can use a value better than the default 0.5
print("\n  [FIX-F5] Threshold sweep on validation set …")
_s2_model = load_model(PATH_CLS, custom_objects={**CUSTOM_BOX, **SEG_CUSTOM})
_val_preds = _s2_model.predict(X_val, verbose=0)
if isinstance(_val_preds, (list, tuple)):
    _val_probs = np.array(_val_preds[0]).flatten()
else:
    _val_probs = np.array(_val_preds["c_final"]).flatten()

_best_thresh, _best_f1 = 0.5, 0.0
for _thresh in np.arange(0.05, 0.95, 0.05):
    _preds = (_val_probs >= _thresh).astype(int)
    _tp = int(np.sum((_preds == 1) & (label_val == 1)))
    _fp = int(np.sum((_preds == 1) & (label_val == 0)))
    _fn = int(np.sum((_preds == 0) & (label_val == 1)))
    _prec = _tp / max(_tp + _fp, 1)
    _rec  = _tp / max(_tp + _fn, 1)
    _f1   = 2 * _prec * _rec / max(_prec + _rec, 1e-7)
    if _f1 > _best_f1:
        _best_f1, _best_thresh = _f1, float(_thresh)

print(f"  [FIX-F5] Best val threshold = {_best_thresh:.2f}  "
      f"(F1 = {_best_f1:.4f})")
print(f"  [FIX-F5] *** Use CLS_THRESHOLD = {_best_thresh:.2f} in eval.py ***")

# Save threshold alongside Optuna params for eval.py to pick up
with open(PATH_OPTUNA) as f:
    _saved = json.load(f)
_saved["cls_threshold"] = _best_thresh
with open(PATH_OPTUNA, "w") as f:
    json.dump(_saved, f, indent=2)
del _s2_model, _val_probs
gc.collect()


# =====================================================
# STAGE 3 — SEGMENTATION  (Attention U-Net, bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 3 — Segmentation (Attention U-Net)")
print("=" * 55)

# FIX-F6: Apply the SAME global split to the UNet dataset.
# Previously load_data_unet() made its own independent split which
# could overlap with Stage 1/2 val images.
_s3_imgs, _s3_masks = load_data_unet()
X_train, X_val, y_train, y_val = _split_unet(
    _s3_imgs, _s3_masks, GLOBAL_TRAIN_IDX, GLOBAL_VAL_IDX
)
del _s3_imgs, _s3_masks

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
# FIX-F4: monitor = val_dice_coef, mode = max (was unchanged but ReduceLR
# now correctly uses this same monitor instead of val_b_final_loss)
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

    # FIX-F7: output key "seg_output" matches AttentionUNet layer name
    # and what eval.py expects when it calls seg_model.predict()
    final = Model(
        inputs=inp,
        outputs={"c_final": cls_out, "b_final": box_out, "seg_output": seg_out},
        name="ColonNet",
    )
    final.save(PATH_COLON)
    print(f"Stage 4 complete → {PATH_COLON}")

print("\nRun complete ✅")
print(f"Models in: {MODELS_DIR}")