"""
training.py  —  ColonNet Training Pipeline  (Combination 3)
=============================================================
Backbone: EfficientNet-B0  |  Segmentation: UNet++  |  Box: YOLO detached
Hyperparameter search: Hyperband via keras-tuner

Fixes applied:
  FIX-1  combined_box_loss: replaced GIoU+smooth-L1 with smooth-L1 only.
         GIoU produces values in [-1,2] → total loss goes negative →
         mode="min" checkpointing treats the most-collapsed model as best.
         Confirmed by training logs: b_final_loss reached -10.8 by epoch 3
         while val mean-IoU stayed 0.0000 every epoch.
         Smooth-L1 is always >= 0, monotonically correct.

  FIX-2  training.py now uses hyperband_tuner (Combo 3's intended tuner)
         instead of RandomSearch (Combo 2's tuner). The hyperband cache
         file is separate (hyperband_params.json) so it does not conflict.

  FIX-3  build_model called with dense_units= (not dense_scale=).
         base_models.py build_model signature uses dense_units.
         Previously dense_scale= caused TypeError on every RS/Hyperband trial.

  FIX-4  Stage 2 backbone freeze uses correct EfficientNetB0 layer name
         "efficientnetb0" (lower-case, as Keras registers it) not "densenet121".

  FIX-5  Stage 1 LR capped at 1e-4 (plain float, no CosineDecay).
         CosineDecay is incompatible with ReduceLROnPlateau — raises TypeError
         when ReduceLROnPlateau fires.

  FIX-6  BoxIoUCallback added: aborts Stage 1 at epoch 10 if mean-IoU < 0.15,
         preventing 25 wasted epochs on a collapsed box head.
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
from utils.hyperband_tuner import run_hyperband

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

PATH_BOX      = mp("CheckPoint1.keras")
PATH_CLS      = mp("classNbox.keras")
PATH_SEG      = mp("segmentation.keras")
PATH_COLON    = mp("ColonNet.keras")
PATH_HB       = mp("hyperband_params.json")
TUNER_DIR     = mp("hyperband_tuner_logs")

# EfficientNetB0 is wrapped directly (not as a sub-model), so the layer
# name inside the outer model is "efficientnetb0" (Keras lower-cases it).
BACKBONE_LAYER_NAME = "efficientnetb0"

# ─────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────

def smooth_l1(y_true, y_pred):
    """Element-wise smooth-L1, averaged over the batch."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))


def combined_box_loss(y_true, y_pred):
    """
    Smooth-L1 loss masked to valid (non-zero area) boxes only.
    Always >= 0 — safe for mode='min' checkpointing and EarlyStopping.

    GIoU has been removed (FIX-1). GIoU produces values in [-1, 2],
    making the total loss negative. mode='min' then treats the
    most-collapsed [0,0,1,1] prediction as the best model.
    Smooth-L1 gives correct per-coordinate gradient signal and
    is the standard loss for single-box regression on [0,1] coords.
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


# giou_loss kept only for load_model() deserialisation of old checkpoints
def giou_loss(y_true, y_pred):
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
# BOX IoU CALLBACK  (FIX-6)
# ─────────────────────────────────────────────────────────────

class BoxIoUCallback(tf.keras.callbacks.Callback):
    """
    Computes val mean-IoU each epoch during Stage 1.
    Aborts training at epoch ABORT_EPOCH if mean-IoU < ABORT_THRESHOLD,
    preventing wasted epochs on a collapsed box head.
    """
    ABORT_THRESHOLD = 0.15
    ABORT_EPOCH     = 10

    def __init__(self, X_val, box_val):
        super().__init__()
        self.X_val   = X_val
        self.box_val = box_val

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        box_pred = np.array(preds[1] if isinstance(preds, (list, tuple))
                            else preds["b_final"])
        box_pred = np.clip(box_pred, 0.0, 1.0)
        box_true = self.box_val

        ix1  = np.maximum(box_true[:, 0], box_pred[:, 0])
        iy1  = np.maximum(box_true[:, 1], box_pred[:, 1])
        ix2  = np.minimum(box_true[:, 2], box_pred[:, 2])
        iy2  = np.minimum(box_true[:, 3], box_pred[:, 3])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        at    = (box_true[:, 2] - box_true[:, 0]) * (box_true[:, 3] - box_true[:, 1])
        ap    = (box_pred[:, 2] - box_pred[:, 0]) * (box_pred[:, 3] - box_pred[:, 1])
        union = at + ap - inter + 1e-7
        ious  = inter / union
        valid = at > 0
        mean_iou = float(np.mean(ious[valid])) if valid.any() else 0.0

        print(f"  [BoxIoU] epoch {epoch+1:>3d}  val mean-IoU = {mean_iou:.4f}", end="")
        if mean_iou < self.ABORT_THRESHOLD and (epoch + 1) >= self.ABORT_EPOCH:
            print(f"\n  ⚠  Box head collapsed — delete CheckPoint1.keras and retrain.")
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
                   es_patience=10, lr_patience=5, extra=None):
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
# HYPERBAND TUNING  (FIX-2)
# ─────────────────────────────────────────────────────────────

if os.path.exists(PATH_HB):
    with open(PATH_HB) as f:
        _p = json.load(f)
    best_lr      = _p["learning_rate"]
    best_dropout = _p["dropout"]
    best_units   = _p["dense_units"]
    print(f"Hyperband params loaded → LR={best_lr:.4e}  "
          f"dropout={best_dropout:.2f}  dense_units={best_units}")
else:
    print("Loading dataset for Hyperband search …")
    _hb_imgs, _hb_boxes, _hb_labels = load_data(with_neg=False, aug=False)
    N   = min(512, len(_hb_imgs))
    idx = np.random.permutation(len(_hb_imgs))[:N]
    _hb_imgs   = _hb_imgs[idx]
    _hb_boxes  = _hb_boxes[idx]
    _hb_labels = _hb_labels[idx]
    print(f"Hyperband subset: {_hb_imgs.shape}")

    best_hp = run_hyperband(
        _hb_imgs, _hb_labels, _hb_boxes,
        tuner_dir  = TUNER_DIR,
        max_epochs = 10,
        factor     = 3,
        batch_size = 16,
        seed       = 42,
        overwrite  = False,
    )
    best_lr      = best_hp["learning_rate"]
    best_dropout = best_hp["dropout"]
    best_units   = best_hp["dense_units"]

    with open(PATH_HB, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"Hyperband done → LR={best_lr:.4e}  "
          f"dropout={best_dropout:.2f}  dense_units={best_units}")

# FIX-5: cap LR at 1e-4 for full 30-epoch Stage 1 training.
# Hyperband tunes on short trials; its LR may be too large for full training.
STAGE1_LR = min(best_lr, 1e-4)
print(f"Stage 1 LR = {STAGE1_LR:.2e}  (Hyperband best={best_lr:.2e}, capped at 1e-4)")

tf.keras.backend.clear_session()

"""
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
                        dense_units=best_units)

# Freeze classification head — box branch trains alone.
# loss_weight=0 does NOT stop gradients; trainable=False does.
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = False

# FIX-5: plain float LR — CosineDecay is incompatible with ReduceLROnPlateau
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=STAGE1_LR, clipnorm=1.0),
    loss=LOSSES,
    loss_weights={"c_final": 0.0, "b_final": 1.0},
)

box_iou_cb = BoxIoUCallback(X_val, box_val)

model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=30, batch_size=8,
    callbacks=make_callbacks(PATH_BOX, monitor="val_b_final_loss",
                             es_patience=10, lr_patience=5,
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

# FIX-4: correct backbone layer name for EfficientNetB0.
# Original used "densenet121" — wrong for this combination.
# Also freeze box branch (b_*, b_dw*, b_gap, b_dense*, b_drop*, b_pw*).
# Step 1 — freeze everything
for layer in model.layers:
    layer.trainable = False

# Step 2 — unfreeze classification head only
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = True

# Step 3 — partially unfreeze backbone (last ~20% of weighted sub-layers)
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
    print(f"  WARNING: backbone '{BACKBONE_LAYER_NAME}' not found — "
          f"only c_ head trains.")

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
                             es_patience=10, lr_patience=5),
    verbose=1,
)
print(f"Stage 2 complete → {PATH_CLS}")

"""
# =====================================================
# STAGE 3 — SEGMENTATION  (UNet++, bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 3 — Segmentation (UNet++)")
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
    print("  No existing weights — building UNet++.")
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
                             es_patience=10, lr_patience=5),
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

    final = Model(
        inputs=inp,
        outputs={"c_final": cls_out, "b_final": box_out, "seg_output": seg_out},
        name="ColonNet",
    )
    final.save(PATH_COLON)
    print(f"Stage 4 complete → {PATH_COLON}")

print("\nRun complete ✅")
print(f"Models in: {MODELS_DIR}")