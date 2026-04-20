"""
training.py  —  ColonNet Training Pipeline  (Combination 3 — FIXED v3)
=======================================================================
Backbone: EfficientNet-B0  |  Segmentation: UNet++  |  Tuner: Hyperband

FIX v3: Removed two-phase loss switch entirely.
  BoxLossSwitchCallback called model.compile() inside on_epoch_end which
  destroys train_function in this Keras version → TypeError: NoneType.
  Solution: use combined_box_loss (IoU + smooth-L1) from epoch 1.

All previous fixes retained:
  FIX-B2  Backbone trainable=True during Stage 1.
  FIX-B3  Box head uses dense_units (not dense_units//2).
  FIX-B4  LR cap at 3e-4.
  FIX-B5  BoxIoUCallback abort thresholds relaxed.
  FIX-1   combined_box_loss: IoU+smooth-L1 always >= 0.
  FIX-2   Hyperband tuner.
  FIX-3   build_model called with dense_units=.
  FIX-4   Stage 2 backbone layer name "efficientnetb0".
  FIX-5   ReduceLROnPlateau only in Stage 2+3.
  FIX-6   mixed_float16 removed.
  FIX-7   class_weight added to Stage 2.
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
# FIX-6: mixed_float16 removed
# ─────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
PATH_HB    = mp("hyperband_params.json")
TUNER_DIR  = mp("hyperband_tuner_logs")

BACKBONE_LAYER_NAME = "efficientnetb0"

# ─────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────

def smooth_l1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))


def masked_smooth_l1(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)
    diff           = tf.abs(y_true - y_pred)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)
    per_sample     = sl1_per_sample * valid_mask
    n_valid        = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(per_sample) / n_valid


def combined_box_loss(y_true, y_pred):
    """IoU loss + 0.5 * smooth-L1, masked to valid boxes. Range [0, 1.5]."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)
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


def giou_loss(y_true, y_pred):
    # kept only for load_model() deserialisation of old checkpoints
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


LOSSES_MAIN = {"c_final": tf.keras.losses.BinaryCrossentropy(),
               "b_final": combined_box_loss}
CUSTOM_BOX  = {"giou_loss":         giou_loss,
               "smooth_l1":         smooth_l1,
               "masked_smooth_l1":  masked_smooth_l1,
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

SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky,
              "dice_coef": dice_coef}


# ─────────────────────────────────────────────────────────────
# BOX IoU CALLBACK
# ─────────────────────────────────────────────────────────────

class BoxIoUCallback(tf.keras.callbacks.Callback):
    ABORT_THRESHOLD = 0.08
    ABORT_EPOCH     = 15

    def __init__(self, X_val, box_val):
        super().__init__()
        self.X_val   = X_val
        self.box_val = box_val

    def _mean_iou(self, preds):
        b_pred = preds[1]
        b_true = self.box_val
        ix1    = np.maximum(b_true[:, 0], b_pred[:, 0])
        iy1    = np.maximum(b_true[:, 1], b_pred[:, 1])
        ix2    = np.minimum(b_true[:, 2], b_pred[:, 2])
        iy2    = np.minimum(b_true[:, 3], b_pred[:, 3])
        inter  = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        at     = ((b_true[:, 2] - b_true[:, 0]) *
                  (b_true[:, 3] - b_true[:, 1]))
        ap     = ((b_pred[:, 2] - b_pred[:, 0]) *
                  (b_pred[:, 3] - b_pred[:, 1]))
        union  = at + ap - inter + 1e-7
        iou    = np.clip(inter / union, 0.0, 1.0)
        valid  = (at > 0).astype(float)
        n      = np.maximum(valid.sum(), 1.0)
        return float((iou * valid).sum() / n)

    def on_epoch_end(self, epoch, logs=None):
        preds    = self.model.predict(self.X_val, verbose=0)
        mean_iou = self._mean_iou(preds)
        print(f"  [BoxIoU] epoch {epoch+1}: val mean-IoU = {mean_iou:.4f}")
        if logs is not None:
            logs["val_box_iou"] = mean_iou
        if epoch + 1 == self.ABORT_EPOCH and mean_iou < self.ABORT_THRESHOLD:
            print(f"\n  [BoxIoU] ABORT: mean-IoU {mean_iou:.4f} < "
                  f"{self.ABORT_THRESHOLD} at epoch {self.ABORT_EPOCH}.")
            self.model.stop_training = True


# ─────────────────────────────────────────────────────────────
# CALLBACK FACTORY
# ─────────────────────────────────────────────────────────────

def make_callbacks(path, monitor="val_loss", mode="min",
                   es_patience=10, lr_patience=5, extra=None,
                   use_reduce_lr=True):
    cbs = [
        EarlyStopping(monitor=monitor, patience=es_patience,
                      restore_best_weights=True, mode=mode),
        ModelCheckpoint(path, monitor=monitor, save_best_only=True, mode=mode),
    ]
    if use_reduce_lr:
        cbs.append(ReduceLROnPlateau(monitor=monitor, factor=0.5,
                                     patience=lr_patience, min_lr=1e-6,
                                     mode=mode))
    if extra:
        cbs.extend(extra)
    return cbs


# ─────────────────────────────────────────────────────────────
# HYPERBAND SEARCH
# ─────────────────────────────────────────────────────────────

if os.path.exists(PATH_HB):
    print(f"Loading cached Hyperband params: {PATH_HB}")
    with open(PATH_HB) as f:
        best_hp = json.load(f)
    best_lr      = best_hp["learning_rate"]
    best_dropout = best_hp["dropout"]
    best_units   = best_hp["dense_units"]
    print(f"  LR={best_lr:.4e}  dropout={best_dropout:.2f}  "
          f"dense_units={best_units}")
else:
    print("Running Hyperband search …")
    images, boxes, labels = load_data(with_neg=False, aug=False)
    np.random.seed(42)
    n   = len(images)
    idx = np.random.choice(n, min(n, 500), replace=False)
    _hb_imgs   = images[idx]
    _hb_boxes  = boxes[idx]
    _hb_labels = labels[idx]
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

STAGE1_LR    = min(best_lr, 3e-4)
STAGE1_TOTAL = 40
print(f"Stage 1 LR = {STAGE1_LR:.2e}  (Hyperband best={best_lr:.2e})")

tf.keras.backend.clear_session()


# # =====================================================
# # STAGE 1 — BOUNDING BOX  (bleeding only)
# # FIX v3: combined_box_loss from epoch 1, no phase switch
# # =====================================================
# print("\n" + "=" * 55)
# print("STAGE 1 — Bounding Box Regression")
# print("=" * 55)

# X_train, X_val, box_train, box_val, label_train, label_val = train_test_split(
#     *load_data(with_neg=False, aug=False), test_size=0.2, random_state=42
# )

# if os.path.exists(PATH_BOX):
#     print(f"  Loading existing weights: {PATH_BOX}")
#     model = load_model(PATH_BOX, custom_objects={**CUSTOM_BOX,
#                        "masked_smooth_l1": masked_smooth_l1})
# else:
#     print("  No existing weights — building fresh model.")
#     model = build_model(dropout_cls=best_dropout, dropout_reg=best_dropout,
#                         dense_units=best_units)

# # Freeze classification head
# for layer in model.layers:
#     if layer.name.startswith("c_"):
#         layer.trainable = False

# # FIX-B2: backbone trainable so BN adapts
# for layer in model.layers:
#     if layer.name == BACKBONE_LAYER_NAME:
#         layer.trainable = True
#         for sub in layer.layers:
#             sub.trainable = True
#         break

# model.compile(
#     optimizer    = tf.keras.optimizers.AdamW(learning_rate=STAGE1_LR,
#                                              clipnorm=1.0),
#     loss         = LOSSES_MAIN,
#     loss_weights = {"c_final": 0.0, "b_final": 1.0},
# )

# box_iou_cb = BoxIoUCallback(X_val, box_val)

# model.fit(
#     X_train, {"c_final": label_train, "b_final": box_train},
#     validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
#     epochs=STAGE1_TOTAL, batch_size=8,
#     callbacks=[
#         EarlyStopping(monitor="val_b_final_loss", patience=12,
#                       restore_best_weights=True, mode="min"),
#         ModelCheckpoint(PATH_BOX, monitor="val_b_final_loss",
#                         save_best_only=True, mode="min"),
#         ReduceLROnPlateau(monitor="val_b_final_loss", factor=0.5,
#                           patience=5, min_lr=1e-6, mode="min"),
#         box_iou_cb,
#     ],
#     verbose=1,
# )
# print(f"Stage 1 complete → {PATH_BOX}")
# tf.keras.backend.clear_session()
# gc.collect()


# # =====================================================
# # STAGE 2 — CLASSIFICATION  (bleeding + non-bleeding)
# # =====================================================
# print("\n" + "=" * 55)
# print("STAGE 2 — Classification")
# print("=" * 55)

# X_train, X_val, box_train, box_val, label_train, label_val = train_test_split(
#     *load_data(with_neg=True, aug=False), test_size=0.2, random_state=42
# )

# # FIX-7: class_weight for imbalance
# n_bleed    = int(label_train.sum())
# n_nonbleed = len(label_train) - n_bleed
# total      = len(label_train)
# class_weight = {
#     0: total / (2.0 * max(n_nonbleed, 1)),
#     1: total / (2.0 * max(n_bleed,    1)),
# }
# print(f"  class_weight: {class_weight}")

# if os.path.exists(PATH_CLS):
#     print(f"  Loading existing Stage 2 weights: {PATH_CLS}")
#     model = load_model(PATH_CLS, custom_objects=CUSTOM_BOX)
# else:
#     print(f"  Fine-tuning from Stage 1: {PATH_BOX}")
#     model = load_model(PATH_BOX, custom_objects={**CUSTOM_BOX,
#                        "masked_smooth_l1": masked_smooth_l1})

# for layer in model.layers:
#     layer.trainable = False
# for layer in model.layers:
#     if layer.name.startswith("c_"):
#         layer.trainable = True

# backbone = None
# for layer in model.layers:
#     if layer.name == BACKBONE_LAYER_NAME:
#         backbone = layer
#         break

# if backbone is not None:
#     sub_layers = [l for l in backbone.layers
#                   if hasattr(l, "trainable") and len(l.weights) > 0]
#     n_unfreeze = max(2, len(sub_layers) // 5)
#     for sub in sub_layers[-n_unfreeze:]:
#         sub.trainable = True
#     print(f"  Backbone: unfroze last {n_unfreeze}/{len(sub_layers)} sub-layers")
# else:
#     print(f"  WARNING: backbone '{BACKBONE_LAYER_NAME}' not found.")

# model.compile(
#     optimizer    = tf.keras.optimizers.AdamW(learning_rate=1e-4, clipnorm=1.0),
#     loss         = LOSSES_MAIN,
#     loss_weights = {"c_final": 1.0, "b_final": 0.0},
#     metrics      = {"c_final": "accuracy"},
# )
# model.fit(
#     X_train, {"c_final": label_train, "b_final": box_train},
#     validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
#     epochs=50, batch_size=8,
#     # class_weight=class_weight,
#     callbacks=make_callbacks(PATH_CLS, monitor="val_c_final_loss",
#                              es_patience=10, lr_patience=5),
#     verbose=1,
# )
# print(f"Stage 2 complete → {PATH_CLS}")
# tf.keras.backend.clear_session()
# gc.collect()


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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss      = focal_tversky,
    metrics   = [dice_coef, tf.keras.metrics.BinaryIoU(threshold=0.5)],
)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10, batch_size=8,
    callbacks=make_callbacks(PATH_SEG, monitor="val_dice_coef", mode="max",
                             es_patience=10, lr_patience=5),
    verbose=1,
)
print(f"Stage 3 complete → {PATH_SEG}")
tf.keras.backend.clear_session()
gc.collect()


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