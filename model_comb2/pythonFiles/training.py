import gc
import json
import os
import sys

# ─────────────────────────────────────────────────────────────
# ROOT = project root (one level above pythonFiles/)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# BACKBONE LAYER NAME
# In build_model() the MobileNetV3Small backbone is wrapped as a
# sub-model named "MobileNetV3Small_MultiScale". That is the name
# that appears as a layer inside the outer ColonSeg_Combo2 model.
# FIX: original used "densenet121" — wrong for this combination.
# ─────────────────────────────────────────────────────────────
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
    GIoU + 0.5 * smooth-L1, masked to valid (non-zero area) boxes only.
    Computed per-sample so the valid_mask zeroes out non-bleeding samples
    before the mean reduction — not after (which would be a no-op).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)       # (N,)

    # per-sample GIoU
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
    giou_per_sample = 1.0 - (iou - (enc - union) / enc)       # (N,)

    # per-sample smooth-L1
    diff           = tf.abs(y_true - y_pred)                   # (N, 4)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)               # (N,)

    per_sample = (giou_per_sample + 0.5 * sl1_per_sample) * valid_mask
    per_sample = tf.maximum(per_sample, 0.0)  # clamp: prevents negative GIoU from corrupting loss and collapsing box head to [0,0,1,1]
    n_valid    = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(per_sample) / n_valid


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
# CALLBACKS
# ─────────────────────────────────────────────────────────────

class PrintMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        metrics = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()
                            if isinstance(v, (int, float)))
        print(f"Epoch {epoch + 1} — {metrics}")


def make_callbacks(ckpt_path, monitor="val_loss", mode="min",
                   es_patience=5, lr_patience=3):
    return [
        PrintMetrics(),
        ModelCheckpoint(ckpt_path, monitor=monitor, mode=mode,
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor=monitor, mode=mode, patience=es_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=lr_patience, min_lr=1e-6, verbose=1),
    ]


# ─────────────────────────────────────────────────────────────
# RANDOM SEARCH HYPERPARAMETER TUNING
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
        epochs=5, batch_size=8, verbose=1,
    )
    val_loss = float(history.history.get("val_loss", [1e6])[-1])
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
    print("Starting Random Search (12 trials) …")
    rs = RandomSearch(n_trials=12, verbose=True, seed=42)
    best_pos, best_score = rs.optimize(
        rs_objective,
        lb=[-6.0, 0.0, 0.5],
        ub=[-2.0, 0.6, 1.5],
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
    *load_data(with_neg=False, aug=False), test_size=0.2
)

if os.path.exists(PATH_BOX):
    print(f"  Loading existing weights: {PATH_BOX}")
    model = load_model(PATH_BOX, custom_objects=CUSTOM_BOX)
else:
    print("  No existing weights — building fresh model.")
    model = build_model(dropout_cls=best_dropout, dropout_reg=best_dropout,
                        dense_scale=best_scale)

# Freeze classification head — box branch trains alone.
# loss_weight=0 does NOT stop gradients; trainable=False does.
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=best_lr, clipnorm=1.0),
    loss=LOSSES,
    loss_weights={"c_final": 0.0, "b_final": 1.0},
)
model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=30, batch_size=8,
    callbacks=make_callbacks(PATH_BOX, monitor="val_b_final_loss",
                             es_patience=10, lr_patience=10),
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
    *load_data(with_neg=True, aug=False), test_size=0.2
)

if os.path.exists(PATH_CLS):
    print(f"  Loading existing Stage 2 weights: {PATH_CLS}")
    model = load_model(PATH_CLS, custom_objects=CUSTOM_BOX)
else:
    print(f"  Fine-tuning from Stage 1: {PATH_BOX}")
    model = load_model(PATH_BOX, custom_objects=CUSTOM_BOX)

# FIX: freeze box branch AND backbone so only the classification
# head (c_gap, c_dense1, c_dropout1, c_dense2, c_final) trains.
# Original used "densenet121" — wrong backbone name for this combination.
# The backbone here is wrapped as "MobileNetV3Small_MultiScale".
# Freezing it prevents the catastrophic overfitting seen in training
# (98% train accuracy, 50% val accuracy on last epoch).
for layer in model.layers:
    if layer.name.startswith("b_") \
            or layer.name.startswith("ssd_") \
            or layer.name == BACKBONE_LAYER_NAME:
        layer.trainable = False
    else:
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=best_lr, clipnorm=1.0),
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
# STAGE 3 — SEGMENTATION  (U-Net, bleeding only)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 3 — Segmentation (U-Net)")
print("=" * 55)

X_train, X_val, y_train, y_val = train_test_split(
    *load_data_unet(), test_size=0.2, shuffle=True,
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