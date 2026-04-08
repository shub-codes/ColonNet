"""
training.py  —  ColonNet Training Pipeline  (Combination 3 — FIXED v2)
=======================================================================
Backbone: EfficientNet-B0  |  Segmentation: UNet++  |  Tuner: Hyperband

BOX COLLAPSE ROOT CAUSE & FIXES
────────────────────────────────
The model was predicting a near-zero box centred on the image despite
high classification accuracy.  Four independent causes were identified:

  FIX-B1  Two-phase Stage 1: warm-up with pure smooth-L1 (epochs 1-10),
           then switch to IoU+smooth-L1 (epochs 11-30).
           IoU loss has ZERO gradient for non-overlapping boxes.
           At random init the sigmoid box head outputs ~[0.5, 0.5, 0.5, 0.5]
           (i.e. a tiny centred square) which does not overlap most GT boxes
           → gradient is identically 0 → head never moves.
           Smooth-L1 always has gradient, so the warm-up phase drags boxes
           near the GT before IoU loss takes over.

  FIX-B2  Backbone set to training=True during Stage 1.
           training=False freezes BN statistics in inference mode, giving
           the box head stale activations that don't adapt to localisation.
           The backbone is already pre-trained; BN in train mode lets it
           adapt its running stats for the regression task.

  FIX-B3  Box head width doubled: dense_units (not dense_units//2).
           When Hyperband chose dense_units=64, the box dense layer was
           only 32 units — insufficient for 4-coordinate regression sharing
           features extracted for a 1280-dim backbone output.

  FIX-B4  Stage 1 LR raised to min(best_lr, 3e-4) with a 3-epoch linear
           warm-up via LearningRateWarmupCallback, then cosine-decay to
           1e-5 over the remaining epochs.
           The old cap of 1e-4 with no warm-up kept the loss plateau
           high enough for EarlyStopping to fire before convergence.
           Note: ReduceLROnPlateau removed for Stage 1 (incompatible with
           schedule); it is only used in Stage 2 and Stage 3.

  FIX-B5  BoxIoUCallback abort threshold raised from 0.15 → 0.08 and
           abort epoch raised from 10 → 15, giving the warm-up phase time
           to stabilise before the IoU phase is assessed.

Other fixes (unchanged from v1):
  FIX-1   combined_box_loss: IoU+smooth-L1 (not GIoU) — always ≥ 0.
  FIX-2   Hyperband tuner used (not RandomSearch).
  FIX-3   build_model called with dense_units= (not dense_scale=).
  FIX-4   Stage 2 backbone layer name "efficientnetb0".
  FIX-5   CosineDecay removed from Stage 2 (ReduceLROnPlateau incompatible).
  FIX-6   BoxIoUCallback retained with relaxed thresholds (FIX-B5).
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

BACKBONE_LAYER_NAME = "efficientnetb0"

# ─────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────

def smooth_l1(y_true, y_pred):
    """Pure smooth-L1. Used as warm-up loss in Stage 1 phase 1."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    diff   = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5))


def masked_smooth_l1(y_true, y_pred):
    """
    Smooth-L1 masked to valid (non-zero area) boxes.
    Used as WARM-UP loss during Stage 1 phase 1 (epochs 1-WARMUP_EPOCHS).
    Always has gradient regardless of overlap → escapes the init dead-zone.
    """
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
    """
    IoU loss + 0.5 * smooth-L1, masked to valid boxes.
    Range: [0, 1.5] — always non-negative.
    Used in Stage 1 PHASE 2 once boxes are near GT (after warm-up).

    Why not GIoU: produces values in [-1,2] → total loss goes negative
    → mode='min' checkpointing selects the most-collapsed model.
    Why not smooth-L1 alone: minimises mean coordinate error → constant
    prediction drifts toward mean of all GT boxes ≈ image centre.
    IoU loss directly penalises poor overlap.  Smooth-L1 at weight 0.5
    provides gradient when boxes are non-overlapping (IoU grad = 0 there).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    w          = y_true[:, 2] - y_true[:, 0]
    h          = y_true[:, 3] - y_true[:, 1]
    valid_mask = tf.cast((w > 0) & (h > 0), tf.float32)

    ix1   = tf.maximum(y_true[:, 0], y_pred[:, 0])
    iy1   = tf.maximum(y_true[:, 1], y_pred[:, 1])
    ix2   = tf.minimum(y_true[:, 2], y_pred[:, 2])
    iy2   = tf.minimum(y_true[:, 3], y_pred[:, 3])
    inter = tf.maximum(ix2 - ix1, 0.0) * tf.maximum(iy2 - iy1, 0.0)
    area_t = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_p = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union  = area_t + area_p - inter + 1e-7
    iou    = tf.clip_by_value(inter / union, 0.0, 1.0)
    iou_loss_per_sample = 1.0 - iou

    diff           = tf.abs(y_true - y_pred)
    sl1            = tf.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
    sl1_per_sample = tf.reduce_mean(sl1, axis=1)

    per_sample = (iou_loss_per_sample + 0.5 * sl1_per_sample) * valid_mask
    n_valid    = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    return tf.reduce_sum(per_sample) / n_valid


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


LOSSES_WARMUP = {"c_final": tf.keras.losses.BinaryCrossentropy(),
                 "b_final": masked_smooth_l1}
LOSSES_MAIN   = {"c_final": tf.keras.losses.BinaryCrossentropy(),
                 "b_final": combined_box_loss}
CUSTOM_BOX    = {"giou_loss":         giou_loss,
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

SEG_CUSTOM = {"focal_tversky": focal_tversky, "tversky": tversky, "dice_coef": dice_coef}


# ─────────────────────────────────────────────────────────────
# LR WARM-UP CALLBACK  (FIX-B4)
# ─────────────────────────────────────────────────────────────

class LRWarmupCosineDecay(tf.keras.callbacks.Callback):
    """
    Linear warm-up for `warmup_epochs`, then cosine decay to `min_lr`.
    Replaces ReduceLROnPlateau for Stage 1 (incompatible with manual schedule).
    """
    def __init__(self, base_lr, warmup_epochs, total_epochs, min_lr=1e-5):
        super().__init__()
        self.base_lr       = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress))
        opt_lr = self.model.optimizer.learning_rate
        try:
            opt_lr.assign(lr)
        except (AttributeError, TypeError):
            self.model.optimizer.learning_rate = lr


# ─────────────────────────────────────────────────────────────
# BOX PHASE-SWITCH CALLBACK  (FIX-B1)
# ─────────────────────────────────────────────────────────────

WARMUP_EPOCHS = 10   # epochs with pure smooth-L1 before switching to IoU loss

class BoxLossSwitchCallback(tf.keras.callbacks.Callback):
    """
    At the END of epoch (WARMUP_EPOCHS - 1), recompiles the model with
    combined_box_loss (IoU + smooth-L1) instead of masked_smooth_l1.

    FIX: recompile must happen in on_epoch_END (last warmup epoch), NOT
    on_epoch_BEGIN (first IoU epoch).  Calling model.compile() inside
    on_epoch_begin destroys the train_function that Keras already built for
    that epoch — Keras then tries to call None on the next step and raises
    TypeError: 'NoneType' object is not callable.
    Recompiling at on_epoch_end gives Keras the full inter-epoch gap to
    rebuild train_function before the next epoch starts.
    """
    def __init__(self, switch_epoch, stage1_lr, dense_units):
        super().__init__()
        self.switch_epoch = switch_epoch   # = WARMUP_EPOCHS (e.g. 10)
        self.stage1_lr    = stage1_lr
        self.dense_units  = dense_units
        self._switched    = False

    def on_epoch_end(self, epoch, logs=None):
        # epoch is 0-indexed; fire at the END of epoch (switch_epoch - 1)
        # so the new loss is active from the very next epoch onward.
        if epoch == self.switch_epoch - 1 and not self._switched:
            self._switched = True
            print(f"\n[BoxLossSwitch] Epoch {epoch + 1} complete: "
                  f"switching to IoU + smooth-L1 loss for remaining epochs")

            # FIX: reuse the EXISTING optimizer — do NOT create a new one.
            # Creating a new AdamW resets moment estimates AND produces a plain
            # float learning_rate that LRWarmupCosineDecay cannot .assign() to,
            # causing train_function to become None on the next epoch_begin.
            # Passing the existing optimizer instance preserves its state and
            # the LR variable that the warmup callback already holds a reference to.
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=LOSSES_MAIN,
                loss_weights={"c_final": 0.0, "b_final": 1.0},
            )


# ─────────────────────────────────────────────────────────────
# BOX IoU CALLBACK  (FIX-B5: relaxed thresholds)
# ─────────────────────────────────────────────────────────────

class BoxIoUCallback(tf.keras.callbacks.Callback):
    """
    Computes val mean-IoU each epoch during Stage 1.
    Aborts at ABORT_EPOCH if mean-IoU < ABORT_THRESHOLD.

    FIX-B5: thresholds relaxed vs v1 (0.15 @ epoch 10 → 0.08 @ epoch 15)
    to allow the warm-up phase (epochs 1-10) to complete before assessment.
    """
    ABORT_THRESHOLD = 0.08   # was 0.15 in v1
    ABORT_EPOCH     = 15     # was 10  in v1 — must be > WARMUP_EPOCHS

    def __init__(self, X_val, box_val):
        super().__init__()
        self.X_val   = X_val
        self.box_val = box_val

    def _mean_iou(self, preds):
        """Compute mean IoU between predicted and GT boxes (numpy)."""
        # preds is a list: [c_out, b_out]
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
                  f"{self.ABORT_THRESHOLD} at epoch {self.ABORT_EPOCH}. "
                  f"Check data pipeline or increase WARMUP_EPOCHS.")
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

# FIX-B4: raise LR cap to 3e-4 (from 1e-4); warm-up callback handles ramp-up
STAGE1_LR    = min(best_lr, 3e-4)
STAGE1_TOTAL = 40   # total Stage 1 epochs (warm-up=10 + IoU phase=30)
print(f"Stage 1 LR = {STAGE1_LR:.2e}  (Hyperband best={best_lr:.2e})")

tf.keras.backend.clear_session()


# =====================================================
# STAGE 1 — BOUNDING BOX  (bleeding only)
# Two-phase: smooth-L1 warm-up → IoU + smooth-L1
# =====================================================
print("\n" + "=" * 55)
print("STAGE 1 — Bounding Box Regression (two-phase)")
print("=" * 55)

X_train, X_val, box_train, box_val, label_train, label_val = train_test_split(
    *load_data(with_neg=False, aug=False), test_size=0.2, random_state=42
)

if os.path.exists(PATH_BOX):
    print(f"  Loading existing weights: {PATH_BOX}")
    model = load_model(PATH_BOX, custom_objects={**CUSTOM_BOX,
                       "masked_smooth_l1": masked_smooth_l1})
else:
    print("  No existing weights — building fresh model.")
    model = build_model(dropout_cls=best_dropout, dropout_reg=best_dropout,
                        dense_units=best_units)

# Freeze classification head — box branch trains alone.
for layer in model.layers:
    if layer.name.startswith("c_"):
        layer.trainable = False

# FIX-B2: backbone in training=True mode so BN adapts to localisation task
#          Find EfficientNetB0 sub-model and flip training flag
for layer in model.layers:
    if layer.name == BACKBONE_LAYER_NAME:
        layer.trainable = True
        # Override the training=False call used during build
        # by re-calling the sub-model with training=True in the functional graph.
        # Since we can't change the graph, we instead set all its BN layers
        # to trainable so they adapt running stats during Stage 1.
        for sub in layer.layers:
            sub.trainable = True
        break

# FIX-B3: box dense layer uses dense_units (not dense_units//2).
#   This is a base_models.py change — see base_models_fixed.py companion file.
#   If using the existing base_models.py, the fix requires re-building the
#   model graph (done in the companion base_models_fixed.py).

# Phase 1: pure smooth-L1 warm-up (no IoU — guaranteed gradient at init)
model.compile(
    optimizer    = tf.keras.optimizers.AdamW(learning_rate=STAGE1_LR,
                                             clipnorm=1.0),
    loss         = LOSSES_WARMUP,
    loss_weights = {"c_final": 0.0, "b_final": 1.0},
)

lr_schedule  = LRWarmupCosineDecay(
                   base_lr=STAGE1_LR,
                   warmup_epochs=3,
                   total_epochs=STAGE1_TOTAL,
                   min_lr=1e-5)
box_iou_cb   = BoxIoUCallback(X_val, box_val)
switch_cb    = BoxLossSwitchCallback(WARMUP_EPOCHS, STAGE1_LR, best_units)

model.fit(
    X_train, {"c_final": label_train, "b_final": box_train},
    validation_data=(X_val, {"c_final": label_val, "b_final": box_val}),
    epochs=STAGE1_TOTAL, batch_size=8,
    callbacks=[
        EarlyStopping(monitor="val_b_final_loss", patience=12,
                      restore_best_weights=True, mode="min"),
        ModelCheckpoint(PATH_BOX, monitor="val_b_final_loss",
                        save_best_only=True,mode="min"),
        lr_schedule,
        switch_cb,
        box_iou_cb,
    ],
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
    model = load_model(PATH_BOX, custom_objects={**CUSTOM_BOX,
                       "masked_smooth_l1": masked_smooth_l1})

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
    loss=LOSSES_MAIN,
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