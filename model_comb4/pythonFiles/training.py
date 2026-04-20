"""
combo4_training.py  —  Combination 4 Training Pipeline
========================================================
Primary  : YOLOv8n-seg  → bounding box + segmentation mask  (ultralytics)
Secondary: EfficientNet-B0  → classification  (Keras / TensorFlow)
Tuner    : Bayesian Optimisation  ≤15 trials  (keras-tuner)

Stage structure
  Stage 1  — Bayesian search over EfficientNet-B0 hyper-parameters
  Stage 2  — EfficientNet-B0 classification head training (two phases:
             frozen backbone head-only, then partial backbone fine-tune)
  Stage 3  — YOLOv8s-seg training on YOLO-format segmentation dataset
  Stage 4  — ColonNet4 manifest JSON saved (combo4_manifest.json)
             YOLO uses .pt weights and cannot be merged into a .keras file,
             so Stage 4 saves a manifest that combo4_eval.py reads at
             inference time to locate both model files.

Outputs (all under SavedModels/)
  combo4_bo_params.json      — best Bayesian hyper-parameters
  combo4_classifier.keras    — trained EfficientNet-B0 classifier
  combo4_yolo_best.pt        — trained YOLOv8s-seg weights
  combo4_manifest.json       — paths to both models + metadata for eval

Install
  pip install ultralytics keras-tuner

Label convention (consistent across all combinations):
  1 = bleeding   (EfficientNet sigmoid ~1, YOLO class 0)
  0 = non-bleeding

FIXES APPLIED
  1. Removed mixed_float16 — breaks YOLO (PyTorch model)
  2. Unwrapped triple-quote comments around Stage 1 and Stage 4
  3. backbone.trainable=False before first forward pass (BN fix)
  4. class_weight added to both fit() calls to handle class imbalance
"""

import gc
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split

from utils.data_loaders import load_data
from utils.base_models import build_cls_model, run_bayesian_search

# -----------------------------------------------------------------
# GPU SETUP
# FIX 1: mixed_float16 removed — YOLO is PyTorch; mixed precision
#         here only affects Keras and causes YOLO dtype conflicts.
# -----------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# -----------------------------------------------------------------
# CONSTANTS  — defined before any stage that references them
# -----------------------------------------------------------------
IMG_SIZE      = 224
BACKBONE_NAME = "efficientnetb0"

# -----------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------
MODELS_DIR   = os.path.join(ROOT, "SavedModels")
DATA_ROOT    = r"C:\Users\Shubham\Desktop\ColonNet\TrainingDataset"
os.makedirs(MODELS_DIR, exist_ok=True)

def mp(name):
    return os.path.join(MODELS_DIR, name)

PATH_CLS      = mp("combo4_classifier.keras")
PATH_BO       = mp("combo4_bo_params.json")
PATH_YOLO_PT  = mp("combo4_yolo_best.pt")
PATH_MANIFEST = mp("combo4_manifest.json")
TUNER_DIR     = mp("combo4_bo_tuner_logs")
YOLO_DATA_DIR = mp("combo4_yolo_dataset")

# -----------------------------------------------------------------
# METRICS / CUSTOM OBJECTS
# -----------------------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    inter    = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (
        tf.keras.backend.sum(y_true_f) +
        tf.keras.backend.sum(y_pred_f) + smooth
    )

CLS_CUSTOM = {"dice_coef": dice_coef}

# -----------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------

class PrintMetrics(tf.keras.callbacks.Callback):
    """Prints a clean one-line metric summary after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs    = logs or {}
        metrics = ", ".join(
            f"{k}: {v:.4f}" for k, v in logs.items()
            if isinstance(v, (int, float))
        )
        print(f"Epoch {epoch + 1} — {metrics}")


def make_callbacks(path, monitor="val_loss", mode="min",
                   es_patience=10, lr_patience=5):
    return [
        PrintMetrics(),
        EarlyStopping(monitor=monitor, patience=es_patience,
                      restore_best_weights=True, mode=mode, verbose=1),
        ModelCheckpoint(path, monitor=monitor, save_best_only=True,
                        mode=mode, verbose=1),
        ReduceLROnPlateau(monitor=monitor, factor=0.5,
                          patience=lr_patience, min_lr=1e-6,
                          mode=mode, verbose=1),
    ]
"""
# =====================================================
# STAGE 1 — Bayesian Optimisation
# FIX 2: Was wrapped in triple-quote comment — never ran.
# =====================================================
print("\n" + "=" * 55)
print("STAGE 1 — Bayesian Optimisation (<=15 trials)")
print("=" * 55)

if os.path.exists(PATH_BO):
    print(f"  Loading cached BO params: {PATH_BO}")
    with open(PATH_BO) as f:
        best_hp = json.load(f)
else:
    images, _, labels = load_data(with_neg=True, aug=False,
                                  data_root=DATA_ROOT)

    # Stratified balanced subset for BO speed (<=600 images total)
    np.random.seed(42)
    bleed_idx    = np.where(labels == 1)[0]
    nonbleed_idx = np.where(labels == 0)[0]
    n_each       = min(300, len(bleed_idx), len(nonbleed_idx))
    idx          = np.concatenate([
        np.random.choice(bleed_idx,    n_each, replace=False),
        np.random.choice(nonbleed_idx, n_each, replace=False),
    ])
    np.random.shuffle(idx)
    X_bo = images[idx]
    y_bo = labels[idx]
    print(f"  BO subset: {X_bo.shape}  "
          f"(bleeding={int(y_bo.sum())}  "
          f"non-bleeding={int((y_bo == 0).sum())})")

    best_hp = run_bayesian_search(
        X_bo, y_bo,
        tuner_dir        = TUNER_DIR,
        max_trials       = 15,
        epochs_per_trial = 10,
        val_split        = 0.2,
        batch_size       = 16,
        seed             = 42,
        overwrite        = False,
    )
    with open(PATH_BO, "w") as f:
        json.dump(best_hp, f, indent=2)
    print(f"  Saved BO params -> {PATH_BO}")

best_lr      = best_hp["learning_rate"]
best_dropout = best_hp["dropout"]
best_units   = best_hp["dense_units"]
print(f"  LR={best_lr:.4e}  dropout={best_dropout:.2f}  "
      f"dense_units={best_units}")

tf.keras.backend.clear_session()
gc.collect()


# =====================================================
# STAGE 2 — EfficientNet-B0 Classification
# Phase A: frozen backbone, head only
# Phase B: partial backbone fine-tuning (last 20%)
# =====================================================
print("\n" + "=" * 55)
print("STAGE 2 — EfficientNet-B0 Classification")
print("=" * 55)

images, _, labels = load_data(with_neg=True, aug=False, data_root=DATA_ROOT)
X_tr, X_val, y_tr, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels,
)
print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  "
      f"(bleeding train={int(y_tr.sum())}  "
      f"non-bleed train={int((y_tr == 0).sum())})")

del images
gc.collect()

# FIX 4: class_weight to handle bleeding/non-bleeding imbalance
n_bleed      = int(y_tr.sum())
n_nonbleed   = len(y_tr) - n_bleed
total        = len(y_tr)
class_weight = {
    0: total / (2.0 * max(n_nonbleed, 1)),
    1: total / (2.0 * max(n_bleed,    1)),
}
print(f"  class_weight: {class_weight}")

if os.path.exists(PATH_CLS):
    print(f"  Loading existing classifier: {PATH_CLS}")
    cls_model = tf.keras.models.load_model(PATH_CLS,
                                           custom_objects=CLS_CUSTOM,
                                           compile=False)
    # Resume fine-tuning with full model unfrozen at reduced LR
    cls_model.compile(
        optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=best_lr * 0.1, clipnorm=1.0),
        loss      = tf.keras.losses.BinaryCrossentropy(),
        metrics   = ["accuracy", dice_coef],
    )
    print("  Resuming fine-tuning from checkpoint.")
    cls_model.fit(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs=40, batch_size=16,
        class_weight=class_weight,
        callbacks=make_callbacks(PATH_CLS, monitor="val_loss",
                                 es_patience=10, lr_patience=5),
        verbose=1,
    )

else:
    print("  Building classifier from scratch.")
    # FIX 3: build_cls_model now sets backbone.trainable=False internally
    #         before the first forward pass so BatchNorm runs in inference
    #         mode during Phase A. Phase B explicitly re-enables it below.
    cls_model = build_cls_model(dropout=best_dropout,
                                dense_units=best_units)

    # -- Phase A: head-only (backbone completely frozen) ----------
    for layer in cls_model.layers:
        if layer.name == BACKBONE_NAME:
            layer.trainable = False
            break

    cls_model.compile(
        optimizer = tf.keras.optimizers.AdamW(learning_rate=best_lr,
                                              clipnorm=1.0),
        loss      = tf.keras.losses.BinaryCrossentropy(),
        metrics   = ["accuracy"],
    )
    print("  Phase A: head-only (backbone frozen) ...")
    cls_model.fit(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs=40, batch_size=16,
        class_weight=class_weight,
        callbacks=make_callbacks(PATH_CLS, monitor="val_loss",
                                 es_patience=10, lr_patience=5),
        verbose=1,
    )
    print(f"  Phase A complete -> {PATH_CLS}")

    # -- Phase B: partial backbone fine-tuning --------------------
    backbone = None
    for layer in cls_model.layers:
        if layer.name == BACKBONE_NAME:
            backbone = layer
            backbone.trainable = True
            break

    if backbone is not None:
        sub_layers = [l for l in backbone.layers if len(l.weights) > 0]
        n_unfreeze = max(2, len(sub_layers) // 5)
        for sub in sub_layers[:-n_unfreeze]:
            sub.trainable = False
        print(f"  Phase B: fine-tuning last {n_unfreeze}/"
              f"{len(sub_layers)} backbone layers ...")
    else:
        print(f"  WARNING: backbone '{BACKBONE_NAME}' not found — "
              "full model will fine-tune.")

    cls_model.compile(
        optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=best_lr * 0.1, clipnorm=1.0),
        loss      = tf.keras.losses.BinaryCrossentropy(),
        metrics   = ["accuracy", dice_coef],
    )
    cls_model.fit(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs=40, batch_size=16,
        class_weight=class_weight,
        callbacks=make_callbacks(PATH_CLS, monitor="val_loss",
                                 es_patience=10, lr_patience=5),
        verbose=1,
    )
    print(f"  Phase B complete -> {PATH_CLS}")

print(f"\nStage 2 complete -> {PATH_CLS}")
tf.keras.backend.clear_session()
gc.collect()
"""
# =====================================================
# STAGE 3 — YOLOv8n-seg Training
# Single unified model: bounding boxes + instance masks.
# =====================================================
print("\n" + "=" * 55)
print("STAGE 3 — YOLOv8n-seg  (box + segmentation)")
print("=" * 55)

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("  WARNING: ultralytics not installed. "
          "Run: pip install ultralytics\n  Stage 3 skipped.")

if _YOLO_AVAILABLE and not os.path.exists(PATH_YOLO_PT):
    from utils.yolo import prepare_yolo_seg_dataset

    yaml_path = prepare_yolo_seg_dataset(
        data_root = DATA_ROOT,
        yolo_root = YOLO_DATA_DIR,
        val_split = 0.2,
        seed      = 42,
    )

    yolo_model = _YOLO("yolov8n-seg.pt")
  # downloads pretrained on first run
#  to implement image augmentation, use the following in train_kwargs:
    train_kwargs = dict(
    data     = yaml_path,
    epochs   = 100,
    imgsz    = IMG_SIZE,
    batch    = 16,
    patience = 40,
    project  = MODELS_DIR,
    name     = "combo4_yolo_seg",
    exist_ok = True,
    verbose  = True,
    mosaic   = 0.0,
    flipud   = 0.5,
    fliplr   = 0.5,
    degrees  = 15.0,
    translate = 0.1,
    scale    = 0.3,
    hsv_h    = 0.015,
    hsv_s    = 0.5,
    resume   = True,   # resume from last.pt if exists, otherwise start fresh
)

    gpu_list = tf.config.list_physical_devices("GPU")
    if gpu_list:
        train_kwargs["device"] = "0"

    results = yolo_model.train(**train_kwargs)

    yolo_metrics = {}
    try:
        rm = results.results_dict
        box_map50    = float(rm.get("metrics/mAP50(B)",    0.0))
        box_map5095  = float(rm.get("metrics/mAP50-95(B)", 0.0))
        seg_map50    = float(rm.get("metrics/mAP50(M)",    0.0))
        seg_map5095  = float(rm.get("metrics/mAP50-95(M)", 0.0))
        box_loss_val = float(rm.get("val/box_loss",         0.0))
        seg_loss_val = float(rm.get("val/seg_loss",         0.0))
        cls_loss_val = float(rm.get("val/cls_loss",         0.0))
        epochs_run   = int(getattr(results, "epoch", 0)) + 1

        yolo_metrics = {
            "epochs_run":   epochs_run,
            "box_mAP50":    round(box_map50,   4),
            "box_mAP50-95": round(box_map5095, 4),
            "seg_mAP50":    round(seg_map50,   4),
            "seg_mAP50-95": round(seg_map5095, 4),
            "val_box_loss": round(box_loss_val, 4),
            "val_seg_loss": round(seg_loss_val, 4),
            "val_cls_loss": round(cls_loss_val, 4),
        }
        print(f"\n  YOLO training summary ({epochs_run} epochs):")
        print(f"  {'─'*40}")
        print(f"  Box  mAP@0.50      = {box_map50:.4f}")
        print(f"  Box  mAP@0.50:0.95 = {box_map5095:.4f}")
        print(f"  Seg  mAP@0.50      = {seg_map50:.4f}")
        print(f"  Seg  mAP@0.50:0.95 = {seg_map5095:.4f}")
        print(f"  Val  box_loss      = {box_loss_val:.4f}")
        print(f"  Val  seg_loss      = {seg_loss_val:.4f}")
        print(f"  Val  cls_loss      = {cls_loss_val:.4f}")
        print(f"  {'─'*40}")
        if box_map50 < 0.05 and epochs_run >= 20:
            print(f"\n  WARNING: box mAP@0.50={box_map50:.4f} after {epochs_run} epochs.")
            print(f"  Box head may have collapsed. Delete {PATH_YOLO_PT} and retrain.")
        if seg_map50 < 0.05 and epochs_run >= 20:
            print(f"\n  WARNING: seg mAP@0.50={seg_map50:.4f} after {epochs_run} epochs.")
            print(f"  Check polygon labels in {YOLO_DATA_DIR} are non-empty.")
    except Exception as e:
        print(f"  Could not parse YOLO metrics: {e}")
        yolo_metrics = {}

    trained_best = os.path.join(MODELS_DIR, "combo4_yolo_seg",
                                "weights", "best.pt")
    if os.path.exists(trained_best):
        shutil.copy2(trained_best, PATH_YOLO_PT)
        print(f"  YOLOv8n-seg best weights -> {PATH_YOLO_PT}")
    else:
        fallback = os.path.join(MODELS_DIR, "combo4_yolo_seg",
                                "weights", "last.pt")
        if os.path.exists(fallback):
            shutil.copy2(fallback, PATH_YOLO_PT)
            print(f"  WARNING: best.pt missing, copied last.pt -> {PATH_YOLO_PT}")
        else:
            print(f"  ERROR: No YOLO weights found under {MODELS_DIR}.")

elif os.path.exists(PATH_YOLO_PT):
    print(f"  Existing YOLO weights found — skipping retraining.")
    print(f"  {PATH_YOLO_PT}")
    yolo_metrics = {}
else:
    yolo_metrics = {}

gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
except ImportError:
    pass

print(f"\nStage 3 complete -> {PATH_YOLO_PT}")

# =====================================================
# STAGE 4 — Save ColonNet4 Manifest
# FIX 2: Was wrapped in triple-quote comment — never ran.
# =====================================================
print("\n" + "=" * 55)
print("STAGE 4 — Saving ColonNet4 Manifest")
print("=" * 55)

# Load best_hp directly from saved JSON so Stage 4 works regardless
# of whether Stages 1+2 ran or were commented out this session.
with open(PATH_BO) as f:
    _hp      = json.load(f)
best_lr      = _hp["learning_rate"]
best_dropout = _hp["dropout"]
best_units   = _hp["dense_units"]
print(f"  BO params loaded → LR={best_lr:.4e}  "
      f"dropout={best_dropout:.2f}  dense_units={best_units}")

# yolo_metrics only populated when Stage 3 runs in this session
if "yolo_metrics" not in dir():
    yolo_metrics = {}

cls_ok  = os.path.exists(PATH_CLS)
yolo_ok = os.path.exists(PATH_YOLO_PT)

if not cls_ok:
    print("  WARNING: classifier not found — Stage 2 may not have completed.")
if not yolo_ok:
    print("  WARNING: YOLO weights not found — Stage 3 may not have completed.")

manifest = {
    "combo":            4,
    "classifier_keras": PATH_CLS      if cls_ok   else None,
    "yolo_pt":          PATH_YOLO_PT  if yolo_ok  else None,
    "img_size":         IMG_SIZE,
    "label_convention": {"1": "bleeding", "0": "non-bleeding"},
    "backbone":         "EfficientNetB0",
    "detector":         "YOLOv8n-seg",
    "tuner":            "BayesianOptimization",
    "best_hp": {
        "learning_rate": best_lr,
        "dropout":       best_dropout,
        "dense_units":   best_units,
    },
    "yolo_train_metrics": yolo_metrics,
}

with open(PATH_MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"  Manifest saved -> {PATH_MANIFEST}")
print("\n  Contents:")
for k, v in manifest.items():
    print(f"    {k}: {v}")

print("\n" + "=" * 55)
print("Combo 4 Training Complete")
print("=" * 55)
print(f"  Stage 1 (Bayesian BO)  -> {PATH_BO}")
print(f"  Stage 2 (Classifier)   -> {PATH_CLS}")
print(f"  Stage 3 (YOLO-seg)     -> {PATH_YOLO_PT}")
print(f"  Stage 4 (Manifest)     -> {PATH_MANIFEST}")
print(f"\n  Run combo4_eval.py to evaluate on test datasets.")