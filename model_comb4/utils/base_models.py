"""
combo4_models.py  —  Combination 4
====================================
Architecture
  Primary Model  : YOLOv8n-seg  (bounding box + instance segmentation)
  Classification : EfficientNet-B0 head  (fine-tuned from ImageNet)
  Tuning         : Bayesian Optimisation  (≤15 trials, keras-tuner BayesianOptimization)

Design rationale
  YOLOv8n-seg is a single unified model that produces both bounding boxes and
  segmentation masks in one forward pass.  It replaces the separate Stage 1
  (box regression) and Stage 3 (UNet++) from Combination 3, eliminating:
    • The two-phase loss warm-up needed to escape the IoU dead-zone
    • The separate UNet++ training loop
    • Stage 4 model fusion (ColonNet assembly)

  The EfficientNet-B0 classification head is trained independently on the
  same images with binary labels (1=bleeding, 0=non-bleeding) and produces
  a confidence score used alongside YOLO's detection.

  Bayesian Optimisation (BO) uses a Gaussian Process surrogate to model the
  loss landscape, selecting the next trial at the point of maximum expected
  improvement.  With ≤15 trials it typically outperforms both random search
  and Hyperband at equal compute budgets for 2–4 dimensional search spaces.

Install
  pip install ultralytics keras-tuner

Label convention (consistent across all combinations):
  1 = bleeding   (YOLO class 0, EfficientNet sigmoid ~1)
  0 = non-bleeding
"""

import os
import gc
import shutil
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, Rescaling,
)
from tensorflow.keras.applications import EfficientNetB0

try:
    import keras_tuner as kt
    _KT_AVAILABLE = True
except ImportError:
    _KT_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

IMG_SIZE = 224


# =====================================================
# EfficientNet-B0 Classification Head
# =====================================================

def build_cls_model(dropout=0.3, dense_units=256, weights="imagenet"):
    """
    Binary classifier: bleeding (1) vs non-bleeding (0).

    Input  : (batch, 224, 224, 3)  float32  pixel values in [0, 1]
    Output : (batch, 1)            float32  sigmoid probability

    Output layer dtype='float32' prevents sigmoid saturation under
    mixed_float16 global policy.
    """
    inp      = Input(shape=(224, 224, 3), name="cls_input")
    x        = Rescaling(scale=255.0, name="rescale_0_255")(inp)
    # In build_cls_model(), change order:
    backbone = EfficientNetB0(include_top=False, weights=weights,
                            input_shape=(224, 224, 3))
    backbone.trainable = False   # ← ADD THIS, move before backbone(x,...)
    feats    = backbone(x, training=False)
# REMOVE: backbone.trainable = True  (line that was there)
    # backbone.trainable = True
    feats    = backbone(x, training=False)
    x        = GlobalAveragePooling2D(name="cls_gap")(feats)
    x        = Dense(dense_units, activation="relu", name="cls_dense1")(x)
    x        = Dropout(dropout, name="cls_drop1")(x)
    x        = Dense(dense_units // 4, activation="relu", name="cls_dense2")(x)
    out      = Dense(1, activation="sigmoid", dtype="float32",
                     name="cls_output")(x)
    return Model(inputs=inp, outputs=out, name="Combo4_Classifier")


# =====================================================
# Bayesian Optimisation for the Classification Head
# =====================================================

class ClsHyperModel(kt.HyperModel):
    """
    Searches over:
      learning_rate : log-uniform [1e-5, 3e-4]
      dropout       : uniform {0.1, 0.2, 0.3, 0.4, 0.5}
      dense_units   : choice {64, 128, 256}
    """
    def build(self, hp):
        lr          = hp.Float("learning_rate", 1e-5, 3e-4, sampling="log")
        dropout     = hp.Float("dropout", 0.1, 0.5, step=0.1)
        dense_units = hp.Choice("dense_units", [64, 128, 256])

        model = build_cls_model(dropout=dropout, dense_units=dense_units)
        model.compile(
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr,
                                                  clipnorm=1.0),
            loss      = tf.keras.losses.BinaryCrossentropy(),
            metrics   = ["accuracy"],
        )
        return model


def run_bayesian_search(X, y, tuner_dir,
                        max_trials=15, epochs_per_trial=10,
                        val_split=0.2, batch_size=16,
                        seed=42, overwrite=False):
    """
    Bayesian Optimisation over the EfficientNet-B0 classification head.

    Parameters
    ----------
    X             : np.ndarray  (N, 224, 224, 3)  float32 images [0,1]
    y             : np.ndarray  (N,)              float32 labels (0/1)
    tuner_dir     : str         directory for keras-tuner logs
    max_trials    : int         maximum BO trials (≤15 recommended)
    epochs_per_trial : int      epochs per trial
    val_split     : float       validation fraction
    batch_size    : int
    seed          : int
    overwrite     : bool        re-run even if cached results exist

    Returns
    -------
    dict with keys: "learning_rate", "dropout", "dense_units"
    """
    if not _KT_AVAILABLE:
        raise ImportError("pip install keras-tuner")

    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_val, y_tr, y_val = _tts(X, y, test_size=val_split,
                                      random_state=seed)

    tuner = kt.BayesianOptimization(
        ClsHyperModel(),
        objective        = "val_accuracy",
        max_trials       = max_trials,
        directory        = tuner_dir,
        project_name     = "combo4_bayesian",
        overwrite        = overwrite,
        seed             = seed,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)

    print(f"\n[Bayesian BO] max_trials={max_trials}  "
          f"epochs_per_trial={epochs_per_trial}")
    tuner.search_space_summary()

    tuner.search(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs          = epochs_per_trial,
        batch_size      = batch_size,
        callbacks       = [stop_early],
        verbose         = 1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    result  = {
        "learning_rate": float(best_hp.get("learning_rate")),
        "dropout":       float(best_hp.get("dropout")),
        "dense_units":   int(best_hp.get("dense_units")),
    }
    print(f"\n[Bayesian BO] Best hyperparameters:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    tf.keras.backend.clear_session()
    gc.collect()
    return result