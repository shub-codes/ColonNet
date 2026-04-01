"""
hyperband_tuner.py  —  Combination 3
Hyperband hyperparameter tuning via keras-tuner.

Replaces random_search.py from Combination 2.

Why Hyperband over Random Search?
  - Hyperband adaptively allocates compute: bad configs are killed early,
    good configs get more epochs automatically.
  - Far more efficient than grid/random search at the same compute budget.
  - keras-tuner integrates natively with Keras models.

Install:
    pip install keras-tuner

Search space (3 dims, same as Combo 2 for comparability):
  learning_rate : log-uniform in [1e-5, 1e-3]
  dropout       : uniform in {0.1, 0.2, 0.3, 0.4, 0.5}
  dense_units   : choice in {64, 128, 256}

Usage (called from training.py):
    from utils.hyperband_tuner import run_hyperband
    best_hp = run_hyperband(X_rs, ann_rs, box_rs, tuner_dir)
    best_lr      = best_hp["learning_rate"]
    best_dropout = best_hp["dropout"]
    best_units   = best_hp["dense_units"]
"""

import os
import gc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

try:
    import keras_tuner as kt
    _KT_AVAILABLE = True
except ImportError:
    _KT_AVAILABLE = False


# ── lazy import to avoid circular dependency ──────────────────────────
def _get_build_model():
    from utils.base_models import build_model
    return build_model


def _get_losses():
    """Returns the LOSSES dict used for the box-only objective."""
    import sys, os
    # losses are defined in training.py; we redefine them here to keep
    # hyperband_tuner.py self-contained.
    import tensorflow as tf

    def giou_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ix1  = tf.maximum(y_true[:, 0], y_pred[:, 0])
        iy1  = tf.maximum(y_true[:, 1], y_pred[:, 1])
        ix2  = tf.minimum(y_true[:, 2], y_pred[:, 2])
        iy2  = tf.minimum(y_true[:, 3], y_pred[:, 3])
        inter = tf.maximum(ix2-ix1, 0) * tf.maximum(iy2-iy1, 0)
        at = (y_true[:,2]-y_true[:,0]) * (y_true[:,3]-y_true[:,1])
        ap = (y_pred[:,2]-y_pred[:,0]) * (y_pred[:,3]-y_pred[:,1])
        union = at + ap - inter + 1e-7
        iou = inter / union
        ex1 = tf.minimum(y_true[:,0], y_pred[:,0])
        ey1 = tf.minimum(y_true[:,1], y_pred[:,1])
        ex2 = tf.maximum(y_true[:,2], y_pred[:,2])
        ey2 = tf.maximum(y_true[:,3], y_pred[:,3])
        enc = (ex2-ex1)*(ey2-ey1) + 1e-7
        return tf.reduce_mean(1.0 - (iou - (enc-union)/enc))

    def smooth_l1(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        diff = tf.abs(y_true - y_pred)
        return tf.reduce_mean(tf.where(diff < 1.0, 0.5*diff**2, diff-0.5))

    def combined_box_loss(y_true, y_pred):
        w = y_true[:,2] - y_true[:,0]
        h = y_true[:,3] - y_true[:,1]
        mask = tf.cast((w > 0) & (h > 0), tf.float32)
        loss = giou_loss(y_true, y_pred) + 0.5 * smooth_l1(y_true, y_pred)
        return tf.reduce_mean(loss * mask)

    return {
        "c_final": tf.keras.losses.BinaryCrossentropy(),
        "b_final": combined_box_loss,
    }


# =====================================================
# HyperModel — wraps build_model for keras-tuner
# =====================================================

class ColonHyperModel(kt.HyperModel):
    """
    Defines the search space and how to build + compile each trial model.
    Tunes: learning_rate, dropout, dense_units.
    Uses box-only loss (c_final weight=0) so the search focuses on
    regression quality — same strategy as Combo 2's random search.
    """

    def __init__(self, losses):
        super().__init__()
        self._losses = losses

    def build(self, hp):
        build_model = _get_build_model()

        lr          = hp.Float("learning_rate", 1e-5, 1e-3, sampling="log")
        dropout     = hp.Float("dropout", 0.1, 0.5, step=0.1)
        dense_units = hp.Choice("dense_units", [64, 128, 256])

        model = build_model(
            dropout_cls  = dropout,
            dropout_reg  = dropout,
            dense_units  = dense_units,
            weights      = "imagenet",
        )
        model.compile(
            optimizer    = tf.keras.optimizers.Adam(learning_rate=lr,
                                                    clipnorm=1.0),
            loss         = self._losses,
            loss_weights = {"c_final": 0.0, "b_final": 1.0},
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, **kwargs)


# =====================================================
# Public API
# =====================================================

def run_hyperband(X, ann, boxes, tuner_dir,
                  max_epochs=10, factor=3,
                  val_split=0.2, batch_size=16,
                  seed=42, overwrite=False):
    """
    Run Hyperband search and return the best hyperparameters as a dict.

    Hyperband settings for low-compute budget:
      max_epochs = 10   — maximum epochs any single trial can run
      factor     = 3    — halving/tripling factor (standard default)
      → roughly equivalent to ~6 random-search trials at 3 epochs each,
        but Hyperband promotes the best trials automatically.

    Parameters
    ----------
    X        : np.ndarray  (N, 224, 224, 3)  images
    ann      : np.ndarray  (N,)              class labels
    boxes    : np.ndarray  (N, 4)            bounding boxes
    tuner_dir: str         directory for keras-tuner logs/checkpoints
    max_epochs: int        Hyperband max_epochs parameter
    factor   : int         Hyperband reduction factor
    val_split: float       fraction used for validation inside the search
    batch_size: int        mini-batch size during search
    seed     : int         random seed
    overwrite: bool        if True, re-run search even if cached results exist

    Returns
    -------
    dict with keys: "learning_rate", "dropout", "dense_units"
    """
    if not _KT_AVAILABLE:
        raise ImportError(
            "keras-tuner is not installed. Run: pip install keras-tuner"
        )

    os.makedirs(tuner_dir, exist_ok=True)

    Xtr, Xval, btr, bval, atr, aval = train_test_split(
        X, boxes, ann, test_size=val_split, random_state=seed
    )

    losses      = _get_losses()
    hypermodel  = ColonHyperModel(losses)

    tuner = kt.Hyperband(
        hypermodel,
        objective        = "val_loss",
        max_epochs       = max_epochs,
        factor           = factor,
        directory        = tuner_dir,
        project_name     = "colonnet_hyperband",
        overwrite        = overwrite,
        seed             = seed,
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor  = "val_loss",
        patience = 3,
        restore_best_weights = True,
    )

    print(f"\n[Hyperband] Search space summary:")
    tuner.search_space_summary()

    print(f"\n[Hyperband] Starting search "
          f"(max_epochs={max_epochs}, factor={factor}) …")

    tuner.search(
        Xtr,
        {"c_final": atr, "b_final": btr},
        validation_data = (Xval, {"c_final": aval, "b_final": bval}),
        batch_size      = batch_size,
        callbacks       = [stop_early],
        verbose         = 1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    result = {
        "learning_rate": float(best_hp.get("learning_rate")),
        "dropout":       float(best_hp.get("dropout")),
        "dense_units":   int(best_hp.get("dense_units")),
    }

    print(f"\n[Hyperband] Best hyperparameters found:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # free memory
    tf.keras.backend.clear_session()
    gc.collect()

    return result
