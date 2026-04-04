"""
hyperband_tuner.py  —  Combination 3
Hyperband hyperparameter tuning via keras-tuner.

FIX: Internal combined_box_loss replaced with smooth-L1 only.
     GIoU produces values in [-1, 2] making the total loss negative.
     With objective="val_loss", Hyperband treats the most-collapsed
     model (predicting [0,0,1,1]) as the best trial — same problem
     as in Combination 2's Random Search. Smooth-L1 is always >= 0.

Why Hyperband over Random Search?
  - Adaptively allocates compute: bad configs are killed early.
  - Good configs get more epochs automatically.
  - More efficient than grid/random at the same compute budget.

Search space (3 dims):
  learning_rate : log-uniform in [1e-5, 3e-4]   (upper bound lowered from 1e-3
                                                   to prevent LR that collapses
                                                   the box head in 2 epochs)
  dropout       : uniform in {0.1, 0.2, 0.3, 0.4, 0.5}
  dense_units   : choice in {64, 128, 256}

Install:
    pip install keras-tuner
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


def _get_build_model():
    from utils.base_models import build_model
    return build_model


def _make_smooth_l1_loss():
    """
    Returns smooth-L1 box loss masked to valid (non-zero area) boxes.
    Always >= 0 — safe for Hyperband's val_loss minimisation objective.
    GIoU has been removed: it produces negative losses, causing Hyperband
    to select the most-collapsed model as the best trial.
    """
    def combined_box_loss(y_true, y_pred):
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

    return {
        "c_final": tf.keras.losses.BinaryCrossentropy(),
        "b_final": combined_box_loss,
    }


# =====================================================
# HyperModel
# =====================================================

class ColonHyperModel(kt.HyperModel):
    """
    Defines the search space and how to build + compile each trial model.
    Tunes: learning_rate, dropout, dense_units.
    Uses box-only loss (c_final weight=0) so the search focuses on
    regression quality.
    """

    def __init__(self, losses):
        super().__init__()
        self._losses = losses

    def build(self, hp):
        build_model = _get_build_model()

        lr          = hp.Float("learning_rate", 1e-5, 3e-4, sampling="log")
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

    losses     = _make_smooth_l1_loss()
    hypermodel = ColonHyperModel(losses)

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

    tf.keras.backend.clear_session()
    gc.collect()

    return result