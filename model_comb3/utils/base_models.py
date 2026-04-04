"""
base_models.py  —  Combination 3
  Classification + Box : EfficientNet-B0  backbone
  Segmentation         : UNet++  (nested dense skip connections)

Label convention (consistent with data_loaders.py and training.py):
  bleeding     → 1   (sigmoid output ~1)
  non-bleeding → 0   (sigmoid output ~0)

FIX 1: Docstring previously said bleeding=0, non-bleeding=1 — inverted.
        Corrected to match LABEL_BLEED=1 in data_loaders.py.

FIX 2: Output layers (c_final, seg_output) now have dtype='float32'.
        Under mixed_float16 global policy, all layers compute in float16
        by default. Sigmoid in float16 saturates to exactly 0.0 or 1.0
        for inputs beyond ±8, causing gradient vanishing on the output.
        Output layers must always be float32.

FIX 3: build_model parameter renamed dense_scale → dense_units to match
        the hyperband_tuner search space ('dense_units' choice in [64,128,256]).
        training.py and hyperband_tuner both use 'dense_units'.
        Previously training.py called build_model(dense_scale=...) which
        raised TypeError immediately on the first RS/Hyperband trial.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    Conv2D, BatchNormalization, Activation,
    MaxPool2D, Conv2DTranspose, Concatenate,
    DepthwiseConv2D, UpSampling2D, Average, Rescaling,
)
from tensorflow.keras.applications import EfficientNetB0


# =====================================================
# Multi-Task Model: EfficientNet-B0 + Detection Head
#
# Stage 1  →  loss_weights {"c_final": 0, "b_final": 1}  (box only)
# Stage 2  →  loss_weights {"c_final": 1, "b_final": 0}  (cls only)
# Both heads share the EfficientNet-B0 backbone.
# =====================================================

def build_model(dropout_cls=0.3,
                dropout_reg=0.3,
                dense_units=256,
                weights="imagenet"):
    """
    Two-output Keras model:
      c_final : (batch, 1)  bleeding probability  (1=bleeding, 0=non-bleed)
      b_final : (batch, 4)  normalised [x1, y1, x2, y2] box

    Parameters
    ----------
    dropout_cls  : float  dropout for classification head
    dropout_reg  : float  dropout for bounding-box head
    dense_units  : int    width of first dense layer (Hyperband tunes this)
    weights      : str    "imagenet" or None
    """
    sp          = (224, 224, 3)
    input_layer = Input(shape=sp, name="Input")

    # EfficientNetB0 expects pixel values in [0, 255].
    # Our loader delivers [0, 1] floats — Rescaling handles this inside the model.
    x_scaled = Rescaling(scale=255.0, name="rescale_0_255")(input_layer)

    backbone = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=sp,
    )
    backbone.trainable = True

    # training=False keeps BN in inference mode during early fine-tuning.
    features = backbone(x_scaled, training=False)   # (batch, 7, 7, 1280)

    # ── Classification Head ───────────────────────────────────
    c = GlobalAveragePooling2D(name="c_gap")(features)
    c = Dense(dense_units, activation="relu", name="c_dense1")(c)
    c = Dropout(dropout_cls, name="c_drop1")(c)
    c = Dense(dense_units // 4, activation="relu", name="c_dense2")(c)
    # FIX 2: dtype='float32' prevents sigmoid saturation under mixed_float16
    c_out = Dense(1, activation="sigmoid", dtype="float32", name="c_final")(c)

    # ── Bounding-Box Head ─────────────────────────────────────
    b = DepthwiseConv2D(3, padding="same", name="b_dw")(features)
    b = BatchNormalization(name="b_dw_bn")(b)
    b = Activation("relu6", name="b_dw_relu")(b)
    b = Conv2D(64, 1, padding="same", name="b_pw")(b)
    b = GlobalAveragePooling2D(name="b_gap")(b)
    b = Dense(dense_units // 2, activation="relu", name="b_dense1")(b)
    b = Dropout(dropout_reg, name="b_drop1")(b)
    # FIX 2: dtype='float32' for box output too
    b_out = Dense(4, activation="sigmoid", dtype="float32", name="b_final")(b)

    model = Model(inputs=input_layer,
                  outputs=[c_out, b_out],
                  name="ColonSeg_Combo3_EfficientNet")
    return model


# =====================================================
# UNet++ (nested dense skip connections)
# =====================================================

def _conv_block(x, filters, name):
    """Two Conv→BN→ReLU layers — the basic UNet++ node."""
    x = Conv2D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_r1")(x)
    x = Conv2D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_r2")(x)
    return x


def _up(x, filters, name):
    """Bilinear upsample ×2, then 1×1 conv to fix filter count."""
    x = UpSampling2D(size=(2, 2), interpolation="bilinear",
                     name=f"{name}_ups")(x)
    x = Conv2D(filters, 1, padding="same", name=f"{name}_conv")(x)
    return x


def Build_UnetPP_Model(num_filters=32, input_shape=(224, 224, 3)):
    """
    Full UNet++ with 4 encoder levels.

    num_filters : base filter count.
                  Use 16 for low-compute budget, 32 for standard.
    """
    F  = num_filters
    Fs = [F, F*2, F*4, F*8, F*16]

    inp = Input(input_shape, name="input")

    # ── Encoder ──────────────────────────────────────────────
    x00 = _conv_block(inp,  Fs[0], "x00")
    p0  = MaxPool2D(2, name="pool0")(x00)

    x10 = _conv_block(p0,   Fs[1], "x10")
    p1  = MaxPool2D(2, name="pool1")(x10)

    x20 = _conv_block(p1,   Fs[2], "x20")
    p2  = MaxPool2D(2, name="pool2")(x20)

    x30 = _conv_block(p2,   Fs[3], "x30")
    p3  = MaxPool2D(2, name="pool3")(x30)

    x40 = _conv_block(p3,   Fs[4], "x40")   # bottleneck

    # ── Decoder column 1 ─────────────────────────────────────
    x31 = _conv_block(Concatenate(name="cat31")([x30, _up(x40, Fs[3], "u4031")]), Fs[3], "x31")
    x21 = _conv_block(Concatenate(name="cat21")([x20, _up(x30, Fs[2], "u3021")]), Fs[2], "x21")
    x11 = _conv_block(Concatenate(name="cat11")([x10, _up(x20, Fs[1], "u2011")]), Fs[1], "x11")
    x01 = _conv_block(Concatenate(name="cat01")([x00, _up(x10, Fs[0], "u1001")]), Fs[0], "x01")

    # ── Decoder column 2 ─────────────────────────────────────
    x22 = _conv_block(Concatenate(name="cat22")([x20, x21, _up(x31, Fs[2], "u3122")]), Fs[2], "x22")
    x12 = _conv_block(Concatenate(name="cat12")([x10, x11, _up(x21, Fs[1], "u2112")]), Fs[1], "x12")
    x02 = _conv_block(Concatenate(name="cat02")([x00, x01, _up(x11, Fs[0], "u1102")]), Fs[0], "x02")

    # ── Decoder column 3 ─────────────────────────────────────
    x13 = _conv_block(Concatenate(name="cat13")([x10, x11, x12, _up(x22, Fs[1], "u2213")]), Fs[1], "x13")
    x03 = _conv_block(Concatenate(name="cat03")([x00, x01, x02, _up(x12, Fs[0], "u1203")]), Fs[0], "x03")

    # ── Decoder column 4 ─────────────────────────────────────
    x04 = _conv_block(
        Concatenate(name="cat04")([x00, x01, x02, x03, _up(x13, Fs[0], "u1304")]),
        Fs[0], "x04")

    # ── Deep supervision output ───────────────────────────────
    out1 = Conv2D(1, 1, padding="same", name="out_col1")(x01)
    out2 = Conv2D(1, 1, padding="same", name="out_col2")(x02)
    out3 = Conv2D(1, 1, padding="same", name="out_col3")(x03)
    out4 = Conv2D(1, 1, padding="same", name="out_col4")(x04)

    avg = Average(name="ds_avg")([out1, out2, out3, out4])
    # FIX 2: dtype='float32' for segmentation output sigmoid
    seg_output = Activation("sigmoid", dtype="float32", name="seg_output")(avg)

    model = Model(inputs=inp, outputs=seg_output, name="UNetPP_4L")
    return model


# backward-compatible alias
Build_Unet_Model = Build_UnetPP_Model