"""
base_models.py  —  Combination 3  (FIXED v2)
=============================================

FIX-B3 (new): Box head dense layer uses `dense_units` (not `dense_units//2`).
  When Hyperband chose dense_units=64, the box dense layer was only 32 units,
  which is insufficient to regress 4 normalised coordinates from 1280-dim
  EfficientNetB0 features.  Doubling it to `dense_units` costs negligible
  compute but eliminates the capacity bottleneck.

All previous fixes retained:
  FIX 1: Label docstring corrected (bleeding=1, non-bleeding=0).
  FIX 2: Output layers use dtype='float32' to prevent sigmoid saturation
         under mixed_float16 policy.
  FIX 3: build_model parameter is dense_units (not dense_scale).
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


def build_model(dropout_cls=0.3,
                dropout_reg=0.3,
                dense_units=256,
                weights="imagenet"):
    """
    Two-output Keras model:
      c_final : (batch, 1)  bleeding probability  (1=bleeding, 0=non-bleed)
      b_final : (batch, 4)  normalised [x1, y1, x2, y2] box

    FIX-B3: Box head first dense layer now uses `dense_units` (was dense_units//2).
    Hyperband minimum is dense_units=64 → box head was 32 units → too narrow.
    """
    sp          = (224, 224, 3)
    input_layer = Input(shape=sp, name="Input")

    x_scaled = Rescaling(scale=255.0, name="rescale_0_255")(input_layer)

    backbone = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=sp,
    )
    backbone.trainable = True

    # training=False here is overridden to True during Stage 1 via
    # the layer.trainable = True path in training_fixed.py (FIX-B2).
    features = backbone(x_scaled, training=False)   # (batch, 7, 7, 1280)

    # ── Classification Head ───────────────────────────────────
    c = GlobalAveragePooling2D(name="c_gap")(features)
    c = Dense(dense_units, activation="relu", name="c_dense1")(c)
    c = Dropout(dropout_cls, name="c_drop1")(c)
    c = Dense(dense_units // 4, activation="relu", name="c_dense2")(c)
    c_out = Dense(1, activation="sigmoid", dtype="float32", name="c_final")(c)

    # ── Bounding-Box Head ─────────────────────────────────────
    b = DepthwiseConv2D(3, padding="same", name="b_dw")(features)
    b = BatchNormalization(name="b_dw_bn")(b)
    b = Activation("relu6", name="b_dw_relu")(b)
    b = Conv2D(64, 1, padding="same", name="b_pw")(b)
    b = GlobalAveragePooling2D(name="b_gap")(b)

    # FIX-B3: use dense_units (not dense_units//2) for adequate capacity
    b = Dense(dense_units, activation="relu", name="b_dense1")(b)
    b = Dropout(dropout_reg, name="b_drop1")(b)
    # Extra dense layer for better coordinate regression
    b = Dense(dense_units // 2, activation="relu", name="b_dense2")(b)
    b_out = Dense(4, activation="sigmoid", dtype="float32", name="b_final")(b)

    model = Model(inputs=input_layer,
                  outputs=[c_out, b_out],
                  name="ColonSeg_Combo3_EfficientNet")
    return model


# =====================================================
# UNet++ (nested dense skip connections) — unchanged
# =====================================================

def _conv_block(x, filters, name):
    x = Conv2D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_r1")(x)
    x = Conv2D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_r2")(x)
    return x


def _up(x, filters, name):
    x = UpSampling2D(size=(2, 2), interpolation="bilinear",
                     name=f"{name}_ups")(x)
    x = Conv2D(filters, 1, padding="same", name=f"{name}_conv")(x)
    return x


def Build_UnetPP_Model(num_filters=32, input_shape=(224, 224, 3)):
    F  = num_filters
    Fs = [F, F*2, F*4, F*8, F*16]
    inp = Input(input_shape, name="input")

    x00 = _conv_block(inp,  Fs[0], "x00")
    p0  = MaxPool2D(2, name="pool0")(x00)
    x10 = _conv_block(p0,   Fs[1], "x10")
    p1  = MaxPool2D(2, name="pool1")(x10)
    x20 = _conv_block(p1,   Fs[2], "x20")
    p2  = MaxPool2D(2, name="pool2")(x20)
    x30 = _conv_block(p2,   Fs[3], "x30")
    p3  = MaxPool2D(2, name="pool3")(x30)
    x40 = _conv_block(p3,   Fs[4], "x40")

    x31 = _conv_block(Concatenate(name="cat31")([x30, _up(x40, Fs[3], "u4031")]), Fs[3], "x31")
    x21 = _conv_block(Concatenate(name="cat21")([x20, _up(x30, Fs[2], "u3021")]), Fs[2], "x21")
    x11 = _conv_block(Concatenate(name="cat11")([x10, _up(x20, Fs[1], "u2011")]), Fs[1], "x11")
    x01 = _conv_block(Concatenate(name="cat01")([x00, _up(x10, Fs[0], "u1001")]), Fs[0], "x01")

    x22 = _conv_block(Concatenate(name="cat22")([x20, x21, _up(x31, Fs[2], "u3122")]), Fs[2], "x22")
    x12 = _conv_block(Concatenate(name="cat12")([x10, x11, _up(x21, Fs[1], "u2112")]), Fs[1], "x12")
    x02 = _conv_block(Concatenate(name="cat02")([x00, x01, _up(x11, Fs[0], "u1102")]), Fs[0], "x02")

    x13 = _conv_block(Concatenate(name="cat13")([x10, x11, x12, _up(x22, Fs[1], "u2213")]), Fs[1], "x13")
    x03 = _conv_block(Concatenate(name="cat03")([x00, x01, x02, _up(x12, Fs[0], "u1203")]), Fs[0], "x03")

    x04 = _conv_block(
        Concatenate(name="cat04")([x00, x01, x02, x03, _up(x13, Fs[0], "u1304")]),
        Fs[0], "x04")

    out1 = Conv2D(1, 1, padding="same", name="out_col1")(x01)
    out2 = Conv2D(1, 1, padding="same", name="out_col2")(x02)
    out3 = Conv2D(1, 1, padding="same", name="out_col3")(x03)
    out4 = Conv2D(1, 1, padding="same", name="out_col4")(x04)

    avg        = Average(name="ds_avg")([out1, out2, out3, out4])
    seg_output = Activation("sigmoid", dtype="float32", name="seg_output")(avg)

    model = Model(inputs=inp, outputs=seg_output, name="UNetPP_4L")
    return model


Build_Unet_Model = Build_UnetPP_Model