from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    Conv2D, BatchNormalization, Activation, MaxPool2D,
    Conv2DTranspose, Concatenate, DepthwiseConv2D,
    Add, Lambda
)
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow as tf


# =====================================================
# Multi-Task Model: MobileNetV3-Small + SSD Lite Head
# =====================================================
def build_model(dropout_cls=0.3,
                dropout_reg=0.3,
                dense_scale=1.0,
                weights="imagenet"):

    sp = (224, 224, 3)
    input_layer = Input(shape=sp, name="Input")

    # ── Backbone: MobileNetV3-Small (lightweight, fast) ──
    backbone = MobileNetV3Small(
        include_top=False,
        weights=weights,
        input_shape=sp,
        include_preprocessing=False   # we handle normalization externally
    )

    # Extract multi-scale feature maps for SSD Lite
    feat_s1 = backbone.get_layer("expanded_conv_3_project_bn").output   # 28x28
    feat_s2 = backbone.get_layer("expanded_conv_8_project_bn").output   # 14x14
    feat_s3 = backbone.output                                            # 7x7

    backbone_model = Model(inputs=backbone.input,
                           outputs=[feat_s1, feat_s2, feat_s3],
                           name="MobileNetV3Small_MultiScale")

    s1, s2, s3 = backbone_model(input_layer)

    # =================================================
    # Classification Branch (MobileNetV3 → GAP → Dense)
    # =================================================
    f1 = GlobalAveragePooling2D(name="c_gap")(s3)

    x = Dense(int(256 * dense_scale), activation="relu", name="c_dense1")(f1)
    x = Dropout(dropout_cls, name="c_dropout1")(x)
    x = Dense(int(64 * dense_scale), activation="relu", name="c_dense2")(x)

    c_out = Dense(1, activation="sigmoid", name="c_final")(x)

    # =================================================
    # Bounding Box Branch: SSD Lite Style
    # Each scale predicts 4 box offsets via depthwise-separable convs
    # =================================================
    def ssd_lite_head(feat, name_prefix):
        """Depthwise-separable conv head — the 'Lite' in SSD Lite."""
        x = DepthwiseConv2D(3, padding="same", name=f"{name_prefix}_dw")(feat)
        x = BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
        x = Activation("relu6", name=f"{name_prefix}_dw_relu")(x)
        x = Conv2D(4, 1, padding="same", name=f"{name_prefix}_pw")(x)
        x = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        return x

    box_s1 = ssd_lite_head(s1, "ssd_s1")   # large-scale features
    box_s2 = ssd_lite_head(s2, "ssd_s2")   # mid-scale features
    box_s3 = ssd_lite_head(s3, "ssd_s3")   # small-scale features

    # FIX: Concatenate instead of Add.
    # Add() summed three (batch, 4) tensors, pushing values well outside [0,1]
    # before the final sigmoid — causing sigmoid saturation and the
    # "predict whole image" degenerate solution.
    # Concatenate gives the Dense layer a (batch, 12) input so it can learn
    # which scale to trust rather than blindly averaging all three.
    fused = Concatenate(name="ssd_fuse")([box_s1, box_s2, box_s3])  # shape (batch, 12)

    r = Dense(int(128 * dense_scale), activation="relu", name="b_dense1")(fused)
    r = Dropout(dropout_reg, name="b_dropout1")(r)

    b_out = Dense(4, activation="sigmoid", name="b_final")(r)

    model = Model(inputs=input_layer,
                  outputs=[c_out, b_out],
                  name="ColonSeg_Combo2")

    return model


# =====================================================
# U-Net Blocks (unchanged — reused in shallow U-Net)
# =====================================================
def double_conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# =====================================================
# Shallow U-Net — 3 Encoder Levels (Combination 2)
# =====================================================
def Build_Unet_Model(num_filters=32, input_shape=(224, 224, 3)):

    inputs = Input(input_shape)

    # ── Encoder (3 levels) ──
    f1 = double_conv_block(inputs, num_filters)
    p1 = MaxPool2D((2, 2))(f1)                         # 112x112

    f2 = double_conv_block(p1, num_filters * 2)
    p2 = MaxPool2D((2, 2))(f2)                         # 56x56

    f3 = double_conv_block(p2, num_filters * 4)
    p3 = MaxPool2D((2, 2))(f3)                         # 28x28

    # ── Bottleneck ──
    c = double_conv_block(p3, num_filters * 8)          # 28x28

    # ── Decoder (mirrors encoder — 3 levels) ──
    d1 = Conv2DTranspose(num_filters * 4, 2, strides=2, padding="same")(c)
    d1 = Concatenate()([d1, f3])
    d1 = double_conv_block(d1, num_filters * 4)

    d2 = Conv2DTranspose(num_filters * 2, 2, strides=2, padding="same")(d1)
    d2 = Concatenate()([d2, f2])
    d2 = double_conv_block(d2, num_filters * 2)

    d3 = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(d2)
    d3 = Concatenate()([d3, f1])
    d3 = double_conv_block(d3, num_filters)

    outputs = Conv2D(1, 1,
                     padding="same",
                     activation="sigmoid",
                     name="seg_output")(d3)

    model = Model(inputs, outputs, name="ShallowUNet_3L")

    return model