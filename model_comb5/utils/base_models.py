"""
base_models.py  — Model Architectures
=============================================================
Models
  • build_model()   — EfficientNet-B0 backbone + YOLOv5n-style bbox head
                             + lightweight classification head
  • Build_AttUnet_Model()  — Attention U-Net (3 encoder levels, base_filters=32)

Design notes
  EfficientNet-B0 backbone
    - ImageNet pre-trained, include_top=False
    - Multi-scale features tapped at three intermediate blocks (P3/P4/P5)
      for the YOLOv5-nano-inspired detection head.

  YOLOv5n-style bbox head
    - Three feature scales fused via a tiny FPN (top-down only, 1×1 projections)
    - Each scale processed by a single C3-lite bottleneck
      (two 3×3 DWConv + 1×1 pointwise — matches YOLOv5n parameter budget)
    - Global-average-pooled, concatenated (batch, 12), then Dense(4, sigmoid)
    - Concatenate, NOT Add — same lesson learned in Combo 2 (sigmoid saturation
      from summing three (batch, 4) tensors that may be in different value ranges)

  Attention U-Net (light, base_filters=32)
    - Standard additive attention gate before each skip-connection merge
    - 3 encoder levels → bottleneck → 3 decoder levels (mirrors Combo 2 depth)
    - Keeps parameter count low while focusing the decoder on salient regions
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    Conv2D, BatchNormalization, Activation, MaxPool2D,
    Conv2DTranspose, Concatenate, DepthwiseConv2D,
    Multiply, Add, Lambda, Reshape
)
from tensorflow.keras.applications import EfficientNetB0


# ─────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────

def _bn_relu(x, name_prefix=""):
    x = BatchNormalization(name=f"{name_prefix}_bn" if name_prefix else None)(x)
    x = Activation("relu", name=f"{name_prefix}_relu" if name_prefix else None)(x)
    return x


def _dw_conv(x, filters, name_prefix):
    """Depthwise-separable convolution (BN+ReLU after each step)."""
    x = DepthwiseConv2D(3, padding="same", name=f"{name_prefix}_dw")(x)
    x = _bn_relu(x, f"{name_prefix}_dw")
    x = Conv2D(filters, 1, padding="same", name=f"{name_prefix}_pw")(x)
    x = _bn_relu(x, f"{name_prefix}_pw")
    return x


def _c3_lite(x, filters, name_prefix):
    """
    C3-lite bottleneck  (inspired by YOLOv5 C3 with depth=1).
    Two parallel branches:
      branch_1 : 1×1 conv → DW-sep conv
      branch_2 : 1×1 conv  (shortcut projection)
    Concatenated then fused with 1×1 conv.
    """
    half = max(filters // 2, 8)

    b1 = Conv2D(half, 1, padding="same", name=f"{name_prefix}_b1_proj")(x)
    b1 = _dw_conv(b1, half, f"{name_prefix}_b1")

    b2 = Conv2D(half, 1, padding="same", name=f"{name_prefix}_b2_proj")(x)

    out = Concatenate(name=f"{name_prefix}_cat")([b1, b2])
    out = Conv2D(filters, 1, padding="same", name=f"{name_prefix}_fuse")(out)
    out = _bn_relu(out, f"{name_prefix}_fuse")
    return out


# ─────────────────────────────────────────────────────────
# COMBINATION 1 DETECTION + CLASSIFICATION MODEL
# EfficientNet-B0 + YOLOv5n-style head
# ─────────────────────────────────────────────────────────

def build_model(dropout_cls=0.3,
                       dropout_reg=0.3,
                       dense_scale=1.0,
                       weights="imagenet"):
    """
    Returns a two-output Keras Model:
      c_final  : (batch, 1) — bleeding probability (sigmoid)
      b_final  : (batch, 4) — normalised [x1, y1, x2, y2] (sigmoid)

    Parameters
    ----------
    dropout_cls  : dropout rate applied in the classification head
    dropout_reg  : dropout rate applied in the bbox head
    dense_scale  : multiplier on intermediate Dense unit counts
    weights      : 'imagenet' or None (random init)
    """
    sp = (224, 224, 3)
    inp = Input(shape=sp, name="Input")

    # ── Backbone: EfficientNet-B0 ──────────────────────────
    # include_preprocessing=False: we normalise inputs externally (same as Combo 2)
    backbone = EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=sp,
        include_preprocessing=False,
    )

    # Three feature scales that give a P3/P4/P5-like pyramid:
    #   block3b_add  → 28×28  (stride-8)
    #   block5c_add  → 14×14  (stride-16)
    #   top_activation→  7×7  (stride-32)
    feat_p3 = backbone.get_layer("block3b_add").output       # 28×28, 40ch
    feat_p4 = backbone.get_layer("block5c_add").output       # 14×14, 112ch
    feat_p5 = backbone.output                                # 7×7,  1280ch

    backbone_model = Model(
        inputs=backbone.input,
        outputs=[feat_p3, feat_p4, feat_p5],
        name="EfficientNetB0_FPN",
    )
    p3, p4, p5 = backbone_model(inp)

    # ── Classification Branch ──────────────────────────────
    # Uses the richest (P5) feature map — same as Combo 2 using s3.
    cls_gap = GlobalAveragePooling2D(name="c_gap")(p5)
    cx = Dense(int(256 * dense_scale), activation="relu", name="c_dense1")(cls_gap)
    cx = Dropout(dropout_cls, name="c_dropout1")(cx)
    cx = Dense(int(64 * dense_scale), activation="relu", name="c_dense2")(cx)
    c_out = Dense(1, activation="sigmoid", dtype="float32", name="c_final")(cx)

    # ── YOLOv5n-style Bbox Head ────────────────────────────
    #
    # Tiny top-down FPN: project every scale to 64 ch, upsample and add.
    #   P5 (7×7)  → project → up ×2 → add P4
    #   P4+P5 (14×14) → project → up ×2 → add P3
    # Then each scale goes through a C3-lite block before GAP.

    FPN_CH = 64   # lightweight projection depth (matches YOLOv5n ethos)

    # Project each level to FPN_CH
    lat_p5 = Conv2D(FPN_CH, 1, padding="same", name="fpn_lat_p5")(p5)
    lat_p4 = Conv2D(FPN_CH, 1, padding="same", name="fpn_lat_p4")(p4)
    lat_p3 = Conv2D(FPN_CH, 1, padding="same", name="fpn_lat_p3")(p3)

    # Top-down merge: p5 → p4 → p3
    up_p5 = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_up_p5")(lat_p5)
    td_p4 = Add(name="fpn_td_p4")([up_p5, lat_p4])
    td_p4 = _c3_lite(td_p4, FPN_CH, "fpn_c3_p4")

    up_p4 = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_up_p4")(td_p4)
    td_p3 = Add(name="fpn_td_p3")([up_p4, lat_p3])
    td_p3 = _c3_lite(td_p3, FPN_CH, "fpn_c3_p3")

    # Per-scale box predictions (DW-sep head, same as Combo 2 SSD-lite)
    def _box_head(feat, name_prefix):
        x = DepthwiseConv2D(3, padding="same", name=f"{name_prefix}_dw")(feat)
        x = BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
        x = Activation("relu6", name=f"{name_prefix}_dw_relu")(x)
        x = Conv2D(4, 1, padding="same", name=f"{name_prefix}_pw")(x)
        x = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
        return x

    box_p3 = _box_head(td_p3, "yolo_p3")   # large objects
    box_p4 = _box_head(td_p4, "yolo_p4")   # medium objects
    box_p5 = _box_head(lat_p5, "yolo_p5")  # small objects

    # Concatenate → (batch, 12) — lets the Dense layer learn scale weighting
    # (NOT Add: same sigmoid-saturation fix as Combo 2's ssd_fuse)
    fused = Concatenate(name="yolo_fuse")([box_p3, box_p4, box_p5])

    rx = Dense(int(128 * dense_scale), activation="relu", name="b_dense1")(fused)
    rx = Dropout(dropout_reg, name="b_dropout1")(rx)
    b_out = Dense(4, activation="sigmoid", name="b_final")(rx)

    model = Model(
        inputs=inp,
        outputs=[c_out, b_out],
        name="ColonSeg",
    )
    return model


# ─────────────────────────────────────────────────────────
# ATTENTION GATE (Additive)
# ─────────────────────────────────────────────────────────

def _attention_gate(g, x, inter_ch, name_prefix):
    """
    Standard additive attention gate (Oktay et al., 2018).

    g  : gating signal from the coarser (decoder) level  — (H, W, C_g)
    x  : skip-connection feature map (encoder)           — (H, W, C_x)
    inter_ch : number of intermediate channels

    Returns x re-weighted by the spatial attention map alpha ∈ [0,1].
    """
    # Both g and x are projected to inter_ch before summation.
    theta_x = Conv2D(inter_ch, 1, padding="same",
                     name=f"{name_prefix}_theta_x")(x)
    phi_g   = Conv2D(inter_ch, 1, padding="same",
                     name=f"{name_prefix}_phi_g")(g)

    # If g is coarser than x spatially, upsample g to match x.
    # (With stride-2 pooling, g is 2× smaller → upsample by 2.)
    phi_g_up = tf.keras.layers.UpSampling2D(size=(2, 2),
                                             name=f"{name_prefix}_phi_up")(phi_g)

    add = Add(name=f"{name_prefix}_add")([theta_x, phi_g_up])
    add = Activation("relu", name=f"{name_prefix}_relu")(add)

    psi = Conv2D(1, 1, padding="same", activation="sigmoid",
                 name=f"{name_prefix}_psi")(add)

    # Broadcast alpha across the channel dimension of x
    attended = Multiply(name=f"{name_prefix}_reweight")([x, psi])
    return attended


# ─────────────────────────────────────────────────────────
# ATTENTION U-NET  (light, base_filters=32, 3 encoder levels)
# ─────────────────────────────────────────────────────────

def _double_conv(inputs, num_filters):
    """Standard double-conv block with BN+ReLU (identical to Combo 2 helper)."""
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def Build_AttUnet_Model(num_filters=32, input_shape=(224, 224, 3)):
    """
    Attention U-Net — 3 encoder levels, base_filters=32.

    Architecture
    ─────────────────────────────────────────────────────
    Encoder
      level 1: double_conv(32)  → pool → 112×112
      level 2: double_conv(64)  → pool →  56×56
      level 3: double_conv(128) → pool →  28×28

    Bottleneck
      double_conv(256)                   28×28

    Decoder (with attention gate on each skip)
      level 3: att_gate + double_conv(128)  56×56
      level 2: att_gate + double_conv(64)  112×112
      level 1: att_gate + double_conv(32)  224×224

    Output: Conv2D(1, sigmoid)  — binary segmentation mask
    ─────────────────────────────────────────────────────
    inter_ch for each attention gate is set to num_filters at that level
    (same as the skip-connection width), which keeps total parameters low.
    """
    inputs = Input(input_shape, name="seg_input")

    # ── Encoder ───────────────────────────────────────────
    f1 = _double_conv(inputs, num_filters)          # 224×224, 32ch
    p1 = MaxPool2D((2, 2))(f1)                      # 112×112

    f2 = _double_conv(p1, num_filters * 2)          # 112×112, 64ch
    p2 = MaxPool2D((2, 2))(f2)                      #  56×56

    f3 = _double_conv(p2, num_filters * 4)          #  56×56, 128ch
    p3 = MaxPool2D((2, 2))(f3)                      #  28×28

    # ── Bottleneck ────────────────────────────────────────
    c = _double_conv(p3, num_filters * 8)           #  28×28, 256ch

    # ── Decoder with Attention Gates ──────────────────────
    # Level 3: gating signal = bottleneck c (28×28)
    #          skip signal   = f3 (56×56)
    att3 = _attention_gate(
        g=c, x=f3,
        inter_ch=num_filters * 4,
        name_prefix="att3",
    )
    # Upsample bottleneck and concatenate with attended skip
    d3 = Conv2DTranspose(num_filters * 4, 2, strides=2, padding="same")(c)
    d3 = Concatenate()([d3, att3])
    d3 = _double_conv(d3, num_filters * 4)          #  56×56, 128ch

    # Level 2: gating signal = d3 (56×56)
    #          skip signal   = f2 (112×112)
    att2 = _attention_gate(
        g=d3, x=f2,
        inter_ch=num_filters * 2,
        name_prefix="att2",
    )
    d2 = Conv2DTranspose(num_filters * 2, 2, strides=2, padding="same")(d3)
    d2 = Concatenate()([d2, att2])
    d2 = _double_conv(d2, num_filters * 2)          # 112×112,  64ch

    # Level 1: gating signal = d2 (112×112)
    #          skip signal   = f1 (224×224)
    att1 = _attention_gate(
        g=d2, x=f1,
        inter_ch=num_filters,
        name_prefix="att1",
    )
    d1 = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(d2)
    d1 = Concatenate()([d1, att1])
    d1 = _double_conv(d1, num_filters)              # 224×224,  32ch

    # ── Segmentation output ───────────────────────────────
    outputs = Conv2D(
        1, 1,
        padding="same",
        activation="sigmoid",
        dtype="float32",
        name="seg_output",
    )(d1)

    model = Model(inputs, outputs, name="AttentionUNet_3L")
    return model