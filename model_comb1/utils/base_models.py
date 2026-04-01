from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import DenseNet121


# =====================================================
# Multi-Task Model (Classification + Bounding Box)
# =====================================================
def build_model(dropout_cls=0.3,
                dropout_reg=0.3,
                dense_scale=1.0,
                weights="imagenet"):   # 🔥 default pretrained

    sp = (224, 224, 3)
    input_layer = Input(shape=sp, name="Input")

    # 🔥 Pretrained DenseNet backbone
    backbone = DenseNet121(include_top=False,
                           weights=weights,
                           input_shape=sp)

    base = backbone(input_layer)

    # =================================================
    # Classification Branch (Light & Clean)
    # =================================================
    f1 = GlobalAveragePooling2D(name="c_gap")(base)

    x = Dense(int(512 * dense_scale),
              activation='relu',
              name="c_dense1")(f1)

    x = Dropout(dropout_cls, name="c_dropout1")(x)

    x = Dense(int(128 * dense_scale),
              activation='relu',
              name="c_dense2")(x)

    c_out = Dense(1,
                  activation='sigmoid',
                  name="c_final")(x)


    # =================================================
    # Bounding Box Branch (LIGHTER VERSION 🔥)
    # =================================================
    f2 = GlobalMaxPooling2D(name="b_gmp")(base)

    r = Dense(int(512 * dense_scale),
              activation='relu',
              name="b_dense1")(f2)

    r = Dropout(dropout_reg, name="b_dropout1")(r)

    r = Dense(int(128 * dense_scale),
              activation='relu',
              name="b_dense2")(r)

    b_out = Dense(4,
                  activation='sigmoid',
                  name="b_final")(r)

    model = Model(inputs=input_layer,
                  outputs=[c_out, b_out],
                  name="ColonSeg")

    return model


# =====================================================
# U-Net Blocks
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
# Lighter U-Net (Default Filters = 32 🔥)
# =====================================================
def Build_Unet_Model(num_filters=32, input_shape=(224, 224, 3)):

    inputs = Input(input_shape)

    # Encoder
    f1 = double_conv_block(inputs, num_filters)
    p1 = MaxPool2D((2, 2))(f1)

    f2 = double_conv_block(p1, num_filters * 2)
    p2 = MaxPool2D((2, 2))(f2)

    f3 = double_conv_block(p2, num_filters * 4)
    p3 = MaxPool2D((2, 2))(f3)

    f4 = double_conv_block(p3, num_filters * 8)
    p4 = MaxPool2D((2, 2))(f4)

    # Bottleneck
    c = double_conv_block(p4, num_filters * 16)

    # Decoder
    d1 = Conv2DTranspose(num_filters * 8, 2, strides=2, padding="same")(c)
    d1 = Concatenate()([d1, f4])
    d1 = double_conv_block(d1, num_filters * 8)

    d2 = Conv2DTranspose(num_filters * 4, 2, strides=2, padding="same")(d1)
    d2 = Concatenate()([d2, f3])
    d2 = double_conv_block(d2, num_filters * 4)

    d3 = Conv2DTranspose(num_filters * 2, 2, strides=2, padding="same")(d2)
    d3 = Concatenate()([d3, f2])
    d3 = double_conv_block(d3, num_filters * 2)

    d4 = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(d3)
    d4 = Concatenate()([d4, f1])
    d4 = double_conv_block(d4, num_filters)

    outputs = Conv2D(1, 1,
                     padding="same",
                     activation="sigmoid",
                     name="seg_output")(d4)

    model = Model(inputs, outputs, name="LightUNet")

    return model