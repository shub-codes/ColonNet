import tensorflow as tf
from tensorflow.keras import backend as K

smooth = 1

def tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # FIX: guard against float16
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos   = K.sum(y_true_pos * y_pred_pos)
    false_neg  = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos  = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )

def focal_tversky(y_true, y_pred):
    pt_1  = tversky(y_true, y_pred)  # tversky now handles casting
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def dice_coef(y_true, y_pred):
    y_true = tf.cast(K.flatten(y_true), tf.float32)  # FIX: cast here too
    y_pred = tf.cast(K.flatten(y_pred), tf.float32)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        K.sum(y_true) + K.sum(y_pred) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)