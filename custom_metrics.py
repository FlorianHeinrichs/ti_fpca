import tensorflow as tf


class NegativeMSE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mse = - tf.reduce_mean(tf.square(y_pred-y_true))
        return mse


class NegativeMAE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mae = - tf.reduce_mean(tf.abs(y_pred-y_true))
        return mae
