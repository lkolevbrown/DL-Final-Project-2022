import numpy as np
import pandas as pd
import tensorflow as tf


class Dense(tf.keras.Model):
    def __init__(self, num_cols):
        super(Dense, self).__init__()
        #self.learning_rate = 0.001
        self.learning_rate = 0.1
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (None, num_cols)),
            #tf.keras.layers.Dense(8192, activation=tf.keras.layers.LeakyReLU()),
            #tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU()),
            tf.keras.layers.Dense(2, activation='softmax')])


    def call(self, inputs):
        return self.dense(inputs)

    def loss(self, probs, labels):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(labels, probs)