import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.math import sigmoid
from tqdm import tqdm
from builder_utils import *
from data_reader import load_data
import tensorflow as tf
from dense import *


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_sparse", action="store_true")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    return args

def train_model(model, is_sparse, inputs, labels, batch_size):
    total_loss = 0

    num_batches = len(inputs) // batch_size

    for batch_num in range(num_batches):
        x_batch = inputs[batch_num * batch_size:batch_num * batch_size + batch_size + 1]
        y_batch = labels[batch_num * batch_size:batch_num * batch_size + batch_size + 1]

        with tf.GradientTape() as tape:
            if is_sparse:
                _, outcomes = model(x_batch)
            else:
                outcomes = model(x_batch)
            loss = model.loss(outcomes, y_batch)
            total_loss = total_loss + loss

        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        model.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if is_sparse:
            print(f1(y_batch, outcomes[-1]).numpy())
        else:
            print(f1(y_batch, outcomes).numpy())

    return total_loss / num_batches

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
        possible_positives = tf.math.reduce_sum(tf.math.multiply(tf.ones(shape=y_true.shape), y_true))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
        predicted_positives = tf.math.reduce_sum(tf.math.multiply(tf.ones(shape=y_pred.shape), y_pred))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.math.round(y_pred)

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def main(args):
    x, response, samples, cols = load_data('P1000_final_analysis_set_cross_hotspots.csv')
    x = tf.convert_to_tensor(x)
    response = tf.convert_to_tensor(response)

    if args.is_sparse:
        model = PNet(cols, cols, 'root_to_leaf', 'tanh', 'softmax', 0, True, False, 'lecun_uniform')
    else:
        model = Dense(len(cols))

    losses = []

    # Train VAE
    for epoch_id in tqdm(range(args.num_epochs)):
        inds = tf.range(start = 0, limit = len(response))
        shuffled = tf.random.shuffle(inds)
        shuffled_inputs = tf.gather(x, shuffled) 
        shuffled_labels = tf.gather(response, shuffled)
        total_loss = train_model(model, args.is_sparse, shuffled_inputs, shuffled_labels, args.batch_size)
        #print(f"Train Epoch: {epoch_id}")
        losses.append(total_loss)

    print(losses)

if __name__ == "__main__":
    args = parseArguments()
    main(args)
