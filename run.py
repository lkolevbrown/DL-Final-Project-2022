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


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    return args

def train_pnet(pnet, inputs, labels, batch_size):
    total_loss = 0

    num_batches = len(inputs) // batch_size

    for batch_num in range(num_batches):
        x_batch = inputs[batch_num * batch_size:batch_num * batch_size + batch_size + 1]
        y_batch = labels[batch_num * batch_size:batch_num * batch_size + batch_size + 1]

        with tf.GradientTape() as tape:
            _, outcomes = pnet(x_batch)
            loss = pnet.loss(outcomes, y_batch)
            total_loss = total_loss + loss

        trainable_vars = pnet.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        pnet.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return total_loss / num_batches

def main(args):
    x, response, samples, cols = load_data('P1000_final_analysis_set_cross_hotspots.csv')

    pnet = PNet(cols, cols, 'root_to_leaf', 'tanh', 'sigmoid', 0, True, False, 'lecun_uniform')

    losses = []

    # Train VAE
    for epoch_id in tqdm(range(args.num_epochs)):
        total_loss = train_pnet(pnet, tf.convert_to_tensor(x), tf.convert_to_tensor(response), args.batch_size)
        #print(f"Train Epoch: {epoch_id}")
        losses.append(total_loss)

    print(losses)

if __name__ == "__main__":
    args = parseArguments()
    main(args)
