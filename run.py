import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.math import sigmoid
from tqdm import tqdm
from builder_utils import *
from data_reader import *
import tensorflow as tf
from dense import *
from sklearn.model_selection import KFold


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

        f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

        if is_sparse:
            f1.update_state(y_batch, outcomes[-1])
            print(f1.result().numpy())
        else:
            f1.update_state(y_batch, outcomes)
            print(f1.result().numpy())

    return total_loss / num_batches

def test_model(model, inputs, labels, batch_size):
    
    num_batches = len(inputs) // batch_size

    for batch_num in range(num_batches):
        x_batch = inputs[batch_num * batch_size:batch_num * batch_size + batch_size + 1]
        y_batch = labels[batch_num * batch_size:batch_num * batch_size + batch_size + 1]

        outcomes = model(x_batch)
        loss = model.loss(outcomes, y_batch)      

    pass 

def cv(k_fold_num, model,is_sparse, train, test, inputs, labels, batch_size):
    kfold = KFold(n_splits = k_fold_num, shuffle=True)
    # check iteration of the fold
    acc_list = []
    fold_no = 1 
    for train, test, in kfold.split(inputs, labels):
        train_model(model,is_sparse,inputs[train],labels[train], batch_size)
        accuracy = test_model(model,inputs[test],labels[test],batch_size)
        acc_list.append[accuracy]
        print(f"Accuracy of {fold_no} is {accuracy}")
    
    return tf.reduce_mean(tf.constant(acc_list))

def main(args):
    paperData = ProstateDataPaper()
    x, response, samples, cols = load_data('P1000_final_analysis_set_cross_hotspots.csv')
    x = tf.convert_to_tensor(x)
    response = tf.one_hot(tf.convert_to_tensor(response), 2)

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
