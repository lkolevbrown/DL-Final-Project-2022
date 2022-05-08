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

    return total_loss / num_batches

def test_model(model, is_sparse, inputs, labels):
    if is_sparse:
        _, outcomes = model(inputs)
    else:
        outcomes = model(inputs)

    f1 = tfa.metrics.F1Score(num_classes=2, threshold=0.5)

    if is_sparse:
        f1.update_state(labels, outcomes[-1])
    else:
        f1.update_state(labels, outcomes)

    f1_score = f1.result().numpy() 
    print(f"F1 Score: {f1_score}")
    return f1.result().numpy() 

def cv(k_fold_num, model, is_sparse, inputs, labels, batch_size):
    kfold = KFold(n_splits = k_fold_num, shuffle=True)
    # check iteration of the fold
    acc_list_0 = []
    acc_list_1 = []
    fold_no = 0
    for train, test in kfold.split(inputs, labels):
        train_model(model,is_sparse,tf.gather(inputs,train),tf.gather(labels,train), batch_size)
        accuracies = test_model(model,is_sparse,tf.gather(inputs,test),tf.gather(labels,test))
        acc_list_0.append(accuracies[0])
        acc_list_1.append(accuracies[1])
        fold_no += 1
    
    return tf.reduce_mean(tf.constant(acc_list_0)),tf.reduce_mean(tf.constant(acc_list_1)) 

def main(args):
    paperData = ProstateDataPaper(data_type='mut_important')
    #x, response, samples, cols = load_data('P1000_final_analysis_set_cross_hotspots.csv')
    #x = tf.convert_to_tensor(x)
    #response = tf.one_hot(tf.convert_to_tensor(response), 2)

    x, y, info, cols = paperData.get_data()
    y = tf.one_hot(tf.convert_to_tensor(y), 2)
    #x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, columns = paperData.get_train_validate_test()

    if args.is_sparse:
        model = PNet(cols, cols, 'root_to_leaf', 'tanh', 'softmax', 0, True, False, 'lecun_uniform')
    else:
        model = Dense(len(cols))

    #losses = []

    #THIS IS A MANUAL TRAINING ROUTINE
    # Train VAE
    # for epoch_id in tqdm(range(args.num_epochs)):
    #     inds = tf.range(start = 0, limit = len(response))
    #     shuffled = tf.random.shuffle(inds)
    #     shuffled_inputs = tf.gather(x_train, shuffled) 
    #     shuffled_labels = tf.gather(y_train, shuffled)
    #     total_loss = train_model(model, args.is_sparse, shuffled_inputs, shuffled_labels, args.batch_size)
    #     #print(f"Train Epoch: {epoch_id}")
    #     losses.append(total_loss)
    # print(losses)

    #USING CROSS VAL:
    ac0, ac1 = cv(5, model, args.is_sparse, x, y, args.batch_size)
    print(f"Average F1 Score for class 0 is {ac0} and for class 1 is {ac1}")


if __name__ == "__main__":
    args = parseArguments()
    main(args)
