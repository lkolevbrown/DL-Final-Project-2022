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
                _, outcomes = model(x_batch,training=True)
            else:
                outcomes = model(x_batch,training=True)
            loss = model.loss(outcomes, y_batch)
            total_loss = total_loss + loss

        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        model.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return total_loss / num_batches

def test_model(model, is_sparse, metric, inputs, labels, validation=False):
    if is_sparse:
        _, outcomes = model(inputs)
    else:
        outcomes = model(inputs)

    if is_sparse:
        metric.update_state(labels, outcomes[-1])
    else:
        metric.update_state(labels, outcomes)

    f1_score = metric.result().numpy() 
    if not validation:
        print(f"F1 Score: {f1_score}")
    return f1_score

def cv(k_fold_num, model, is_sparse, metric, inputs_train, labels_train, inputs_validate, labels_validate, batch_size):
    kfold = KFold(n_splits = k_fold_num, shuffle=True)
    # check iteration of the fold
    acc_list_0 = []
    acc_list_1 = []
    fold_no = 0
    losses = []
    for train, test in kfold.split(inputs_train, labels_train):
        loss = train_model(model,is_sparse,tf.gather(inputs_train,train),tf.gather(labels_train,train), batch_size)
        losses.append(loss)
        accuracies = test_model(model,is_sparse,metric,tf.gather(inputs_train,test),tf.gather(labels_train,test))
        acc_list_0.append(accuracies[0])
        acc_list_1.append(accuracies[1])
        fold_no += 1
        validation_score = test_model(model, is_sparse, metric, inputs_validate, labels_validate, validation=True)
        print(f"Validation F1 Score: {validation_score}")
    
    return tf.reduce_mean(tf.constant(acc_list_0)),tf.reduce_mean(tf.constant(acc_list_1)), losses

def main(args):
    paperData = ProstateDataPaper(data_type='mut_important')

    #x, y, info, cols = paperData.get_data()
    #y = tf.one_hot(tf.convert_to_tensor(y), 2)
    x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, cols = paperData.get_train_validate_test()
    y_train = tf.one_hot(tf.convert_to_tensor(y_train), 2)
    y_validate = tf.one_hot(tf.convert_to_tensor(y_validate), 2)
    y_test = tf.one_hot(tf.convert_to_tensor(y_test), 2)

    if args.is_sparse:
        model = PNet(cols, cols, 'root_to_leaf', 'tanh', 'sigmoid', 0, True, False)
    else:
        model = Dense(len(cols))

    losses = []

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

    f1 = tfa.metrics.F1Score(num_classes=2)

    #USING CROSS VAL:
    for epoch_id in tqdm(range(args.num_epochs)):
        ac0, ac1, epoch_loss = cv(5, model, args.is_sparse, f1, x_train, y_train, x_validate, y_validate, args.batch_size)
        print(f"Average F1 Score for class 0 is {round(ac0.numpy(), 4)} and for class 1 is {round(ac1.numpy(), 4)}")
        losses.extend(epoch_loss)

    test_f1 = test_model(model, args.is_sparse, f1, x_test, y_test, validation=True)
    print(f"Testing F1 Score for class 0 is {round(test_f1[0], 4)} and for class 1 is {round(test_f1[1], 4)}")
    print(f"Losses across the entire training rountine: {losses}")

if __name__ == "__main__":
    args = parseArguments()
    main(args)
