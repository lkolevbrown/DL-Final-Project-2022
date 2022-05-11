import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from builder_utils import *
from data_reader import *
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

def test_model(model, is_sparse, f1, acc, inputs, labels, validation=False):
    if is_sparse:
        _, outcomes = model(inputs)
    else:
        outcomes = model(inputs)

    if is_sparse:
        f1.update_state(labels, outcomes[-1])
        acc.update_state(labels, outcomes[-1])
    else:
        f1.update_state(labels, outcomes)
        acc.update_state(labels, outcomes)

    f1_score = f1.result().numpy() 
    acc_score = acc.result().numpy()
    if not validation:
        print(f"F1 Score: {f1_score}")
        print(f"Accuracy: {acc_score}")
    return f1_score, acc_score

def cv(k_fold_num, model, is_sparse, f1, acc, inputs_train, labels_train, inputs_validate, labels_validate, batch_size):
    kfold = KFold(n_splits = k_fold_num, shuffle=True)
    # check iteration of the fold
    f1_list_0 = []
    f1_list_1 = []
    accuracies = []
    fold_no = 0
    losses = []
    for train, test in kfold.split(inputs_train, labels_train):
        print(f"Fold Number: {fold_no}")
        #train model on non-validation splits
        loss = train_model(model,is_sparse,tf.gather(inputs_train,train),tf.gather(labels_train,train), batch_size)
        losses.append(loss)

        #test model on validation split
        f1_scores, acc_score = test_model(model,is_sparse,f1,acc,tf.gather(inputs_train,test),tf.gather(labels_train,test))
        f1_list_0.append(f1_scores[0])
        f1_list_1.append(f1_scores[1])
        accuracies.append(acc_score)
        fold_no += 1
        
    #test model on given validation set
    validation_f1, validation_acc = test_model(model, is_sparse, f1, acc, inputs_validate, labels_validate, validation=True)
    print(f"Independent Validation F1 Score: {validation_f1}")
    print(f"Independent Validation Accurcacy: {validation_acc}")
        
    return tf.reduce_mean(f1_list_0),tf.reduce_mean(f1_list_1), tf.reduce_mean(accuracies), losses

def main(args):
    paperData = ProstateDataPaper(data_type='mut_important')

    x_train, x_validate, x_test, y_train, y_validate, y_test, info_train, info_validate, info_test, cols = paperData.get_train_validate_test()
    y_train = tf.one_hot(tf.convert_to_tensor(y_train), 2)
    y_validate = tf.one_hot(tf.convert_to_tensor(y_validate), 2)
    y_test = tf.one_hot(tf.convert_to_tensor(y_test), 2)

    if args.is_sparse:
        model = PNet(cols, cols, 'root_to_leaf', 'tanh', 'sigmoid', 0, True, False)
    else:
        model = Dense(len(cols))

    losses = []

    f1 = tfa.metrics.F1Score(num_classes=2)
    acc = tf.keras.metrics.Accuracy()

    #USING CROSS VAL:
    for epoch_id in tqdm(range(args.num_epochs)):
        avg_f1_0, avg_f1_1, avg_acc, epoch_loss = cv(5, model, args.is_sparse, f1, acc, x_train, y_train, x_validate, y_validate, args.batch_size)
        print(f"Average F1 Score for class 0 is {round(avg_f1_0.numpy(), 4)} and for class 1 is {round(avg_f1_1.numpy(), 4)}")
        print(f"Average Accuracy is {round(avg_acc.numpy(), 4)}")
        losses.extend(epoch_loss)

    test_f1, test_acc = test_model(model, args.is_sparse, f1, acc, x_test, y_test, validation=True)
    print(f"Testing F1 Score for class 0 is {round(test_f1[0], 4)} and for class 1 is {round(test_f1[1], 4)}")
    print(f"Testing Accuracy is {round(test_acc, 4)}")
    print(f"Losses across the entire training rountine: {losses}")

if __name__ == "__main__":
    args = parseArguments()
    main(args)
