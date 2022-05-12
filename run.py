import argparse
import math
import numpy as np
import os
import random
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from pnet import *
from data_reader import *
from dense import *
from sklearn.model_selection import KFold
from pylab import *


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_sparse", action="store_true")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    return args

def visualize_data(data, x_label, y_label, graph_title):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes
    """

    x_values = np.arange(0, len(data), 1)
    y_values = data
    plot(x_values, y_values)
    xlabel(x_label)
    ylabel(y_label)
    title(graph_title)
    grid(True)
    show()

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

    f1.reset_state()
    acc.reset_state()
    return f1_score, acc_score

def cv(k_fold_num, model, is_sparse, f1, acc, inputs_train, labels_train, batch_size):
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
        losses.append(loss.numpy())

        #test model on validation split
        f1_scores, acc_score = test_model(model,is_sparse,f1,acc,tf.gather(inputs_train,test),tf.gather(labels_train,test))
        f1_list_0.append(f1_scores[0])
        f1_list_1.append(f1_scores[1])
        accuracies.append(acc_score)
        fold_no += 1
        
    return f1_list_0, f1_list_1, accuracies, losses

def main(args):
    paperData = ProstateDataPaper(data_type='mut_important')

    x_train, x_test, y_train, y_test, info_train, info_test, cols = paperData.get_train_validate_test()
    y_train = tf.one_hot(tf.convert_to_tensor(y_train), 2)
    y_test = tf.one_hot(tf.convert_to_tensor(y_test), 2)

    if args.is_sparse:
        model = PNet(cols, cols, 'root_to_leaf', 'tanh', 'sigmoid', 0, True, False)
    else:
        model = Dense(len(cols))

    losses = []
    valid_f1s_0 = []
    valid_f1s_1 = []
    valid_accs = []

    f1 = tfa.metrics.F1Score(num_classes=2)
    acc = tf.keras.metrics.Accuracy()

    #USING CROSS VAL:
    for epoch_id in tqdm(range(args.num_epochs)):
        valid_f1_0, valid_f1_1, valid_acc, epoch_loss = cv(5, model, args.is_sparse, f1, acc, x_train, y_train, args.batch_size)
        print(f"Average F1 Score for class 0 is {round(tf.reduce_mean(valid_f1_0).numpy(), 4)} and for class 1 is {round(tf.reduce_mean(valid_f1_1).numpy(), 4)}")
        print(f"Average Accuracy is {round(tf.reduce_mean(valid_acc).numpy(), 4)}")
        losses.append(epoch_loss[-1])
        valid_f1s_0.append(valid_f1_0[-1])
        valid_f1s_1.append(valid_f1_1[-1])
        valid_accs.append(valid_acc[-1])

    test_f1, test_acc = test_model(model, args.is_sparse, f1, acc, x_test, y_test, validation=True)
    print(f"Testing F1 Score for class 0 is {round(test_f1[0], 4)} and for class 1 is {round(test_f1[1], 4)}")
    print(f"Testing Accuracy is {round(test_acc, 4)}")
    print(f"Losses across the entire training rountine: {losses}")

    visualize_data(valid_f1s_0, "Epoch Num", "F1 Score for Primary Cancers", "F1 Score for Class 0 Over Epochs")
    visualize_data(valid_f1s_1, "Epoch Num", "F1 Score for Metastatic Cancers", "F1 Score for Class 1 Over Epochs")
    visualize_data(valid_accs, "Epoch Num", "Model Accuracy", "Accuracy Over Epochs")
    visualize_data(losses, "Epoch Num", "Model Loss", "Binary Cross Entropy Loss Over Training Routine")


if __name__ == "__main__":
    args = parseArguments()
    main(args)
