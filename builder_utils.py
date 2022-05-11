import itertools
import logging

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.layers import Dense, Dropout, Activation

# from data.pathways.pathway_loader import get_pathway_files
from data.pathways.reactome import ReactomeNetwork
from layers.custom_layers import Diagonal, SparseTF


def get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    print('pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    print('genes', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = list(pathways).index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    return df.T


def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print('filtered_map',  filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1
        ####

        filtered_map = filtered_map.fillna(0)
        print('filtered_map', filter_df.shape)
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def shuffle_genes_map(mapp):
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp


class PNet(tf.keras.Model):
    def __init__(self, features, genes, direction, activation, activation_decision, dropout, sparse, add_unk_genes, kernel_initializer='random_normal', use_bias=False,
             shuffle_genes=False):
        super(PNet, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        n_features = len(features)
        n_genes = len(genes)

        self.layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)

        self.dense1 = Dense(2, activation='linear', name='o_linear{}'.format(0))
        self.drop1 = Dropout(dropout, name='dropout_{}'.format(0))
        self.activation1 = Activation(activation=activation_decision, name='o{}'.format(1))


        maps = get_layer_maps(genes, 5, direction, add_unk_genes)

        mapp = maps[0]
        mapp = mapp.values
        if shuffle_genes in ['all', 'pathways']:
            mapp = shuffle_genes_map(mapp)
        n_genes, n_pathways = mapp.shape
        logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
        self.hidden_layer1 = SparseTF(n_pathways, mapp, activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=use_bias)

        self.decision1 = Dense(2, activation=activation_decision)
        self.dec_drop1 = Dropout(dropout)

        mapp = maps[1]
        mapp = mapp.values
        if shuffle_genes in ['all', 'pathways']:
            mapp = shuffle_genes_map(mapp)
        n_genes, n_pathways = mapp.shape

        logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))

        self.hidden_layer2 = SparseTF(n_pathways, mapp, activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=use_bias)

        self.decision2 = Dense(2, activation=activation_decision)
        self.dec_drop2 = Dropout(dropout)

        mapp = maps[2]
        mapp = mapp.values
        if shuffle_genes in ['all', 'pathways']:
            mapp = shuffle_genes_map(mapp)
        n_genes, n_pathways = mapp.shape
        logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
        # print 'map # ones {}'.format(np.sum(mapp))
        layer_name = 'h{}'.format(1)
        self.hidden_layer3 = SparseTF(n_pathways, mapp, activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=use_bias)

        self.decision3 = Dense(2, activation=activation_decision)
        self.dec_drop3 = Dropout(dropout)

        mapp = maps[3]
        mapp = mapp.values
        if shuffle_genes in ['all', 'pathways']:
            mapp = shuffle_genes_map(mapp)
        n_genes, n_pathways = mapp.shape
        logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
        self.hidden_layer4 = SparseTF(n_pathways, mapp, activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=use_bias)

        self.decision4 = Dense(2, activation=activation_decision)
        self.dec_drop4 = Dropout(dropout)

        mapp = maps[4]
        mapp = mapp.values
        if shuffle_genes in ['all', 'pathways']:
            mapp = shuffle_genes_map(mapp)
        n_genes, n_pathways = mapp.shape
        logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
        self.hidden_layer5 = SparseTF(n_pathways, mapp, activation=activation,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=use_bias)

        self.decision5 = Dense(2, activation=activation_decision)
        self.dec_drop5 = Dropout(dropout)


    def call(self, inputs, training=False):
        outcome = self.layer1(inputs)

        decision_outcomes = []

        decision_outcome = self.dense1(outcome)

        if (training):
            outcome = self.drop1(outcome, training=training)

        decision_outcome = self.activation1(decision_outcome)
        decision_outcomes.append(decision_outcome)

        outcome = self.hidden_layer1(outcome)

        # hidden block 1
        decision_outcome = self.decision1(outcome)
        decision_outcomes.append(decision_outcome)
        if (training):
            outcome = self.dec_drop1(outcome, training=training)

        #hidden block 2
        outcome = self.hidden_layer2(outcome)

        decision_outcome = self.decision2(outcome)
        decision_outcomes.append(decision_outcome)
        if (training):
            outcome = self.dec_drop2(outcome, training=training)

        #hidden block 3
        outcome = self.hidden_layer3(outcome)

        decision_outcome = self.decision3(outcome)
        decision_outcomes.append(decision_outcome)
        if (training):
            outcome = self.dec_drop3(outcome, training=training)

        #hidden block 4
        outcome = self.hidden_layer4(outcome)

        decision_outcome = self.decision4(outcome)
        decision_outcomes.append(decision_outcome)
        if (training):
            outcome = self.dec_drop4(outcome, training=training)

        #hidden block 5
        outcome = self.hidden_layer5(outcome)

        decision_outcome = self.decision5(outcome)
        decision_outcomes.append(decision_outcome)
        if (training):
            outcome = self.dec_drop5(outcome, training=training)

        return outcome, decision_outcomes

    def loss(self, probs, labels):
        loss_weights = np.exp(range(1, len(probs) + 1))

        losses = []
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        for i in range(len(probs)):
            losses.append(bce(labels, probs[i]))
        
        return tf.math.reduce_sum(tf.math.multiply(losses, loss_weights))