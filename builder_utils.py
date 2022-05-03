import itertools
import logging

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from tensorflow.keras.regularizers import L2

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
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # logging.info('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp


class PNet(tf.keras.Model):
    def __init__(self, features, genes, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
        super(PNet, self).__init__()
        n_features = len(features)
        n_genes = len(genes)

        if not type(w_reg) == list:
            w_reg = [w_reg] * 10

        if not type(w_reg_outcomes) == list:
            w_reg_outcomes = [w_reg_outcomes] * 10

        if not type(dropout) == list:
            dropout = [w_reg_outcomes] * 10

        w_reg0 = w_reg[0]
        w_reg_outcome0 = w_reg_outcomes[0]
        w_reg_outcome1 = w_reg_outcomes[1]
        constraints = {}

        self.layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=L2(w_reg0),
                                  use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)

        self.dense1 = Dense(1, activation='linear', name='o_linear{}'.format(0), activity_regularizer=L2(w_reg_outcome0))
        self.drop1 = Dropout(dropout[0], name='dropout_{}'.format(0))
        self.activation1 = Activation(activation=activation_decision, name='o{}'.format(1))


        mapp = get_layer_maps(genes, 1, direction, add_unk_genes)[0]

        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]

        w_reg = w_regs[1]
        w_reg_outcome = w_reg_outcomes[1]
        # dropout2 = dropouts[i]
        dropout = dropouts[1]
        names = mapp.index
        # names = list(mapp.index)
        mapp = mapp.values
        if shuffle_genes in ['all', 'pathways']:
            mapp = shuffle_genes_map(mapp)
        n_genes, n_pathways = mapp.shape
        logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
        # print 'map # ones {}'.format(np.sum(mapp))
        print ('layer {}, dropout  {} w_reg {}'.format(1, dropout, w_reg))
        layer_name = 'h{}'.format(1)
        self.hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=L2(w_reg),
                                    name=layer_name, kernel_initializer=kernel_initializer,
                                    use_bias=use_bias, **constraints)

        self.dense2 = Dense(1, activation='linear', name='o_linear{}'.format(1 + 2), activity_regularizer=L2(w_reg_outcome))
        self.activation2 = Activation(activation=activation_decision, name='o{}'.format(1 + 2))
        self.drop2 = Dropout(dropout, name='dropout_{}'.format(1 + 1))


    def call(self, inputs, training=False):
        outcome = self.layer1(inputs)

        decision_outcomes = []

        decision_outcome = self.dense1(outcome)

        if (training):
            outcome = self.drop1(outcome, training=training)

        decision_outcome = self.activation1(decision_outcome)
        decision_outcomes.append(decision_outcome)

        outcome = self.hidden_layer(outcome)

        decision_outcome = self.dense2(outcome)
        decision_outcome = self.activation2(decision_outcome)
        decision_outcomes.append(decision_outcome)
        if (training):
            outcome = self.drop2(outcome, training=dropout_testing)

        return outcome, decision_outcomes

    def loss(self, probs, labels):
        pass


def get_pnet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not type(w_reg) == list:
        w_reg = [w_reg] * 10

    if not type(w_reg_outcomes) == list:
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not type(dropout) == list:
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    w_reg_outcome0 = w_reg_outcomes[0]
    w_reg_outcome1 = w_reg_outcomes[1]
    reg_l = l2
    constraints = {}

    layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)

    outcome = layer1(inputs)

    decision_outcomes = []

    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(0), W_regularizer=reg_l(w_reg_outcome0))(
        inputs)
    
    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(1),
                             W_regularizer=reg_l(w_reg_outcome1 / 2.))(outcome)

    drop2 = Dropout(dropout[0], name='dropout_{}'.format(0))

    outcome = drop2(outcome, training=dropout_testing)

    # testing
    if batch_normal:
        decision_outcome = BatchNormalization()(decision_outcome)

    decision_outcome = Activation(activation=activation_decision, name='o{}'.format(1))(decision_outcome)
    decision_outcomes.append(decision_outcome)

    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        layer_inds = range(1, len(maps))
        
        print ('original dropout', dropout)
        print ('dropout', layer_inds, dropout, w_reg)
        w_regs = w_reg[1:]
        w_reg_outcomes = w_reg_outcomes[1:]
        dropouts = dropout[1:]
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            w_reg_outcome = w_reg_outcomes[i]
            # dropout2 = dropouts[i]
            dropout = dropouts[1]
            names = mapp.index
            # names = list(mapp.index)
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
            # print 'map # ones {}'.format(np.sum(mapp))
            print ('layer {}, dropout  {} w_reg {}'.format(i, dropout, w_reg))
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=reg_l(w_reg),
                                        name=layer_name, kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=reg_l(w_reg),
                                     name=layer_name, kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)

            decision_outcome = Dense(1, activation='linear', name='o_linear{}'.format(i + 2),
                                     W_regularizer=reg_l(w_reg_outcome))(outcome)

            if batch_normal:
                decision_outcome = BatchNormalization()(decision_outcome)
            decision_outcome = Activation(activation=activation_decision, name='o{}'.format(i + 2))(decision_outcome)
            decision_outcomes.append(decision_outcome)
            drop2 = Dropout(dropout, name='dropout_{}'.format(i + 1))
            outcome = drop2(outcome, training=dropout_testing)

            feature_names['h{}'.format(i)] = names
        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
    return outcome, decision_outcomes, feature_names