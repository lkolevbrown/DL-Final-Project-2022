import tensorflow as tf
import keras
import numpy as np
from keras import regularizers
from tensorflow.keras.layers import Layer
# from keras import initializations
from tensorflow.keras.initializers import glorot_uniform, Initializer
from tensorflow.keras import activations, initializers, constraints
# our layer will take input shape (nb_samples, 1)
from tensorflow.keras.regularizers import Regularizer


# assume the inputs are connected to the layer nodes according to a pattern. The first node is connected to the first n inputs
# the second to the second n inputs and so on.
class Diagonal(Layer):
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        # self.output_dim = output_dim
        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.W_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = bias_constraint
        super(Diagonal, self).__init__(**kwargs)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)
        print ('input dimension {} self.units {}'.format(input_dimension, self.units))
        self.n_inputs_per_node = input_dimension // self.units
        print ('n_inputs_per_node {}'.format(self.n_inputs_per_node))

        rows = np.arange(input_dimension)
        cols = np.arange(self.units)
        cols = np.repeat(cols, self.n_inputs_per_node)
        self.nonzero_ind = np.column_stack((rows, cols))

        # print 'self.nonzero_ind', self.nonzero_ind
        print ('self.kernel_initializer', self.W_regularizer, self.kernel_initializer, self.kernel_regularizer)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),
                                      # initializer='uniform',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Diagonal, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        n_features = x.shape[1]
        print ('input dimensions {}'.format(x.shape))

        kernel = tf.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        mult = tf.reshape(mult, (-1, self.n_inputs_per_node))
        mult = tf.math.reduce_sum(mult, axis=1)
        output = tf.reshape(mult, (-1, self.units))

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        # config = {
        #         'units': self.units, 'activation':self.activation,
        # 'kernel_shape': self.kernel_shape, 'nonzero_ind':self.nonzero_ind, 'n_inputs_per_node': self.n_inputs_per_node }

        config = {

            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            # 'W_regularizer' : self.W_regularizer,
            # 'bias_regularizer' : self.bias_regularizer,

        }
        # 'kernel_initializer' : self.kernel_initializer,
        # 'bias_initializer' : self.bias_initializer,
        # 'W_regularizer' : ,
        # 'bias_regularizer' : None
        # 'kernel_shape': self.kernel_shape
        # dsve
        base_config = super(Diagonal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SparseTF(Layer):
    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True,
                 bias_initializer='zeros', bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation_fn = activations.get(activation)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(SparseTF, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        # random sparse constarints on the weights
        # if self.map is None:
        #     mapp = np.random.rand(input_dim, self.units)
        #     mapp = mapp > 0.9
        #     mapp = mapp.astype(np.float32)
        #     self.map = mapp
        # else:
        if not self.map is None:
            self.map = self.map.astype(np.float32)

        # can be initialized directly from (map) or using a loaded nonzero_ind (useful for cloning models or create from config)
        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.map)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)
        nonzero_count = self.nonzero_ind.shape[0]

        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SparseTF, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        tt = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        
        output = tf.linalg.matmul(inputs, tt)
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            # 'kernel_shape': self.kernel_shape,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
            # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
            # 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(SparseTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def call(self, inputs):
    #     print self.kernel.shape, inputs.shape
    #     tt= tf.sparse.transpose(self.kernel)
    #     output = tf.sparse.matmul(tt, tf.transpose(inputs ))
    #     return tf.transpose(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

from scipy.sparse import csr_matrix


class RandomWithMap(Initializer):
    """Initializer that generates tensors initialized to random array.
    """

    def __init__(self, mapp):
        self.map = mapp

    def __call__(self, shape, dtype=None):
        map_sparse = csr_matrix(self.map)
        # init = np.random.rand(*map_sparse.data.shape)
        init = np.random.normal(10.0, 1., *map_sparse.data.shape)
        print ('connection map data shape {}'.format(map_sparse.data.shape))
        # init = np.random.randn(*map_sparse.data.shape).astype(np.float32) * np.sqrt(2.0 / (map_sparse.data.shape[0]))
        initializers.glorot_uniform().__call__()
        map_sparse.data = init
        return K.variable(map_sparse.toarray())


class L1L2_with_map(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, mapp, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.connection_map = mapp

    def __call__(self, x):

        # x_masked = x *self.connection_map.astype(theano.config.floatX)
        x_masked = x * self.connection_map.astype(K.floatx())
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x_masked))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x_masked))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}


from keras import backend as K


# taken from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))