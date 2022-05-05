import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.layers import Layer
# from keras import initializations
from tensorflow.keras.initializers import glorot_uniform, Initializer
from tensorflow.keras import activations, initializers, constraints
# our layer will take input shape (nb_samples, 1)


# assume the inputs are connected to the layer nodes according to a pattern. The first node is connected to the first n inputs
# the second to the second n inputs and so on.
class Diagonal(Layer):
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
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
        print ('self.kernel_initializer', self.kernel_initializer)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),
                                      # initializer='uniform',
                                      initializer=self.kernel_initializer,
                                      trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Diagonal, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        n_features = x.shape[1]
        #print ('input dimensions {}'.format(x.shape))

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

class SparseTF(Layer):
    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer='glorot_uniform',
                 activation='tanh', use_bias=True,
                 bias_initializer='zeros', kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
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
                                             trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
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

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)