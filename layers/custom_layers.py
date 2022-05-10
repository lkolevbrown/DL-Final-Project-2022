import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras.layers import Layer
# from keras import initializations
from tensorflow.keras.initializers import Initializer
from tensorflow.keras import activations, initializers, constraints
# our layer will take input shape (nb_samples, 1)


# assume the inputs are connected to the layer nodes according to a pattern. The first node is connected to the first n inputs
# the second to the second n inputs and so on. n is equal to the input dim // output dim
class Diagonal(Layer):
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='random_normal',
                 bias_initializer='zeros',
                 **kwargs):
        super(Diagonal, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)
        print ('input dimension {} self.units {}'.format(input_dimension, self.units))
        self.n_inputs_per_node = input_dimension // self.units
        print ('n_inputs_per_node {}'.format(self.n_inputs_per_node))

        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,input_dimension),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None

        super(Diagonal, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        mult = x * self.kernel
        mult = tf.reshape(mult, (-1, self.n_inputs_per_node))
        mult = tf.math.reduce_sum(mult, axis=1)
        output = tf.reshape(mult, (-1, self.units))

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

class SparseTF(Layer):
    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer='random_normal',
                 activation='tanh', use_bias=True,
                 bias_initializer='zeros',
                 **kwargs):
        super(SparseTF, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.activation_fn = activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[1]
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
                                             trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
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