import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import backend as K

class CustomRNNCell(tf.keras.layers.Layer):

    def __init__(self,
                 output_size,
                 state_size,
                 activation='tanh',
                 use_bias=True,
                 **kwargs):
        super(CustomRNNCell, self).__init__(**kwargs)
        self.output_size = output_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.state_size = state_size

    def build(self, input_shape):
        self.sx_kernel = self.add_weight(shape=(input_shape[-1], self.state_size),
                                         initializer='uniform',
                                         name='sx_kernel')
        self.ss_kernel = self.add_weight(shape=(self.state_size, self.state_size),
                                         initializer='orthogonal',
                                         name='ss_kernel')
        self.kernel = self.add_weight(shape=(self.state_size, self.output_size),
                                      initializer='orthogonal',
                                      name='kernel')
        if self.use_bias:
            self.s_bias = self.add_weight(shape=(self.state_size,),
                                          initializer='zeros',
                                          name='s_bias')
            self.bias = self.add_weight(shape=(self.output_size,),
                                        initializer='zeros',
                                        name='bias')
        else:
            self.s_bias = None
            self.bias = None
        self.built = True

    def call(self, inputs, states):
        prev_state = states[0]
        new_state = K.dot(inputs, self.sx_kernel) + K.dot(prev_state, self.ss_kernel)
        if self.s_bias is not None:
            new_state = K.bias_add(new_state, self.s_bias)
        if self.activation is not None:
            new_state = self.activation(new_state)
        output = K.dot(new_state, self.kernel)
        if self.bias is not None:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output, [new_state]
