#!/usr/bin/env python
from keras.engine import InputSpec, Layer
from keras.layers import Dense
from keras.layers.convolutional import Convolution2D
from keras import initializations, regularizers
from keras import backend as K

import numpy as np
import tensorflow as tf


class MultiBatchNorm(Layer):
    def __init__(self, num=3, epsilon=1e-3, momentum=0.99, weights=None,
                 beta_init='zero', gamma_init='one', gamma_regularizer=None,
                 beta_regularizer=None, **kwargs):
        self.supports_masking = False
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.num = num
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.uses_learning_phase = True
        super(MultiBatchNorm, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('MultiBatchNorm requires two inputs')
        
        # Check shapes
        x_shape = input_shape[0]
        w_shape = input_shape[1]
        if any(a != b for a, b in zip(x_shape[:-1], w_shape[:-1])):
            raise ValueError('Incompatible input dimensions')
        if w_shape[-1] != self.num:
            raise ValueError(
                'The number of weights per sample must match the num parameter')
        
        self.input_spec = [InputSpec(shape=x_shape), InputSpec(shape=w_shape)]
        p_shape = (x_shape[-1],)
        s_shape = (x_shape[-1], self.num)
        
        self.gamma = self.gamma_init(p_shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(p_shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]
        
        self.regularizers = []
        if self.gamma_regularizer:
            self.gamma_regularizer.set_param(self.gamma)
            self.regularizers.append(self.gamma_regularizer)

        if self.beta_regularizer:
            self.beta_regularizer.set_param(self.beta)
            self.regularizers.append(self.beta_regularizer)
        
        self.running_mean = K.zeros(
            s_shape, name='{}_running_mean'.format(self.name))
        self.running_std = K.ones(
            s_shape, name='{}_running_std'.format(self.name))
        self.non_trainable_weights = [self.running_mean, self.running_std]
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(MultiBatchNorm, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        assert self.built, 'Layer must be built before being called'
        input_shape = self.input_spec[0].shape
        
        # Useful stuff
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[-1]
        
        # Calculate mean and std
        in_sm = tf.nn.softmax(inputs[1], -1)
        W = K.expand_dims(in_sm, len(input_shape) - 1)
        X = K.expand_dims(inputs[0], -1)
        W_sum = K.sum(in_sm, reduction_axes)
        
        mean = K.sum(W * X, reduction_axes) / W_sum
        x_minus_mean_sq = (X - mean) ** 2
        std = K.sum(W * x_minus_mean_sq, reduction_axes) / W_sum
        
        # Running updates
    self.add_updates(
        [K.moving_average_update(self.running_mean, mean, self.momentum),
         K.moving_average_update(self.running_std, std, self.momentum)], inputs)
        
        xn = (X - K.in_train_phase(mean, self.running_mean)) / \
            K.sqrt(K.in_train_phase(std, self.running_std) + self.epsilon)
        xn = K.sum(W * xn, -1)
        return xn * self.gamma + self.beta
    
    def get_output_shape_for(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'num': self.num,
                  'epsilon': self.epsilon,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  'momentum': self.momentum}
        base_config = super(MultiBatchNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def multibn_block(x, num, bias=10., weight_decay=5e-4):
    channels = int(x.get_shape()[-1])
    bias_init = np.ones((num,), dtype=np.float32) * bias
    
    if len(x.get_shape()) == 4:
        # Convolutional mode
        filter_shape = (1, 1, channels, num)
        filter_init = np.random.standard_normal(filter_shape).astype(np.float32)
        filter_init = filter_init * np.sqrt(2.0 / channels)
        
        w = Convolution2D(num, 1, 1, W_regularizer=regularizers.l2(weight_decay),
                          weights=[filter_init, bias_init])(x)
    else:
        # Fully connected mode
        filter_shape = (channels, num)
        filter_init = np.random.standard_normal(filter_shape).astype(np.float32)
        filter_init = filter_init * np.sqrt(2.0 / channels)
        
        w = Dense(num, W_regularizer=regularizers.l2(weight_decay),
                  weights=[filter_init, bias_init])(x)
    
    return MultiBatchNorm(num=num)([x, w])
