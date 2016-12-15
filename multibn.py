#!/usr/bin/env python
from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K

import numpy as np

class MultiBatchNorm(Layer):
    def __init__(self, num=3, epsilon=1e-3, axis=-1, momentum=0.99,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):
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
            raise ValueError('Incompatible inputs')
        if w_shape[-1] != self.num:
            raise ValueError('The number of weights must match the num parameter')
        
        self.input_spec = [InputSpec(shape=x_shape), InputSpec(shape=w_shape)]
        shape = (x_shape[-1],)
        
        self.gamma = self.gamma_init(shape, 
                                     name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, 
                                   name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]
        
        self.regularizers = []
        if self.gamma_regularizer:
            self.gamma_regularizer.set_param(self.gamma)
            self.regularizers.append(self.gamma_regularizer)

        if self.beta_regularizer:
            self.beta_regularizer.set_param(self.beta)
            self.regularizers.append(self.beta_regularizer)
        
        self.running_means = []
        self.running_stds = []
        for i in xrange(self.num):
            self.running_means.append(
                K.zeros(shape,
                        name='{}_running_mean_{}'.format(self.name, i)))
            self.non_trainable_weights.append(self.running_means[-1])
            self.running_stds.append(
                K.ones(shape,
                       name='{}_running_std_{}'.format(self.name, i)))
            self.non_trainable_weights.append(self.running_stds[-1])
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    
    def call(self, inputs, mask=None):
        assert self.built, 'Layer must be built before being called'
        input_shape = self.input_spec[0].shape
        
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[-1]
        
        # Define means and standard deviations
        means = []
        stds = []
        for i in xrange(self.num):
            w = inputs[1][...,i]
            w_expanded = K.expand_dims(w, dim=-1)
            sw = K.sum(w, axis=reduction_axes)
            
            means.append(K.sum(w_expanded * inputs[0], axis=reduction_axes) / sw)
            
            x_minus_mean_sq = (inputs[0] - means[-1]) ** 2
            stds.append(K.sum(w_expanded * x_minus_mean_sq, axis=reduction_axes) / sw)
        
        # Apply batch norms
        x_normed = []
        for i in xrange(self.num):
            xn = K.batch_normalization(
                inputs[0], means[i], stds[i], self.beta, self.gamma,
                epsilon=self.epsilon)
            x_normed.append(xn)
        
        # Recombine outputs
        x_out = None
        for i in xrange(self.num):
            w = inputs[1][...,i]
            w_expanded = K.expand_dims(w, dim=-1)
            
            if x_out == None:
                x_out = w_expanded * x_normed[i]
            else:
                x_out = x_out + w_expanded * x_normed[i]
        
        # Running updates
        for i in xrange(self.num):
            self.add_updates(
                [K.moving_average_update(self.running_means[i], means[i], self.momentum),
                 K.moving_average_update(self.running_stds[i], stds[i], self.momentum)],
                inputs)
        
        # Apply running batch norms
        x_normed_running = []
        for i in xrange(self.num):
            xn = K.batch_normalization(
                inputs[0], self.running_means[i], self.running_stds[i],
                self.beta, self.gamma,
                epsilon=self.epsilon)
            x_normed_running.append(xn)
        
        # Recombine running outputs
        x_out_running = None
        for i in xrange(self.num):
            w = inputs[1][...,i]
            w_expanded = K.expand_dims(w, dim=-1)
            
            if x_out_running == None:
                x_out_running = w_expanded * x_normed_running[i]
            else:
                x_out_running = x_out_running + w_expanded * x_normed_running[i]
        
        return K.in_train_phase(x_out, x_out_running)
    
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
