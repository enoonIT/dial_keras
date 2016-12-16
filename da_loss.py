#!/usr/bin/env python
from keras import backend as K
import tensorflow as tf

class DaLoss:
    def __init__(self, slice_point, weight, name='daloss'):
        self.slice_point = slice_point
        self.weight = weight
        self.__name__ = name
    
    def __call__(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=1)
        
        y_pred_src = tf.slice(y_pred, [0, 0], [self.slice_point, -1])
        y_true_src = tf.slice(y_true, [0], [self.slice_point])
        y_pred_tgt = tf.slice(y_pred, [self.slice_point, 0], [-1, -1])
        
        L_source = K.sparse_categorical_crossentropy(y_pred_src, y_true_src)
        L_target = K.sum(-y_pred_tgt * K.log(y_pred_tgt), axis=1)
        
        return K.concatenate([L_source, self.weight * L_target], axis=0)
