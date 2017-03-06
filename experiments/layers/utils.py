#!/usr/bin/env python3
from math import sqrt
import numpy as np
import theano.tensor as T
import theano

theano.config.floatX = 'float32'

def dropout_from_layer(layer, probability):
    '''
    probability is a probability of droping a unit in a layer
    '''
    rng = np.random.RandomState(12345)
    theano_rng = T.shared_randomstreams.RandomStreams(
            rng.randint(999999)
            )
    # 1 - p since 1 is to keep a neuron and p is to drop it
    mask = theano_rng.binomial(n = 1, p = 1 - probability, size = layer.shape)
    
    output = layer * T.cast(mask, dtype = theano.config.floatX)
    
    return output

def initialize_weights(filter_shape, n_in):
    n_filters, input_depth, width, height = filter_shape
    W = np.asanyarray(
            np.random.rand(n_filters, input_depth, width, height) * sqrt(2.0/n_in),
            dtype = theano.config.floatX
            )
    return W

def initialize_bias(filter_shape):
    b = np.zeros((filter_shape[0], ), dtype = theano.config.floatX)
    return b