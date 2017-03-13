#!/usr/bin/env python3
from math import sqrt
import numpy as np
import theano.tensor as T
import theano

theano.config.floatX = 'float32'

def dropout_from_layer(layer, probability = 0.5):
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

def bilinear_upsample(X, batch_size, channels, factor):
   return theano.tensor.nnet.abstract_conv.bilinear_upsampling(X, factor, batch_size=batch_size, num_input_channels=channels)

# From https://github.com/Newmu/dcgan_code/blob/master/lib/ops.py

def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            _u = u.dimshuffle('x', 0, 'x', 'x')
            _s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            _u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            _s = T.mean(T.sqr(X - _u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            _u = (1. - a) * 0. + a * _u
            _s = (1. - a) * 1. + a * _s
        X = (X - _u) / T.sqrt(_s + e)
        if g is not None and b is not None:
            X = X * g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a) * 0. + a * u
            s = (1. - a) * 1. + a * s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X * g + b
    else:
        raise NotImplementedError
    return X