#!/usr/bin/env python3
import numpy as np
import theano
from layers.utils import initialize_weights, initialize_bias

# first layer with  no batch normalization
encoder_params = [[64, 5, 3, 64, False], [128, 5, 64, 32, True],
                  [256, 5, 128, 16, True]]#,[512, 5, 256, 8, True],
                  #[512, 5, 512, 4, True], [512, 5, 512, 2, True]]

# * 2 because of the U-structure
decoder_params = [#[512, 5, 512, 2, True, 0.5], [512, 5, 512*2, 4, True, 0.5],
                  #[256, 5, 256*2, 8, True, 0.5], 
                  [128, 5, 128*2, 16, True, 0.5],
                  [64, 5, 128*2, 32, True, 0.0], [3, 5, 64*2, 64, False, 0.0]]

discriminator_params = [[64, 5, 6, 64, True, (2, 2)], [128, 5, 64, 32, True, (2, 2)],
                        [256, 5, 128, 16, True, (2, 2)], [1, 5, 256, 8, True, (1, 1)]]

def get_layers_params(batch_size, params):
    layers_params = []
    for l_params in params:
        filter_depth = l_params[0]
        filter_size = l_params[1]
        input_depth = l_params[2]
        input_size = l_params[3]
        is_batch_norm = l_params[4]
        filter_shape = (filter_depth,
                        input_depth,
                        filter_size,
                        filter_size)
        input_shape = (batch_size,
                       input_depth,
                       input_size,
                       input_size)
        n_in = np.prod(filter_shape[1:])
        W = theano.shared(
                initialize_weights(filter_shape, n_in),
                name = 'W',
                borrow = True
                )
        b = theano.shared(
                initialize_bias(filter_shape),
                name = 'b',
                borrow = True
                )
        if is_batch_norm:
            gamma =  theano.shared(value = np.ones(
                    (filter_shape[0],), dtype=theano.config.floatX
                    ), name='gamma')
            beta = theano.shared(value = np.zeros(
                    (filter_shape[0],), dtype=theano.config.floatX
                    ), name='beta')
        else:
            gamma = None
            beta = None
        layers_params.append([filter_shape, input_shape, is_batch_norm, W, b, gamma, beta])
    return layers_params