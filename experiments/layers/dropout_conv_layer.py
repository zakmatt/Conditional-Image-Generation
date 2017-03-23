#!/usr/bin/env python3
from convolutional_layer import ConvolutionalLayer
from layers_parameters import encoder_params, get_layers_params
from utils import dropout_from_layer

class DropoutConvLayer(ConvolutionalLayer):
    
    def __init__(self, input, filter_shape, input_shape, 
                 is_batch_norm, rng, subsample=(2,2), W = None,
                 b = None, gamma = None, beta = None):
        super().__init__(input, filter_shape, input_shape, 
                 is_batch_norm, subsample, W, b, gamma, beta)
        self.rng = rng
        
    def output(self, activation, probability = 0.5, alpha = 0.2):
        output = super().output(activation, alpha)
        self.output = dropout_from_layer(output, self.rng, probability)
        return self.output
        
if __name__ == '__main__':
    import numpy as np
    import theano
    import theano.tensor as T
    BATCH_SIZE = 30
    layers_params = get_layers_params(BATCH_SIZE, encoder_params)
    x = T.tensor4('x')
    input_x = x.reshape((BATCH_SIZE, 3, 64, 64))
    layers = []    
    for layer_params in layers_params:
        filter_shape = layer_params[0]
        input_shape = layer_params[1]
        is_batch_norm = layer_params[2]
        W = layer_params[3]
        b = layer_params[4]
        gamma = layer_params[5]
        beta = layer_params[6]
        layer = DropoutConvLayer(input_x, filter_shape, input_shape,
                                 is_batch_norm, W = W, b = b,gamma = gamma,
                                 beta=beta)
        layers.append(layer)
        input_x = layers[-1].output('relu')
    
    inputss = np.random.randn(BATCH_SIZE, 3, 64, 64) * 100
    inputss = theano.shared(value = np.asanyarray(inputss, dtype = theano.config.floatX))
    a = theano.function(
            [],
            input_x,
            givens = {
                    x: inputss
                    }
            )
    temp = a()