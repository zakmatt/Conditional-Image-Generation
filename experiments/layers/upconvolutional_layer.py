#!/usr/bin/env python3
from layer import Layer
from layers_parameters import get_layers_params, decoder_params
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from utils import bilinear_upsample

theano.config.floatX = 'float32'

class UpconvolutionalLayer(Layer):
    
    def __init__(self, input, filter_shape, input_shape, 
                 is_batch_norm, scale = 2, W = None,
                 b = None, gamma = None, beta = None):
        super().__init__(filter_shape, input_shape, 
                       is_batch_norm)
        
        self.scale = scale
        # upsample input scale times
        upsampled_out = bilinear_upsample(input, input_shape[0], input_shape[1], scale)
        self.input = upsampled_out
        self.gamma = gamma
        self.beta = beta
    
    def output(self, activation, alpha = 0.2):
        
        # Convolve input feature maps with filters
        # conv2d from tensor.nnet implements the convolutional layers 
        # present in convolutional neural networks
        # (where filters are 3D and pool over several input channels)

        output = conv2d(
                input = self.input,
                filters = self.W,
                input_shape = self.input_shape,
                filter_shape = self.filter_shape,
                border_mode = 'half'
                )

        #output = self.input
        if self.is_batch_norm:
            if self.gamma is None:
                self.gamma = theano.shared(value = np.ones(
                        (self.filter_shape[0],), dtype=theano.config.floatX
                        ), name='gamma')
            else:
                self.gamma = self.gamma
            
            if self.beta is None:
                self.beta = theano.shared(value = np.zeros(
                        (self.filter_shape[0],), dtype=theano.config.floatX
                        ), name='beta')
            else:
                self.beta = self.beta
            
        output += self.b.dimshuffle('x', 0, 'x', 'x')
        
        if activation == 'relu':
            self.output = T.nnet.relu(output)
        elif activation == 'lrelu':
            self.output = T.nnet.relu(output, alpha)
        elif activation == 'tanh':
            self.output = T.tanh(output)
        elif activation == None:
            self.output = output

        return self.output
    
if __name__ == '__main__':
    BATCH_SIZE = 30
    layers_params = get_layers_params(BATCH_SIZE, decoder_params)
    x = T.tensor4('x')
    input_x = x.reshape((BATCH_SIZE, 512, 1, 1))
    layers = []    
    for layer_params in layers_params:
        filter_shape = layer_params[0]
        input_shape = layer_params[1]
        is_batch_norm = layer_params[2]
        W = layer_params[3]
        b = layer_params[4]
        gamma = layer_params[5]
        beta = layer_params[6]
        layer = UpconvolutionalLayer(input_x, filter_shape, input_shape,
                                   is_batch_norm, W = W, b = b,gamma = gamma,
                                   beta=beta)
        layers.append(layer)
        input_x = layers[-1].output('relu')
    inputss = np.random.randn(BATCH_SIZE, 512, 1, 1) * 100
    inputss = theano.shared(value = np.asanyarray(inputss, dtype = theano.config.floatX))
    a = theano.function(
            [],
            input_x,
            givens = {
                    x: inputss
                    }
            )
    temp2 = a()
    print(temp2[0, 0, 0, 2])
    print(temp2.shape)
    