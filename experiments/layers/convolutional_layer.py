#!/usr/bin/env python3
from layer import Layer
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from utils import batchnorm

theano.config.floatX = 'float32'

class ConvolutionalLayer(Layer):
    
    def __init__(self, input, filter_shape, input_shape, 
                 is_batch_norm, subsample=(2,2)):
        super().__init__(filter_shape, input_shape, 
                       is_batch_norm)
        self.input = input
        self.subsample = subsample
        
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
                border_mode = 'half',
                subsample = self.subsample
                )

        if self.is_batch_norm:
            self.gamma = theano.shared(value = np.ones(
                    (self.filter_shape[0],), dtype=theano.config.floatX
                    ), name='gamma')
            self.beta = theano.shared(value = np.zeros(
                    (self.filter_shape[0],), dtype=theano.config.floatX
                    ), name='beta')
            self.params += [self.gamma, self.beta]
            output = batchnorm(output, self.gamma, self.beta)
            
        output += self.b.dimshuffle('x', 0, 'x', 'x')
        
        if activation == 'relu':
            self.output = T.nnet.relu(output)
        elif activation == 'lrelu':
            self.output = T.nnet.relu(output, alpha)
        elif activation == 'tanh':
            self.output = T.tanh(output)
        elif activation == 'sigm':
            self.output = T.nnet.sigmoid(output)
        elif activation == None:
            self.output = output
            
        return self.output

    
if __name__ == '__main__':
    inputss = np.random.randn(30, 512, 16, 16) * 100
    inputss = theano.shared(value = np.asanyarray(inputss, dtype = theano.config.floatX))
    x = T.tensor4('x')
    input_x = x.reshape((30, 512, 16, 16))
    layer = ConvolutionalLayer(input_x, (512, 512, 4, 4), (30, 512, 2, 2), True)

    #activation = T.ls
    a = theano.function(
            [],
            layer.output('lrelu'),
            givens = {
                    x: inputss
                    }
            )
    temp = a()
    print(temp.shape)