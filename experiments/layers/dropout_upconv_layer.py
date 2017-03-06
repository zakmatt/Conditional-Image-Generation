#!/usr/bin/env python3
from upconvolutional_layer import UpconvolutionalLayer
from utils import dropout_from_layer

class DropoutUpconvLayer(UpconvolutionalLayer):
    
    def __init__(self, input, filter_shape, input_shape, 
                 is_batch_norm, scale = 2):
        super().__init__(input, filter_shape, input_shape, is_batch_norm, scale)
        
    def output(self, activation, probability = 0.5, alpha = 0.2):
        output = super().output(activation, alpha)
        self.output = dropout_from_layer(output, probability)
        return self.output
    
if __name__ == '__main__':
    import numpy as np
    import theano
    import theano.tensor as T
    
    inputss = np.random.randn(30, 512, 1, 1) * 100
    inputss = theano.shared(value = np.asanyarray(inputss, dtype = theano.config.floatX))
    x = T.tensor4('x')
    input_x = x.reshape((30, 512, 1, 1))
    layer = DropoutUpconvLayer(input=input_x, filter_shape= (512, 512, 4, 4),
                                 input_shape=(30, 512, 2, 2), 
                                 is_batch_norm = False,
                                 scale = 2)
    #activation = T.ls
    a = theano.function(
            [],
            layer.output('lrelu'),
            givens = {
                    x: inputss
                    }
            )
    temp = a()
        