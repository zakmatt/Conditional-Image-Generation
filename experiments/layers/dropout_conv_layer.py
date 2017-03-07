#!/usr/bin/env python3
from convolutional_layer import ConvolutionalLayer
from utils import dropout_from_layer

class DropoutConvLayer(ConvolutionalLayer):
    
    def __init__(self, input, filter_shape, input_shape, 
                 is_batch_norm, poolsize=(2,2)):
        super().__init__(input, filter_shape, input_shape, 
                 is_batch_norm, poolsize=(2,2))
        
    def output(self, activation, probability = 0.5, alpha = 0.2):
        output = super().output(activation, alpha)
        self.output = dropout_from_layer(output, probability)
        return self.output
        
if __name__ == '__main__':
    import numpy as np
    import theano
    import theano.tensor as T
    
    inputss = np.random.randn(30, 3, 64, 64) * 100
    inputss = theano.shared(value = np.asanyarray(inputss, dtype = theano.config.floatX))
    x = T.tensor4('x')
    input_x = x.reshape((30, 3, 64, 64))
    layer = DropoutConvLayer(input_x, (64, 3, 4, 4), (30, 3, 64, 64), False)

    #activation = T.ls
    a = theano.function(
            [],
            layer.output('lrelu'),
            givens = {
                    x: inputss
                    }
            )
    temp = a()