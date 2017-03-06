#!/usr/bin/env python3
import numpy as np
import theano
from utils import initialize_weights, initialize_bias

class Layer(object):
    
    def __init__(self, filter_shape, input_shape, 
                 is_batch_norm, is_dropout):
        """
        Convolution layer with pooling
        
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor of shape image_shape
        
        :type filter_shape: a tuple or a list of 4. elements
        :param filter_shape: (number of filters, number of input feature maps,
                              filter height, filter width)
        
        :type image_shape: a tuple of a list of 4. elements
        :param image_shape: (batch size, number of channels or feature maps,
                             image height, image width)
        
        :type poolsize: a tuple or a list of 2 elements
        :param poolsize: pooling size - downsampling. (width, height)
        """
        
        assert input_shape[1] == filter_shape[1]
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        
        # number of input to a feature map
        n_in = np.prod(filter_shape[1:])
        W = initialize_weights(filter_shape, n_in)
        # One bias per feature map
        b = initialize_bias(filter_shape)
        
        self.W = theano.shared(W, borrow = True)
        self.b = theano.shared(b, borrow = True)
        
        # parameters of the layer
        self.params = [self.W, self.b]
        
if __name__ == '__main__':
    pass