#!/usr/bin/env python3
import numpy as np
import theano
from layers.utils import initialize_weights, initialize_bias

class Layer(object):
    
    def __init__(self, filter_shape, input_shape, 
                 is_batch_norm, W = None, b = None):
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
        
        if W is None:
            # number of input to a feature map
            n_in = np.prod(filter_shape[1:])
            W = initialize_weights(filter_shape, n_in)
            self.W = theano.shared(W, name = 'W', borrow = True)
        else:
            self.W = W
            
        if b is None:
            # One bias per feature map
            b = initialize_bias(filter_shape)
            self.b = theano.shared(b, name = 'b', borrow = True)
        else:
            self.b = b
            
        # parameters of the layer
        self.params = [self.W, self.b]
        
if __name__ == '__main__':
    BATCH_SIZE = 30
    # num offilters, filter size, input depth, input size, is_batch_norm
    params = [[64, 5, 3, 64, True],[64, 5, 3, 64, True],
              [64, 5, 3, 64, True],[64, 5, 3, 64, True]]
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
        input_shape = (BATCH_SIZE,
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
        layers_params.append([filter_shape, input_shape, is_batch_norm, W, b])
    
    layers = []    
    for layer_params in layers_params:
        filter_shape = layer_params[0]
        input_shape = layer_params[1]
        is_batchnorm = layer_params[2]
        W = layer_params[3]
        b = layer_params[4]
        layer = Layer(filter_shape, input_shape, is_batch_norm, W, b)
        layers.append(layer)
        
    print(len(layers))