#!/usr/bin/env python3
from math import sqrt
import numpy as np
import theano
from theano.tensor.nnet import conv2d

class DeconvolutionalLayer(object):
    
    def __init__(self, input, filter_shape, input_shape, poolsize=(1,1), scale = 2):
        """
        Deconvolution layer
        
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
        
        self.input = input
        
        assert input_shape[1] == filter_shape[1]
        
        # number of input to a feature map
        n_in = np.prod(filter_shape[1:])
        W = np.asanyarray(
                np.random.rand(size = filter_shape) * sqrt(2.0/n_in),
                dtype = theano.config.floatX
                )
        # One bias per feature map
        b = np.zeros((filter_shape[0], ), dtype = theano.config.floatX)
        
        self.W = theano.shared(W, borrow = True)
        self.b = theano.shared(b, borrow = True)
        
        # upsample input scale times
        upsampled_out = input.repeat(scale, axis = 2).repeat(scale, axis = 3)
        
        # convolve upsampled feature maps with filters
        self.convolution_output = conv2d(
                input = upsampled_out,
                filters = self.W,
                filter_shape = filter_shape,
                border_mode = filter_shape[2] - 1
                )
        
        self.output = self.convolution_output + self.b.dimshuffle('x', 0, 'x', 'x')
        
        self.params = [self.W, self.b]
        