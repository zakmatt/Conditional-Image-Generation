#!/usr/bin/env python3
from math import sqrt
import numpy as np
import theano
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

theano.config.floatX = 'float32'

class ConvolutionalLayer(object):
    
    def __init__(self, input, filter_shape, input_shape, poolsize=(2,2)):
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
        self.input = input
        
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
        
        # Convolve input feature maps with filters
        # conv2d from tensor.nnet implements the convolutional layers 
        # present in convolutional neural networks
        # (where filters are 3D and pool over several input channels)
        self.convolution_output = conv2d(
                input = input,
                filters = self.W,
                input_shape = input_shape,
                filter_shape = filter_shape,
                border_mode = filter_shape[2] - 1
                )
        
        if poolsize is not None:
            # pool each feature map individually using maxpooling
            self.output = pool.pool_2d(
                    input = self.convolution_output,
                    ds = poolsize,
                    ignore_border = True
                    )
            
        # calculate the output of the layer
        #self.output = T.nnet.relu(pooled_out + 
        #                          self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # parameters of the layer
        self.params = [self.W, self.b]

    #def output(activation)
        
if __name__ == '__main__':
    cl = ConvolutionalLayer('ala', 'blabl')
    print(cl.input, cl.temp)