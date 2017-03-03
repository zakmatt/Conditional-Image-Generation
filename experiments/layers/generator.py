#!/usr/bin/env python3
from convolutional_layer import ConvolutionalLayer
from deconvolutional_layer import DeconvolutionalLayer
from math import sqrt
import numpy as np
import theano.tensor as T

class Generator(object):
    
    def __init__(input, batch_size):
        x = T.matrix('x')
        y = T.matrix('y')
        
        #############
        ## Encoder ##
        #############
        
        input_layer = x.reshape((batch_size, 3, 64, 64))
        
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (64 , 64) - image padded
        # maxpooling reduces this further to (64/2, 64/2) = (32, 32)
        # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
        layer_0 = ConvolutionalLayer(
                input = input_layer,
                filter_shape = (64, 3, 4, 4),
                input_shape = (batch_size, 3, 64, 64),
                poolsize = (2, 2)
                )
        # create output for Convolutional layer
        # LeakyReLU alpha = 0.2
        layer_0_output = None
        
        # maxpooling reduces the size (32 / 2, 32 / 2) = (16, 16)
        # filter shape: # of filters, filter depth, length, width
        # filter depth should be equal to the input depth shape[1]
        layer_1 = ConvolutionalLayer(
                input = layer_0_output, 
                filter_shape = (128, 64, 4, 4),
                input_shape = (batch_size, 64, 32, 32),
                poolsize = (2, 2)
                )
        
        # create output for Convolutional layer
        # Batch normalisation
        # LeakyReLU alpha = 0.2
        layer_1_batch_output = None
        layer_1_output = None
        
        # maxpooling reduces the size (16 / 2, 16 / 2) = (8, 8)
        # filter shape: # of filters, input depth, filter shape
        layer_2 = ConvolutionalLayer(
                input = layer_1_output,
                filter_shape = (256, 128, 4, 4),
                input_shape = (batch_size, 128, 16, 16),
                poolsize = (2, 2)
                )
        
        # Batch normalisation
        layer_2_batch_output = None
        # LeakyReLU
        layer_2_output = None
        
        # maxpooling reduces the size (8 / 2, 8 / 2) = (4, 4)
        layer_3 = ConvolutionalLayer(
                input = layer_2_output,
                filter_shape = (512, 256, 4, 4),
                input_shape = (batch_size, 256, 8, 8),
                poolsize = (2, 2)
                )
        
        # Batch normalisation
        layer_3_batch_norm = None
        # LeakyReLU
        layer_3_output = None

        # maxpooling reduces the size (4 / 2, 4 / 2) = (2, 2)        
        layer_4 = ConvolutionalLayer(
                input = layer_3_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 4, 4),
                poolsize = (2, 2)
                )
        
        # Batch normalisation
        layer_4_batch_norm = None
        # LeakyReLU
        layer_4_output = None
        
        # maxpooling reduces the size (2 / 2, 2 / 2) = (1, 1)        
        layer_5 = ConvolutionalLayer(
                input = layer_3_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 2, 2),
                poolsize = (2, 2)
                )
        
        # Batch normalisation
        layer_5_batch_norm = None
        # LeakyReLU
        layer_5_output = None
        
        #############
        ## Decoder ##
        #############
        
        layer_6 = DeconvolutionalLayer(
                input = layer_5_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 1, 1),
                scale = 2
                )
        
        # Batch normalisation
        layer_6_batch_norm = None
        # Dropout
        layer_6_dropout = None
        # ReLU
        layer_6_activation_output = None
        # Merge with the Encoder layer
        layer_6_output = layer_6_activation_output + layer_4_output
        
        layer_7 = DeconvolutionalLayer(
                input = layer_6_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 2, 2),
                scale = 2
                )
        
        # Batch normalisation
        layer_7_batch_norm = None
        # Dropout
        layer_7_dropout = None
        # ReLU
        layer_7_activation_output = None
        # Merge with the Encoder layer
        layer_7_output = layer_7_activation_output + layer_3_output
        
        layer_8 = DeconvolutionalLayer(
                input = layer_7_output,
                filter_shape = (256, 512, 4, 4),
                input_shape = (batch_size, 512, 4, 4),
                scale = 2
                )
        
        # Batch normalisation
        layer_8_batch_norm = None
        # Dropout
        layer_8_dropout = None
        # ReLU
        layer_8_activation_output = None
        # Merge with the Encoder layer
        layer_8_output = layer_8_activation_output + layer_2_output
        
        
        layer_9 = DeconvolutionalLayer(
                input = layer_8_output,
                filter_shape = (128, 256, 4, 4),
                input_shape = (batch_size, 256, 8, 8),
                scale = 2
                )
        
        # Batch normalisation
        layer_9_batch_norm = None
        # Dropout ??????
        layer_9_dropout = None
        # ReLU
        layer_9_activation_output = None
        # Merge with the Encoder layer
        layer_9_output = layer_9_activation_output + layer_1_output
        
        
        layer_10 = DeconvolutionalLayer(
                input = layer_9_output,
                filter_shape = (64, 128, 4, 4),
                input_shape = (batch_size, 128, 16, 16),
                scale = 2
                )
        
        # Batch normalisation
        layer_10_batch_norm = None
        # Dropout ????
        layer_10_dropout = None
        # ReLU
        layer_10_activation_output = None
        # Merge with the Encoder layer
        layer_10_output = layer_10_activation_output + layer_0_output
        
        layer_11 = DeconvolutionalLayer(
                input = layer_10_output,
                filter_shape = (3, 64, 4, 4),
                input_shape = (batch_size, 64, 32, 32),
                scale = 2
                )
        
        # Batch normalisation
        layer_11_batch_norm = None
        # ReLU
        layer_11_activation_output = None
        
        # add input to output and convolve - one option
        # generator_output = layer_11_activation_output + input
        generator_output = layer_11_activation_output