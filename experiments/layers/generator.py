#!/usr/bin/env python3
from convolutional_layer import ConvolutionalLayer
from upconvolutional_layer import UpconvolutionalLayer
from dropout_upconv_layer import DropoutUpconvLayer

class Generator(object):
    
    def __init__(self, input, batch_size):
        
        self.dropout_layers = []
        self.layers = []
        
        
        #############
        ## Encoder ##
        #############
        
        input_layer = input
        
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (64 , 64) - image padded
        # maxpooling reduces this further to (64/2, 64/2) = (32, 32)
        # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
        layer_0 = ConvolutionalLayer(
                input = input_layer,
                filter_shape = (64, 3, 4, 4),
                input_shape = (batch_size, 3, 64, 64),
                is_batch_norm = True,
                poolsize = (2, 2)
                )
        # create output for Convolutional layer
        # LeakyReLU alpha = 0.2
        layer_0_output = layer_0.output(activation = 'lrelu', alpha = 0.2)
        
        # maxpooling reduces the size (32 / 2, 32 / 2) = (16, 16)
        # filter shape: # of filters, filter depth, length, width
        # filter depth should be equal to the input depth shape[1]
        layer_1 = ConvolutionalLayer(
                input = layer_0_output, 
                filter_shape = (128, 64, 4, 4),
                input_shape = (batch_size, 64, 32, 32),
                is_batch_norm = True,
                poolsize = (2, 2)
                )
        
        # create output for Convolutional layer
        # LeakyReLU alpha = 0.2
        layer_1_output = layer_1.output(activation = 'lrelu', alpha = 0.2)
        
        # maxpooling reduces the size (16 / 2, 16 / 2) = (8, 8)
        # filter shape: # of filters, input depth, filter shape
        layer_2 = ConvolutionalLayer(
                input = layer_1_output,
                filter_shape = (256, 128, 4, 4),
                input_shape = (batch_size, 128, 16, 16),
                is_batch_norm = True,
                poolsize = (2, 2)
                )
        # LeakyReLU
        layer_2_output = layer_2.output(activation = 'lrelu', alpha = 0.2)
        
        # maxpooling reduces the size (8 / 2, 8 / 2) = (4, 4)
        layer_3 = ConvolutionalLayer(
                input = layer_2_output,
                filter_shape = (512, 256, 4, 4),
                input_shape = (batch_size, 256, 8, 8),
                is_batch_norm = True,
                poolsize = (2, 2)
                )
        
        # LeakyReLU
        layer_3_output = layer_3.output(activation = 'lrelu', alpha = 0.2)

        # maxpooling reduces the size (4 / 2, 4 / 2) = (2, 2)        
        layer_4 = ConvolutionalLayer(
                input = layer_3_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 4, 4),
                is_batch_norm = True,
                poolsize = (2, 2)
                )
        
        # LeakyReLU
        layer_4_output = layer_4.output(activation = 'lrelu', alpha = 0.2)
        
        # maxpooling reduces the size (2 / 2, 2 / 2) = (1, 1)        
        layer_5 = ConvolutionalLayer(
                input = layer_3_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 2, 2),
                is_batch_norm = True,
                poolsize = (2, 2)
                )
        
        # LeakyReLU
        layer_5_output = layer_5.output(activation = 'lrelu', alpha = 0.2)
        
        #############
        ## Decoder ##
        #############
        
        layer_6 = DropoutUpconvLayer(
                input = layer_5_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 1, 1),
                is_batch_norm = True,
                scale = 2
                )
        
        # ReLU
        layer_6_activation_output = layer_6.output(activation = 'relu', 
                                                   probability = 0.5)
        # Merge with the Encoder layer
        layer_6_output = layer_6_activation_output + layer_4_output
        
        layer_7 = DropoutUpconvLayer(
                input = layer_6_output,
                filter_shape = (512, 512, 4, 4),
                input_shape = (batch_size, 512, 2, 2),
                is_batch_norm = True,
                scale = 2
                )
        # ReLU
        layer_7_activation_output = layer_7.output(activation = 'relu', 
                                                   probability = 0.5)
        # Merge with the Encoder layer
        layer_7_output = layer_7_activation_output + layer_3_output
        
        layer_8 = DropoutUpconvLayer(
                input = layer_7_output,
                filter_shape = (256, 512, 4, 4),
                input_shape = (batch_size, 512, 4, 4),
                is_batch_norm = True,
                scale = 2
                )
        
        # ReLU
        layer_8_activation_output = layer_8.output(activation = 'relu', 
                                                   probability = 0.5)
        # Merge with the Encoder layer
        layer_8_output = layer_8_activation_output + layer_2_output
        
        
        layer_9 = UpconvolutionalLayer(
                input = layer_8_output,
                filter_shape = (128, 256, 4, 4),
                input_shape = (batch_size, 256, 8, 8),
                is_batch_norm = True,
                scale = 2
                )
        # ReLU
        layer_9_activation_output = layer_9.output(activation = 'relu')
        # Merge with the Encoder layer
        layer_9_output = layer_9_activation_output + layer_1_output
        
        
        layer_10 = UpconvolutionalLayer(
                input = layer_9_output,
                filter_shape = (64, 128, 4, 4),
                input_shape = (batch_size, 128, 16, 16),
                is_batch_norm = True,
                scale = 2
                )
        # ReLU
        layer_10_activation_output = layer_10.output(activation = 'relu')
        # Merge with the Encoder layer
        layer_10_output = layer_10_activation_output + layer_0_output
        
        layer_11 = UpconvolutionalLayer(
                input = layer_10_output,
                filter_shape = (3, 64, 4, 4),
                input_shape = (batch_size, 64, 32, 32),
                is_batch_norm = True,
                scale = 2
                )

        # ReLU
        layer_11_activation_output = layer_11.output(activation = 'tanh')
        
        # add input to output and convolve - one option
        # generator_output = layer_11_activation_output + input
        self.generator_output = layer_11_activation_output