#!/usr/bin/env python3
from convolutional_layer import ConvolutionalLayer
from generator import Generator
import theano.tensor as T

class Discriminator(object):
    def __init__(self, generated_input, oryginal_image, batch_size):
        
        self.layers = []
        input_layer = T.concatenate([generated_input, oryginal_image], axis = 1)
        
        # Construct the first convolutional layer:
        # filtering reduces the image size to (64 , 64) - image padded
        # subsampling this further to (64/2, 64/2) = (32, 32)
        # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
        layer_0 = ConvolutionalLayer(
                input = input_layer,
                filter_shape = (64, 6, 5, 5),
                input_shape = (batch_size, 6, 64, 64),
                is_batch_norm = True,
                subsample = (2, 2)
                )
        self.layers.append(layer_0)
        # create output for Convolutional layer
        # LeakyReLU alpha = 0.2
        layer_0_output = layer_0.output(activation = 'lrelu', alpha = 0.2)
        
        # maxpooling reduces the size (32 / 2, 32 / 2) = (16, 16)
        # filter shape: # of filters, filter depth, length, width
        # filter depth should be equal to the input depth shape[1]
        layer_1 = ConvolutionalLayer(
                input = layer_0_output, 
                filter_shape = (128, 64, 5, 5),
                input_shape = (batch_size, 64, 32, 32),
                is_batch_norm = True,
                subsample = (2, 2)
                )
        self.layers.append(layer_1)
        # create output for Convolutional layer
        # LeakyReLU alpha = 0.2
        layer_1_output = layer_1.output(activation = 'lrelu', alpha = 0.2)
        
        # subsampling the size (16 / 2, 16 / 2) = (8, 8)
        # filter shape: # of filters, input depth, filter shape
        layer_2 = ConvolutionalLayer(
                input = layer_1_output,
                filter_shape = (256, 128, 5, 5),
                input_shape = (batch_size, 128, 16, 16),
                is_batch_norm = True,
                subsample = (2, 2)
                )
        self.layers.append(layer_2)
        # LeakyReLU
        layer_2_output = layer_2.output(activation = 'lrelu', alpha = 0.2)
        
        layer_3 = ConvolutionalLayer(
                input = layer_2_output,
                filter_shape = (30, 256, 5, 5),
                input_shape = (batch_size, 256, 8, 8),
                is_batch_norm = False,
                subsample = (1, 1)
                )
        self.layers.append(layer_3)
        self.params = [param for layer in self.layers 
                       for param in layer.params]
        
    def output(self, activation):
        if activation == 'tanh':
            self.discriminator_output = self.layers[-1].output(activation = 'tanh')
        elif activation == 'relu':
            self.discriminator_output = self.layers[-1].output(activation = 'relu')
        elif activation == 'sigm':
            self.discriminator_output = self.layers[-1].output(activation = 'sigm')
        else:
            self.discriminator_output = self.layers[-1].output(activation = None)
            
        return self.discriminator_output
    
if __name__ == '__main__':
    import numpy as np
    import theano
    theano.config.floatX = 'float32'
    inputss = np.random.randn(30, 3, 64, 64) * 100
    inputss = theano.shared(value = np.asanyarray(inputss, dtype = theano.config.floatX))
    x = T.tensor4('x')
    input_x = x.reshape((30, 3, 64, 64))
    generator = Generator(input_x, 30)
    a = theano.function(
            [],
            generator.output('tanh'),
            givens = {
                    x: inputss
                    }
            )
    gen_output = a()
    print(gen_output.shape)
    
    discr_input = theano.shared(value = np.asanyarray(gen_output, dtype = theano.config.floatX))
    d_i = T.tensor4('d_i')
    d_i = d_i.reshape((30, 3, 64, 64))
    discriminator = Discriminator(d_i, input_x, 30)
    b = theano.function(
            [],
            discriminator.output('sigm'),
            givens = {
                    d_i: discr_input,
                    x: inputss
                    }
            )
    discriminator_output = b()
    print(discriminator_output.shape)