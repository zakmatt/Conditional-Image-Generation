#!/usr/bin/env python3
from convolutional_layer import ConvolutionalLayer
from generator import Generator
import theano.tensor as T

class Discriminator(object):
    def __init__(self, corrupted_images, target_images, batch_size):
        
        self.layers = []
        input_layer = T.concatenate([corrupted_images, target_images], axis = 1)
        
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
    
    def _predict_score(self, contour, target):
        predict = theano.function(
                [],
                self.discriminator_output,
                givens = {
                        disc_corrupted: contour,
                        disc_input: target
                        }
                )
        return predict()

if __name__ == '__main__':
    import numpy as np
    import theano
    theano.config.floatX = 'float32'
        
    full_images = np.random.randn(30, 3, 64, 64) * 100
    full_images = theano.shared(value = np.asanyarray(full_images, dtype = theano.config.floatX))
    
    contour = T.set_subtensor(full_images[:, :, 16:32, 16:32], 0)
    
    contour_input = T.tensor4('contour_input')
    contour_input_image = contour_input.reshape((30, 3, 64, 64))
    generator = Generator(contour_input_image, 30)
    gen_output = theano.function(
            [],
            generator.output('tanh'),
            givens = {
                    contour_input: contour
                    }
            )
    generator_output = gen_output()
    
    disc_input = T.tensor4('disc_input')
    disc_target_images = disc_input.reshape((30, 3, 64, 64))
    
    disc_corrupted = T.tensor4('disc_corrupted')
    disc_corrupted_images = disc_corrupted.reshape((30, 3, 64, 64))
    
    discriminator = Discriminator(disc_corrupted_images, disc_target_images, 30)
    discriminator.output('sigm')
    pred_fake = discriminator._predict_score(contour, generator_output)
    pred_real = discriminator._predict_score(contour, full_images)
    '''
    predict_fake = theano.function(
            [],
            disc_output,
            givens = {
                    disc_corrupted: contour,
                    disc_input: generator_output
                    }
            )
    
    predict_true = theano.function(
            [],
            disc_output,
            givens = {
                    disc_corrupted: contour,
                    disc_input: full_images
                    }
            )
    
    pred_fake = predict_fake()
    pred_real = predict_true()
    '''
    print('fake shape: ', pred_fake.shape)
    print('real shape: ', pred_real.shape)
