#!/usr/bin/env python3
from layers.convolutional_layer import ConvolutionalLayer
from layers.generator import Generator
from layers.layers_parameters import encoder_params, decoder_params, discriminator_params, get_layers_params

EPS = 1e-12

class Discriminator(object):
    def __init__(self, input_layer, batch_size, discriminator_parameters):
        
        self.layers = []
        
        for disc_params in discriminator_parameters[:-1]:
            filter_shape = disc_params[0]
            input_shape = disc_params[1]
            is_batch_norm = disc_params[2]
            W = disc_params[3]
            b = disc_params[4]
            gamma = disc_params[5]
            beta = disc_params[6]
            layer = ConvolutionalLayer(input_layer, filter_shape, input_shape,
                                       is_batch_norm, W = W, b = b,gamma = gamma,
                                       beta=beta)
            self.layers.append(layer)
            input_layer = self.layers[-1].output('lrelu')
            
        disc_params = discriminator_parameters[-1]
        filter_shape = disc_params[0]
        input_shape = disc_params[1]
        is_batch_norm = disc_params[2]
        W = disc_params[3]
        b = disc_params[4]
        gamma = disc_params[5]
        beta = disc_params[6]
        layer = ConvolutionalLayer(input_layer, filter_shape, input_shape,
                                   is_batch_norm, W = W, b = b,gamma = gamma,
                                   beta=beta)
        self.layers.append(layer)

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
    import theano.tensor as T
    theano.config.floatX = 'float32'
    BATCH_SIZE = 30
    encoder_parameters = get_layers_params(BATCH_SIZE, encoder_params)
    decoder_parameters = get_layers_params(BATCH_SIZE, decoder_params)
    for pos, dec_params in enumerate(decoder_params):
        decoder_parameters[pos].append(dec_params[5])
    discriminator_parameters = get_layers_params(BATCH_SIZE, discriminator_params)
    
    corrupted_images = T.tensor4('corrupted_images')
    corrupted_input_images = corrupted_images.reshape((BATCH_SIZE, 3, 64, 64))
    
    oryginal_images = T.tensor4('oryginal_images')
    oryginal_input_images = oryginal_images.reshape((BATCH_SIZE, 3, 64, 64))
    
    generator = Generator(corrupted_input_images, BATCH_SIZE, 
                          encoder_parameters, decoder_parameters)
    fake_image = generator.output('tanh')
    fake_input = T.concatenate([corrupted_input_images, fake_image], axis = 1)
    predict_fake = Discriminator(fake_input, BATCH_SIZE, discriminator_parameters)
    real_image = T.concatenate([corrupted_input_images, oryginal_input_images], axis = 1)
    predict_real = Discriminator(real_image, BATCH_SIZE, discriminator_parameters)
    
    oryginal = np.random.randn(30, 3, 64, 64) * 100
    oryginal = np.asarray(oryginal, dtype = theano.config.floatX)
    oryginal = theano.shared(value = oryginal,
                                borrow = True)
    corrupted = T.set_subtensor(oryginal[:, :, 16:32, 16:32],  0)
    
    pred_real = theano.function(
            [],
            predict_real.output('sigm'),
            givens = {
                    oryginal_images : oryginal,
                    corrupted_images: corrupted
                    }
            )
    pred_fake = theano.function(
            [],
            predict_fake.output('sigm'),
            givens = {
                    corrupted_images: corrupted
                    }
            )
    
    fake = pred_fake()
    real = pred_real()
    print(real.shape)
    print(fake.shape)