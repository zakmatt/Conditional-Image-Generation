#!/usr/bin/env python3
from layers.generator import Generator
from layers.discriminator import Discriminator
from layers.layers_parameters import encoder_params, decoder_params, discriminator_params, get_layers_params
from lib.updates import Adam, Regularizer
import numpy as np
import theano
import theano.tensor as T

EPS = 1e-12
BATCH_SIZE = 30

bce = T.nnet.binary_crossentropy


class Model(object):
    def __init__(self, corrupted_images, full_images, batch_size):
        
        encoder_parameters = get_layers_params(batch_size, encoder_params)
        decoder_parameters = get_layers_params(batch_size, decoder_params)
        for pos, dec_params in enumerate(decoder_params):
            decoder_parameters[pos].append(dec_params[5])
        discriminator_parameters = get_layers_params(batch_size, discriminator_params)
        self.corrupted_images = corrupted_images
        self.full_images = full_images
        
        # Create Generator
        self.generator = Generator(self.corrupted_images, batch_size,
                                   encoder_parameters, decoder_parameters)
        self.generator_output = self.generator.output('tanh')
        
        # Create discriminator based on generator output
        self.fake_input = T.concatenate([self.corrupted_images, self.generator_output], axis = 1)
        self.discriminator_fake = Discriminator(self.fake_input, batch_size, discriminator_parameters)
        
        # Create discriminator based on real images
        self.real_input = T.concatenate([self.corrupted_images, self.full_images], axis = 1)
        self.discriminator_real = Discriminator(self.real_input, batch_size, discriminator_parameters)
        
        self.predict_real = self.discriminator_real.output('sigm')
        self.predict_fake = self.discriminator_fake.output('sigm')
    
    def compute_outputs(self, corrupted_images, full_images):
        # Generator output
        self.generator_output = self.generator._get_image(corrupted_images)
        self.full_images = full_images
        
        # input layer to discriminator - concatenate
        real = T.concatenate([corrupted_images, full_images], axis = 1)
        self.predict_real = bce(self.discriminator_output, real)
        #self.predict_real = self.discriminator._predict_score(corrupted_images, self.full_images)
        #self.predict_fake = self.discriminator._predict_score(corrupted_images, self.gen_output)
    
    def discriminator_loss(self):#, corrupted_images, full_images):
        #self.compute_outputs(corrupted_images, full_images)
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        return -T.mean(T.log(self.predict_real + EPS) + 
                       T.log(1 - self.predict_fake + EPS))

    def generator_loss(self, gan_weight, l1_weight):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = T.mean(-T.log(self.predict_fake + EPS))
        gen_loss_L1 = T.mean(T.abs_(self.full_images - self.generator_output))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight
        return gen_loss
    
if __name__ == '__main__':
    oryginal = np.random.randn(30, 3, 64, 64) * 100
    oryginal = np.asarray(oryginal, dtype = theano.config.floatX)
    oryginal = theano.shared(value = oryginal,
                                borrow = True)
    corrupted = T.set_subtensor(oryginal[:, :, 16:32, 16:32],  0)
    
    corrupted_images = T.tensor4('corrupted_images')
    corrupted_input_images = corrupted_images.reshape((BATCH_SIZE, 3, 64, 64))
    
    full_images = T.tensor4('full_images')
    full_input_images = full_images.reshape((BATCH_SIZE, 3, 64, 64))
    
    model = Model(corrupted_input_images, full_input_images, BATCH_SIZE)
    
    discrm_loss = model.discriminator_loss()
    gen_loss = model.generator_loss(1.0, 100.0)
    generator_updater = Adam(lr=2.0, 
                             b1=0.5, 
                             clipnorm=10.,
                             regularizer=Regularizer(l2=1e-5))
    discriminator_updater = Adam(lr=2.0, 
                                 b1=0.5, 
                                 clipnorm=10., 
                                 regularizer=Regularizer(l2=1e-5))
    gen_params = model.generator.params
    gen_updates = generator_updater(gen_params, gen_loss)
    dis_params = model.discriminator_real.params
    dis_updates = discriminator_updater(dis_params, discrm_loss)