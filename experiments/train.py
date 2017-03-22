#!/usr/bin/env python3
from model import Model
from lib.updates import Adam, SGD, Regularizer
import logging
import numpy as np
import theano
import theano.tensor as T
theano.config.floatX = 'float32'

BATCH_SIZE = 30
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_Adam(full_images, batch_size, n_epochs=200):
    corrupted_images = T.set_subtensor(oryginal[:, :, 16:48, 16:48],  0)
    
    # compute number of minibatches for training
    n_train_batches = full_images.get_value(borrow=True).shape[0]
    n_train_batches //= BATCH_SIZE
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    
    corrupted_images_T = T.tensor4('corrupted_images')
    corrupted_input_images_T = corrupted_images_T.reshape((BATCH_SIZE, 3, 64, 64))
    
    full_images_T = T.tensor4('full_images')
    full_input_images_T = full_images_T.reshape((BATCH_SIZE, 3, 64, 64))
    
    model = Model(corrupted_input_images_T, full_input_images_T, BATCH_SIZE)
    
    discrm_cost = model.discriminator_loss()
    gen_cost = model.generator_loss(1.0, 100.0)
    generator_updater = Adam(lr=2.0, 
                             b1=0.5, 
                             clipnorm=10.,
                             regularizer=Regularizer(l2=1e-5))
    discriminator_updater = Adam(lr=2.0, 
                                 b1=0.5, 
                                 clipnorm=10., 
                                 regularizer=Regularizer(l2=1e-5))
    gen_params = model.generator.params
    gen_updates = generator_updater(gen_params, gen_cost)
    dis_params = model.discriminator_real.params
    dis_updates = discriminator_updater(dis_params, discrm_cost)
    
    train_generator_fn = theano.function(
            [index],
            gen_cost,
            updates = gen_updates,
            givens = {
                    corrupted_images_T: corrupted_images[index * batch_size:(index + 1) * batch_size],
                    full_images_T: full_images[index * batch_size:(index + 1) * batch_size]
                    }
            )
    train_discriminator_fn = theano.function(
            [index],
            discrm_cost,
            updates = dis_updates,
            givens = {
                    corrupted_images_T: corrupted_images[index * batch_size:(index + 1) * batch_size],
                    full_images_T: full_images[index * batch_size:(index + 1) * batch_size]
                    }
            )
    
    # train
    epoch = 0
    done_looping = False
    print('start learning')
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        print('epoch: %d' % epoch)
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            print('iter: %d' % iter)

            cost_gen = train_generator_fn(minibatch_index)
            cost_disc = train_discriminator_fn(minibatch_index)
                   
            if iter % 5 == 0:
                print('training @ iter = %s' % iter)
                print('generator cost = %.3f' % cost_gen)
                print('generator cost = %.3f' % cost_disc)
                

if __name__ == '__main__':
    oryginal = np.random.randn(300, 3, 64, 64) * 100
    oryginal = np.asarray(oryginal, dtype = theano.config.floatX)
    oryginal = theano.shared(value = oryginal,
                                borrow = True)
    corrupted = T.set_subtensor(oryginal[:, :, 16:48, 16:48],  0)
    
    train_Adam(oryginal, BATCH_SIZE)
    
    