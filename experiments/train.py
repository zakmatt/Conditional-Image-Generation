#!/usr/bin/env python3
import argparse
from controllers.controllers import fill_missing_part, load_data
from model import Model
from lib.updates import Adam, SGD, RMSprop, Regularizer
import logging
import numpy as np
import os
import pickle
import theano
import theano.tensor as T
theano.config.floatX = 'float32'

BATCH_SIZE = 64
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(full_images, full_validate, batch_size, method, save_dir, n_epochs=200):
    
    corrupted_images = T.set_subtensor(full_images[:, :, 16:48, 16:48],  0)
    corrupted_validate = T.set_subtensor(full_validate[:, :, 16:48, 16:48],  0)
    oryginal_images  = full_images.get_value(borrow=True)
    
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
    
    if method == 'adam':
        generator_updater = Adam(lr=2.0, 
                                 b1=0.5, 
                                 clipnorm=10.,
                                 regularizer=Regularizer(l2=1e-5))
        discriminator_updater = Adam(lr=2.0, 
                                     b1=0.5, 
                                     clipnorm=10., 
                                     regularizer=Regularizer(l2=1e-5))
    elif method == 'rmsprop':
        generator_updater = RMSprop(lr=2.0,
                                    clipnorm=10.,
                                    regularizer=Regularizer(l2=1e-5))
        discriminator_updater = RMSprop(lr=2.0, 
                                        rho=0.5, 
                                        clipnorm=10., 
                                        regularizer=Regularizer(l2=1e-5))
    else:
        generator_updater = SGD(lr=2.0, 
                                b1=0.5,
                                clipnorm=10.,
                                regularizer=Regularizer(l2=1e-5))
        discriminator_updater = SGD(lr=2.0, 
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

    generate_train = theano.function(
            [index],
            model.generate_image,
            givens = {
                    corrupted_images_T: corrupted_images[index * batch_size:(index + 1) * batch_size]
                    }
            )
    
    generate_validation = theano.function(
            [index],
            model.generate_image,
            givens = {
                    corrupted_images_T: corrupted_validate[index * batch_size:(index + 1) * batch_size]
                    }
            )    
    # train
    epoch = 0
    logging.info('start training')
    while (epoch < n_epochs):
        epoch += 1
        logging.info('epoch: %d' % epoch)

        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            logging.info('iter: %d' % iter)

            cost_gen = train_generator_fn(minibatch_index)
            cost_disc = train_discriminator_fn(minibatch_index)
                   
            if iter % 100 == 0:
                logging.info('training @ iter = %s' % iter)
                logging.info('generator cost = %.3f' % cost_gen)
                logging.info('discriminator cost = %.3f' % cost_disc)

        # put images through the network, reshape and save 
        if epoch % 20 == 0:
            current_images = oryginal_images[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
            pred = generate_train(minibatch_index)
            file_name = 'train_generated_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            
            pred = fill_missing_part(current_images, pred)
            file_name = 'train_generated_filled_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            
            pred = generate_validation(minibatch_index)
            file_name = 'validate_generated_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            pred = fill_missing_part(current_images, pred)
            file_name = 'validate_generated_filled_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            
    # saving the model
    logging.info('Saving...')
    params = model.generator.model_params
    params_save_path = os.path.join(save_dir, 'best_model.npz')
    with open(params_save_path, 'wb') as file:
        pickle.dump(params, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--data_directory",
                        help = 'Directory to the data',
                        required = True)
    parser.add_argument("-s",
                        "--save_directory",
                        help = 'Saving data directory',
                        required = True)
    args = parser.parse_args()
    data_dir = args.data_directory
    save_dir = args.save_directory
    training_dataset = os.path.join(data_dir, 'images.train.npz')
    validation_dataset = os.path.join(data_dir, 'images.validate.npz')
    train = load_data(training_dataset)
    validate = load_data(validation_dataset)

    train_model(train, validate, BATCH_SIZE, 'adam', save_dir)
    
    