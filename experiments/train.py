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

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 32
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_updaters(method):
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
        
    return generator_updater, discriminator_updater

def train_model(full_images, full_validate, batch_size, method, save_dir, n_epochs = 200):
    # compute number of minibatches for training
    n_train_batches = full_images.shape[0]
    n_train_batches //= BATCH_SIZE
    n_train_micro_batches = BATCH_SIZE // MICRO_BATCH_SIZE
    
    # macro batch
    macro_batch_full_images = theano.shared(np.empty((BATCH_SIZE, ) + full_images.shape[1:], dtype = theano.config.floatX),
                                            borrow = True)
    macro_batch_corrupted_images = theano.shared(np.empty((BATCH_SIZE, ) + full_images.shape[1:], dtype = theano.config.floatX),
                                                 borrow = True)
    macro_batch_corrupted_images_val = theano.shared(np.empty((BATCH_SIZE, ) + full_validate.shape[1:], dtype = theano.config.floatX),
                                                     borrow = True)
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a microbatch
    corrupted_images_T = T.tensor4('corrupted_images')
    corrupted_input_images_T = corrupted_images_T.reshape((MICRO_BATCH_SIZE, 3, 64, 64))
    
    full_images_T = T.tensor4('full_images')
    full_input_images_T = full_images_T.reshape((MICRO_BATCH_SIZE, 3, 64, 64))
    
    model = Model(corrupted_input_images_T, full_input_images_T, MICRO_BATCH_SIZE)
    
    discrm_cost = model.discriminator_loss()
    gen_cost = model.generator_loss(1.0, 100.0)
    
    # set update methods
    generator_updater, discriminator_updater = set_updaters(method)
    
    gen_params = model.generator.params
    gen_updates = generator_updater(gen_params, gen_cost)
    dis_params = model.discriminator_real.params
    dis_updates = discriminator_updater(dis_params, discrm_cost)
    
    train_generator_fn = theano.function(
            [index],
            gen_cost,
            updates = gen_updates,
            givens = {
                    corrupted_images_T: macro_batch_corrupted_images[index * MICRO_BATCH_SIZE :(index + 1) * MICRO_BATCH_SIZE],
                    full_images_T: macro_batch_full_images[index * MICRO_BATCH_SIZE:(index + 1) * MICRO_BATCH_SIZE]
                    }
            )
    train_discriminator_fn = theano.function(
            [index],
            discrm_cost,
            updates = dis_updates,
            givens = {
                    corrupted_images_T: macro_batch_corrupted_images[index * MICRO_BATCH_SIZE:(index + 1) * MICRO_BATCH_SIZE],
                    full_images_T: macro_batch_full_images[index * MICRO_BATCH_SIZE:(index + 1) * MICRO_BATCH_SIZE]
                    }
            )

    generate_train = theano.function(
            [index],
            model.generate_image,
            givens = {
                    corrupted_images_T: macro_batch_corrupted_images[index * MICRO_BATCH_SIZE:(index + 1) * MICRO_BATCH_SIZE]
                    }
            )
    
    generate_validation = theano.function(
            [index],
            model.generate_image,
            givens = {
                    corrupted_images_T: macro_batch_corrupted_images_val[index * MICRO_BATCH_SIZE:(index + 1) * MICRO_BATCH_SIZE]
                    }
            )
    
    # train
    epoch = 0
    logging.info('start training')
    while (epoch < n_epochs):
        epoch += 1
        logging.info('epoch: %d' % epoch)
        
        for batch_index in range(n_train_batches):
            
            # Create a shared variable of a batch
            macro_batch_full_images.set_value(
                    full_images[batch_index * batch_size:(batch_index + 1) * batch_size],
                    borrow = True
                    )
            macro_batch_corrupted_images.set_value(
                    T.set_subtensor(macro_batch_full_images[:, :, 1:4, 1:4],  0).eval(),
                    borrow = True
                    )
            
            # Iterate throught a batch by taking micro batches
            total_gen_cost = 0.
            total_disc_cost = 0.
            for micro_batch_index in range(n_train_micro_batches):
                total_gen_cost += train_generator_fn(micro_batch_index)
                total_disc_cost += train_discriminator_fn(micro_batch_index)
                
            total_gen_cost /= n_train_micro_batches
            total_disc_cost /= n_train_micro_batches
            
            iter = (epoch - 1) * n_train_batches + batch_index
            if iter % 100 == 0:
                logging.info('training @ iter = %s' % iter)
                logging.info('generator cost = %.3f' % total_gen_cost)
                logging.info('discriminator cost = %.3f' % total_disc_cost)
        
        # put images through the network, reshape and save 
        if epoch % 20 == 0:
            current_images = macro_batch_full_images.get_value(borrow = True)[micro_batch_index * MICRO_BATCH_SIZE:(micro_batch_index + 1) * MICRO_BATCH_SIZE]
            pred = generate_train(micro_batch_index)
            file_name = 'train_generated_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            
            pred = fill_missing_part(current_images, pred)
            file_name = 'train_generated_filled_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            
            current_val_images = full_validate[0:batch_size]
            # current_val_images = current_val_images[0:MICRO_BATCH_SIZE]
            current_val_images_corrupted = current_val_images
            current_val_images_corrupted[:, :, 16:48, 16:48] = 0
            current_val_images = current_val_images[0:MICRO_BATCH_SIZE]
            macro_batch_corrupted_images_val.set_value(current_val_images_corrupted, borrow = True)
            
            pred = generate_validation(0)
            file_name = 'validate_generated_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
            pred = fill_missing_part(current_val_images, pred)
            file_name = 'validate_generated_filled_%s_%d.npy' % (method, epoch)
            np.save(os.path.join(save_dir, file_name), pred)
    
    # saving the model
    logging.info('Saving...')
    params = model.generator.model_params
    params_save_path = os.path.join(save_dir, 'best_model.p')
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
    
    