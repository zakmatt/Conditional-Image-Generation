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
        self.discriminator_output = self.layers[-1].output(activation = 'sigm')
        self.params = [param for layer in self.layers 
                       for param in layer.params]
    '''   
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
    '''
    def output(self):
        return self.discriminator_output
    
    def create_predict(self):
        contour = T.tensor4('contour')
        target = T.tensor4('target')
        predict = theano.function(
                [contour, target],
                self.output('sigm'),
                givens = {
                        disc_corrupted: contour,
                        disc_input: target
                        }
                )
        self.predict = predict

'''
def discriminator(batch_size, discriminator_targets, discriminator_inputs = None):
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
    #discriminator_inputs = T.tensor4('discriminator_inputs')
    #discriminator_inputs = discriminator_inputs.reshape((batch_size, 3, 64, 64))
    #input_layer = T.concatenate([discriminator_inputs, discriminator_targets], axis = 1)
    #discriminator_inputs = lasagne.layers.InputLayer(shape=(batch_size, 3, 64, 64),
    #                                                 input_var=discriminator_inputs)
    #discriminator_targets = lasagne.layers.InputLayer(shape=(batch_size, 3, 64, 64),
    #                                                 input_var=discriminator_targets)
    network = lasagne.layers.ConcatLayer([discriminator_inputs, discriminator_targets], axis = 1)
    network = lasagne.layers.InputLayer(shape=(batch_size, 6, 64, 64),
                                        input_var = network)
    # Construct the first convolutional layer:
    # filtering reduces the image size to (64 , 64) - image padded
    # subsampling this further to (64/2, 64/2) = (32, 32)
    # 4D output tensor is thus of shape (batch_size, 64, 32, 32)
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                         stride=2, nonlinearity=lrelu)
    network = lasagne.layers.BatchNormLayer(network)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(5, 5),
                                         stride=2, nonlinearity=lrelu)
    network = lasagne.layers.BatchNormLayer(network)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(5, 5),
                                         stride=2, nonlinearity=lrelu)
    network = lasagne.layers.BatchNormLayer(network)
    
    network = lasagne.layers.Conv2DLayer(network, num_filters=30, filter_size=(5, 5),
                                         stride=1, nonlinearity=lasagne.nonlinearities.sigmoid)
    
    #self.network = network
    #self.get_output_for = network
    return network
        
def _get_disc_score(disc, discriminator_inputs):
    return lasagne.layers.get_output(disc.layers, discriminator_inputs)
    
def get_loss(disc_real, disc_sample):

    #WGAN loss
    discriminator_loss = -(T.log(disc_real) + T.log(1.-disc_sample)).mean()
    generator_loss = -T.log(disc_sample).mean()

    return discriminator_loss, generator_loss
'''     
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
    
    #discriminator = Discriminator(disc_input_images, generator_output, 30)
    discriminator = Discriminator(disc_corrupted_images, disc_target_images, 30)
    #discriminator.create_predict()
    
    #cont = T.tensor4('cont')
    #target = T.tensor4('target')
    predict_fake = theano.function(
            [],
            discriminator.output(),
            givens = {
                    disc_corrupted: contour,
                    disc_input: generator_output
                    }
            )
    
    predict_true = theano.function(
            [],
            discriminator.output(),
            givens = {
                    disc_corrupted: contour,
                    disc_input: full_images
                    }
            )
    
    pred_fake = predict_fake()
    pred_real = predict_true()
    #predict_fake = discriminator.predict(contour, generator_output)
    #predict_real = discriminator.predict(contour, full_images)
    print('fake shape: ', pred_fake.shape)
    print('real shape: ', pred_real.shape)
    '''

    inputss = np.random.randn(30, 3, 64, 64) * 100
    inputss = np.asarray(inputss, dtype = theano.config.floatX)
    inputss = theano.shared(value = inputss,
                                borrow = True)
    contour = T.set_subtensor(inputss[:, :, 16:32, 16:32], 0)
    contour_input = T.tensor4('contour_input')
    contour_input_image = contour_input.reshape((30, 3, 64, 64))
    generator = Generator(contour_input_image, 30)
    gen_output = generator.output('tanh')
    
    input_image = T.tensor4('input_image')
    discriminator_input = input_image.reshape((30, 3, 64, 64))

    oryginal_image = T.tensor4('oryginal_image')
    discriminator_target = input_image.reshape((30, 3, 64, 64))
    
    dis = Discriminator(contour_input_image, discriminator_target, 30)
    
    predict_fake = theano.function(
                [],
                dis.output('sigm'),
                givens = {
                        contour_input: contour,
                        discriminator_target: gen_output
                }
            )
    
    
    #discr_input = gen_output

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

    # x is contour
    # d_i is contour
    # oryginal image is inputss
    # fake image is gen_output
    d_i = T.tensor4('d_i')
    d_i = d_i.reshape((30, 3, 64, 64))
    disc = discriminator(30, d_i)
    #fake = discriminator._get_disc_score(gen_output)
    #oryginal_image = T.tensor4('oryginal_image')
    #oryginal_image = oryginal_image.reshape((30, 3, 64, 64))
    origin_real_img = T.tensor4('real_img')
    real_img = origin_real_img.dimshuffle((0, 3, 1, 2))
    #oryginal_image = oryginal_image.dimshuffle((0, 3, 1, 2))
    #fake_image = T.tensor4('fake_image')
    #fake_image = fake_image.reshape((30, 3, 64, 64))
    predict_real = _get_disc_score(disc, real_img)
    predict_fake = _get_disc_score(disc, gen_output)
    
    # Our loss
    discriminator_loss, generator_loss = get_loss(predict_real, predict_fake)

    #Discriminator update
    discriminator_params = lasagne.layers.get_all_params(disc, trainable=True)
    discriminator_updates = lasagne.updates.adam(discriminator_loss, discriminator_params, learning_rate=0.1, beta1=0.5)

    train_discriminator_fn = theano.function([input_x + origin_real_img], [discriminator_loss],
                                             updates=discriminator_updates,
                                             allow_input_downcast=True)

    xyz = train_discriminator_fn(contour, inputss)
    '''