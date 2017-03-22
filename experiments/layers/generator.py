from layers.convolutional_layer import ConvolutionalLayer
from layers.upconvolutional_layer import UpconvolutionalLayer
from layers.dropout_upconv_layer import DropoutUpconvLayer
from layers.layers_parameters import encoder_params, decoder_params, get_layers_params
import theano.tensor as T

EPS = 1e-12

class Generator(object):
    
    def __init__(self, input_values, batch_size, encoder_parameters, decoder_parameters):
        
        self.dropout_layers = []
        self.input_values = input_values
        input_without_dropout = input_values
        self.layers = []
        
        #############
        ## Encoder ##
        #############
        
        for enc_params in encoder_parameters:
            filter_shape = enc_params[0]
            input_shape = enc_params[1]
            is_batch_norm = enc_params[2]
            W = enc_params[3]
            b = enc_params[4]
            gamma = enc_params[5]
            beta = enc_params[6]
            dropout_layer = ConvolutionalLayer(input_values, filter_shape, input_shape,
                                               is_batch_norm, W = W, b = b,gamma = gamma,
                                               beta=beta)
            layer = ConvolutionalLayer(input_without_dropout, filter_shape, input_shape,
                                       is_batch_norm, W = W, b = b,gamma = gamma,
                                       beta=beta)
            
            self.dropout_layers.append(dropout_layer)
            self.layers.append(layer)
            input_values = self.dropout_layers[-1].output('lrelu', alpha = 0.2)
            input_without_dropout = self.layers[-1].output('lrelu', alpha = 0.2)
        
        #############
        ## Decoder ##
        #############
        num_encoder_layers = len(self.layers)
        for decoder_layer, dec_params in enumerate(decoder_parameters[:-1]):
            filter_shape = dec_params[0]
            input_shape = dec_params[1]
            is_batch_norm = dec_params[2]
            W = dec_params[3]
            b = dec_params[4]
            gamma = dec_params[5]
            beta = dec_params[6]
            dropout = dec_params[7]
            
            skip_layer = num_encoder_layers - decoder_layer - 1
            
            if decoder_layer > 0:
                # change input shape!
                input_values = T.concatenate([input_values, self.dropout_layers[skip_layer].output], axis=1)
                input_without_dropout = T.concatenate([input_without_dropout, self.layers[skip_layer].output], axis=1)
            
            if dropout > 0.0:
                dropout_layer = DropoutUpconvLayer(input_values, filter_shape, input_shape,
                                                   is_batch_norm, W = W, b = b,gamma = gamma,
                                                   beta=beta)
            else:
                dropout_layer = UpconvolutionalLayer(input_values, filter_shape, input_shape,
                                                     is_batch_norm, W = W, b = b,gamma = gamma,
                                                     beta=beta)
            # without dropout
            # if dropout 0.0 then W = W, else W = W * (1 - dropout)
            layer = UpconvolutionalLayer(input_without_dropout, filter_shape, input_shape,
                                         is_batch_norm, W = W * (1 - dropout), b = b,gamma = gamma,
                                         beta=beta)
            self.layers.append(layer)
            input_without_dropout = self.layers[-1].output('relu')
            
            self.dropout_layers.append(dropout_layer)
            input_values = self.dropout_layers[-1].output('relu')
        
        dec_params = decoder_parameters[-1]
        filter_shape = dec_params[0]
        input_shape = dec_params[1]
        is_batch_norm = dec_params[2]
        W = dec_params[3]
        b = dec_params[4]
        gamma = dec_params[5]
        beta = dec_params[6]
        dropout = dec_params[7]
        
        input_values = T.concatenate([input_values, self.dropout_layers[0].output], axis=1)
        input_without_dropout = T.concatenate([input_without_dropout, self.layers[0].output], axis=1)
        
        if dropout > 0.0:
            layer = DropoutUpconvLayer(input_values, filter_shape, input_shape,
                                       is_batch_norm, W = W, b = b,gamma = gamma,
                                       beta=beta)
        else:
            layer = UpconvolutionalLayer(input_values, filter_shape, input_shape,
                                         is_batch_norm, W = W, b = b,gamma = gamma,
                                         beta=beta)
        self.dropout_layers.append(layer)

        self.params = [param for layer in self.dropout_layers 
                       for param in layer.params]
    
    def output(self, activation):

        if activation == 'tanh':
            self.generator_output = self.dropout_layers[-1].output(activation = 'tanh')
        elif activation == 'relu':
            self.generator_output = self.dropout_layers[-1].output(activation = 'relu')
        else:
            self.generator_output = self.dropout_layers[-1].output(activation = None)
            
        # concatenate with input
        #self.generator_output = T.concatenate([self.generator_output,
        #                                       self.input_values], axis = 1)
            
        return self.generator_output   

    def _get_image(self, contour):
        image = theano.function(
                [],
                self.generator_output,
                givens = {
                        self.input_layer: contour
                        }
                )
        return image()
        
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
    x = T.tensor4('x')
    input_x = x.reshape((30, 3, 64, 64))
    generator = Generator(input_x, BATCH_SIZE, 
                          encoder_parameters, decoder_parameters)
    
    input_values = np.random.randn(30, 3, 64, 64) * 100
    input_values = theano.shared(value = np.asanyarray(input_values, dtype = theano.config.floatX))
    a = theano.function(
            [],
            generator.output('tanh'),
            givens = {
                    x: input_values
                    }
            )
    temp = a()
    print(temp.shape)