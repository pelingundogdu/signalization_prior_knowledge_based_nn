#!/usr/bin/env python
# coding: utf-8
import os
SEED = 91 # for reproducible result
# to get reproducible results
import tensorflow as tf
tf.random.set_seed(SEED)
import numpy as np
np.random.seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
# Required libraries
from tensorflow import keras

def autoencoder_one_hidden_layer(X, bio_layer, select_optimizer, select_activation):
    '''    
    Autoencoder architecture, with 1-layer option
    
    Parameters
    ----------
    X : dataframe
        The features of training set
    bio_layer : dataframe
        The biological knowledge of the nodes which are in 1st hidden layer
    select_optimizer : str
        Defining the optimizer parameter, choosing Adam or SGD
    select_optimizer : str
        Selec the optimizer parameter, choosing Adam or SGD
    Returns
    -------
    model : model
        The model information
    '''
    
    print('AUTOENCODER SCRIPT OK')
    
    unit_size = bio_layer.shape[1] 
    input_size = X.shape[1]
    
    print(f'input size {input_size}, unit_size {unit_size}')
    
    keras.backend.clear_session()
    
    init = keras.initializers.GlorotUniform(seed=SEED)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # This is our input image
        input_layer = keras.Input(shape=(input_size,))
        # "encoded" is the encoded representation of the input
        encoded = keras.layers.Dense(unit_size
                                     , kernel_initializer=init
                                     , bias_initializer='zeros'
                                     , activation=select_activation
                                     , name='layer1')(input_layer)
        # "decoded" is the lossy reconstruction of the input
        decoded = keras.layers.Dense(input_size, activation='sigmoid')(encoded)

        # This model maps an input to its reconstruction
        autoencoder = keras.Model(input_layer, decoded)

        if select_optimizer == 'Adam':
            optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        elif select_optimizer == 'SGD':
            optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # the parameter from paper 
        else:
            raise Exception('*** ERRROR in OPTIMIZER SELECTION, please select Adam or SGD')

    autoencoder.compile(optimizer=optimizer
                  , loss='mean_squared_error')
    
    
    return autoencoder
