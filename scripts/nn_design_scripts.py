#!/usr/bin/env python
# coding: utf-8

SEED=91
from numpy.random import seed
seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
# Required libraries
import os
# import glob
import numpy as np
import pandas as pd
import datetime as dt
from tensorflow import keras
import kerastuner as kt # for hypermodel


def proposed_NN(X, y, bio_layer, select_optimizer, select_activation, **kwargs):    
    '''    
    Proopsed NN architecture, with 1-layer or 2-layer options
    
    Parameters
    ----------
    X : dataframe
        The features of training set
    y : dataframe
        The truth label value of training set
    bio_layer : dataframe
        The biological knowledge of the nodes which are in 1st hidden layer
    select_optimizer : str
        Defining the optimizer parameter, choosing Adam or SGD
    select_optimizer : str
        Selec the optimizer parameter, choosing Adam or SGD
    
    **second_layer : boolean (default value is False)
        The definition of the 2nd layer. If it is FALSE then the design is with 1-Layer, if it is TRUE then the second hidden layer is included.
    
    Returns
    -------
    model : model
        The model information
    '''
    
    try:
        input_size = X.shape[1]
        unit_size = len(np.array(bio_layer)[0])
        size_output_layer = len(set(y.reshape(1,-1)[0]))
        
        second_layer = kwargs.get('second_layer', None)
    
        print('------------- NETWORK DESIGN - ARGUMENTS -------------')
        print('-- X.shape           ,', X.shape)
        print('-- y.shape           ,', y.shape)
        print('-- bio_layer.shape   ,', bio_layer.shape)
        print('-- input_size        ,', input_size)
        print('-- first_hidden_size ,', unit_size)
        print('-- size_output_layer ,', size_output_layer)
        print('-- optimizer         ,', select_optimizer)
        print('-- activation        ,', select_activation)
        
        keras.backend.clear_session()
        init = keras.initializers.GlorotUniform(seed=SEED)
        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(units = unit_size
                                         , input_dim=input_size
                                         , kernel_initializer=init
                                         , bias_initializer='zeros'
                                         , activation=select_activation
                                         , name='layer1'))

            model.set_weights([model.get_weights()[0] * np.array(bio_layer),  np.zeros((unit_size,)) ])

            if (second_layer==True):
                print('second_layer applied!! ')
                model.add(keras.layers.Dense(100, activation=select_activation, name='layer2'))

            model.add(keras.layers.Dense(size_output_layer, activation='softmax', name='layer3'))
            
            if select_optimizer == 'Adam':
                optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
            elif select_optimizer == 'SGD':
                optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # the parameter from paper 
            else:
                raise Exception('*** ERRROR in OPTIMIZER SELECTION, please select Adam or SGD')
                
            model.compile(optimizer=optimizer
                              , loss='categorical_crossentropy'
                              , metrics=['accuracy'] )
            
        return model
        
    except Exception as error:
            print('\n{0}'.format(error))    
    except:
        print("Unexpected error:", sys.exc_info()[0])
        

    
class tuning(kt.HyperModel):
    
    def __init__(self, X, y, bio_layer, select_optimizer, select_activation, **kwargs):
        self.X = X
        self.y = y
        self.select_optimizer = select_optimizer
        self.select_activation = select_activation

        self.input_size = X.shape[1]
        self.unit_size = len(np.array(bio_layer)[0])
        self.bio_layer = bio_layer
        self.size_output_layer = len(set(y.reshape(1,-1)[0]))
        self.second_layer = kwargs.get('second_layer', None)
            
        
    def build(self, hp):

        keras.backend.clear_session()
        init = keras.initializers.GlorotUniform(seed=SEED)
        
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(units = self.unit_size
                                         , input_dim=self.input_size
                                         , kernel_initializer=init
                                         , bias_initializer='zeros'
                                         , activation=self.select_activation
                                         , name='layer1'))

            model.set_weights([model.get_weights()[0] * np.array(self.bio_layer),  np.zeros((self.unit_size,)) ])
                
            if (self.second_layer==True):
                for i in range(hp.Int('n_layers', 1, 1)):  # adding variation of layers.
                    model.add(keras.layers.Dense(hp.Int(f'layer_{i}_units'
                                                        ,min_value=0
                                                        ,max_value=200
                                                        ,step=50)))
                    model.add(keras.layers.Activation(self.select_activation))    
                
            model.add(keras.layers.Dense(self.size_output_layer, activation='softmax', name='layer3'))
        
        hp_learning_rate = hp.Choice('learning_rate', values = [0.0001, 0.001, 0.01, 0.1, 0.2] )
        hp_momentum = hp.Choice('momentum', values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] )
        hp_decay = hp.Choice('decay', values=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8] )

        optimizer = keras.optimizers.SGD(learning_rate = hp_learning_rate
                                                        , momentum=hp_momentum
                                                        , decay=hp_decay)
        
        model.compile(optimizer = optimizer
                      , loss = 'categorical_crossentropy'
                      , metrics = ['accuracy'])
        return model
