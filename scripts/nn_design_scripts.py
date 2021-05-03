#!/usr/bin/env python
# coding: utf-8



# Required libraries
import os
# import glob
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from tensorflow import keras

def proposed_NN(X, y, bio_layer, design_type, select_optimizer, **kwargs):    
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
    design_type : list (fully or bio)
        To describe the design of neural network, if there is biologic knowledge then "bio", if not then "fully"
    
    **second_layer : boolean (default value is False)
        The definition of the 2nd layer. If it is FALSE then the design is with 1-Layer, if it is TRUE then the second hidden layer is included.
    **size_units : int
        The size of first hidden layer, if there is no bio_layer then specify the size of first hidden layer (for NN design)
    
    Returns
    -------
    model : model
        The model information
    '''
    input_size = X.shape[1]

    second_layer = kwargs.get('second_layer', None)
    size_units = kwargs.get('size_units', None)

#     print('second_layer, ', second_layer)
#     print('size_units,', size_units)

    try:
    
        if design_type == 'fully' and size_units!=None:
            first_hidden_layer_size = size_units
            print('  -> Network is fully connected with {} nodes in first hidden layer.'.format(size_units))
        elif design_type == 'bio':
            first_hidden_layer_size = len(np.array(bio_layer)[0])
            print('  -> Network designed with prior biological knowledge with {} nodes in first hidden layer.'.format(first_hidden_layer_size))
            if size_units!=None:
                print('    -> do not need to define size_units!!')
        elif design_type == 'fully' and size_units==None:
            raise Exception('ERROR!, please define size_units parameter!!')
        else:
            raise Exception('ERROR with parameters!!!')

        size_output_layer = len(set(y.reshape(1,-1)[0]))
    
        print('------------- NETWORK DESIGN - ARGUMENTS -------------')
        print('-- X.shape                  ,', X.shape)
        print('-- y.shape                  ,', y.shape)
        print('-- bio_layer.shape          ,', bio_layer.shape)
        print('-- design_type              ,', design_type)
        print('------------- NETWORK DESIGN - CALCULATED -------------')
        print('-- input_size               ,', input_size)
        print('-- first_hidden_layer_size  ,', first_hidden_layer_size)
        print('-- size_output_layer        ,', size_output_layer)
    
        
        keras.backend.clear_session()
        
        print('\n')
        strategy = tf.distribute.MirroredStrategy()
        print('first_hidden_layer_size', first_hidden_layer_size)
        with strategy.scope():

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(units = first_hidden_layer_size
                                         , input_dim=input_size
                                         , kernel_initializer='glorot_uniform'
                                         , bias_initializer='zeros'
                                         , activation='relu'
                                         , name='layer1'))

            if (design_type=='bio'):
                #print('set_weight applied!!')
                model.set_weights([model.get_weights()[0] * np.array(bio_layer),  np.zeros((first_hidden_layer_size,)) ])

            if (second_layer==True):
                print('second_layer applied!! ')
                model.add(keras.layers.Dense(100, activation='relu', name='layer2'))

            model.add(keras.layers.Dense(size_output_layer, activation='softmax', name='layer3'))
            
            if select_optimizer == 'Adam':
                optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999) # the parameter from paper 
            elif select_optimizer == 'SGD':
                optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # the parameter from paper 
            else:
                raise Exception('*** ERRROR in OPTIMIZER SELECTION, please select Adam or SGD')
                
            model.compile(optimizer=optimizer
                          , loss='categorical_crossentropy'
                          , metrics=['accuracy'] )
            
        print(model.summary())
# #         print('\ninput layer size       , ', model.layers[0].input_shape)
# #         print('first hidden layer size, ', model.layers[0].output_shape)

        return model
        
    except Exception as error:
            print('\n{0}'.format(error))    
    except:
        print("Unexpected error:", sys.exc_info()[0])
        
