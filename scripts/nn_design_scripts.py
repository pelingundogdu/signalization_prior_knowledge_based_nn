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

def proposed_NN(X, y, bio_layer, design_type, **kwargs):    
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

            optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999) # the parameter from paper 
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
        
# def NN_design(train_X, train_y, test_X, groups, bio_layer, size_epochs, size_batch, design_type, val_split=0.1, **kwargs):    
#     '''    
#     Proopsed NN architecture, with 1-layer or 2-layer options
    
#     Parameters
#     ----------
#     train_X : dataframe
#         The features of training set
#     train_y : dataframe
#         The truth label value of training set
#     test_X : dataframe
#         The features of test set
#     bio_layer : dataframe
#         The biological knowledge of the nodes which are in 1st hidden layer
#     size_epochs : int
#         The epoch number (for NN design)
#     size_batch : int
#         The batch size (for NN design)
#     design_type : list (fully or bio)
#         To describe the design of neural network, if there is biologic knowledge then "bio", if not then "fully"
#     val_split : float (default, 0.1)
#         The validation split size in fitting step of model (for NN design)
        
#     **second_layer : boolean (default value is False)
#         The definition of the 2nd layer. If it is FALSE then the design is with 1-Layer, if it is TRUE then the second hidden layer is included.
#     **size_units : int
#         The size of first hidden layer, if there is no bio_layer then specify the size of first hidden layer (for NN design)
    
#     Returns
#     -------
#     model : model
#         The model information
#     y_pred : list
#         The prediction list
#     '''

#     time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
#     print('started!!!     ', time_start)

#     input_size = train_X.shape[1]

#     second_layer = kwargs.get('second_layer', None)
#     size_units = kwargs.get('size_units', None)

# #     print('second_layer, ', second_layer)
# #     print('size_units,', size_units)

#     if design_type == 'fully' and size_units!=None:
#         first_hidden_layer_size = size_units
#         print('  -> WARNING! Network is fully connected with {} nodes in first hidden layer.'.format(size_units))
#     elif design_type == 'bio':
#         first_hidden_layer_size = len(np.array(bio_layer)[0])
#         print('  -> Network designed with prior biological knowledge with {} nodes in first hidden layer.'.format(first_hidden_layer_size))
#         if size_units!=None:
#             print('    -> do not need to define size_units!!')
#     elif design_type == 'fully' and size_units==None:
#         ex = 'WARNING!, please define size_units parameter!!'
#     else:
#         ex = 'ERROR with parameters!!!'

#     try: 
#         print("Network designed with \n  input layer {:}, \n  first hidden layer {:}, \n  epoch {:}, \n  batch_size {:}".format(input_size, first_hidden_layer_size, size_epochs, size_batch))

#         keras.backend.clear_session()
#         size_output_layer = len(set(groups))
#         print('\n')
#         strategy = tf.distribute.MirroredStrategy()

#         with strategy.scope():

#             model = keras.models.Sequential()
#             model.add(keras.layers.Dense(units = first_hidden_layer_size
#                                          , input_dim=input_size
#                                          , kernel_initializer='glorot_uniform'
#                                          , bias_initializer='zeros'
#                                          , activation='relu'
#                                          , name='layer1'))

#             if (design_type=='bio'):
#                 #print('set_weight applied!!')
#                 model.set_weights([model.get_weights()[0] * np.array(bio_layer),  np.zeros((first_hidden_layer_size,)) ])

#             if (second_layer==True):
#                 print('second_layer applied!! ')
#                 model.add(keras.layers.Dense(100, activation='relu', name='layer2'))

#             model.add(keras.layers.Dense(size_output_layer, activation='softmax', name='layer3'))

#             optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999) # the parameter from paper 
#             model.compile(optimizer=optimizer
#                           , loss='categorical_crossentropy'
#                           , metrics=['accuracy']
#                                  )
# #         cuda.select_device(0)
# #         cuda.close()

#         print(model.summary())
# #         print('\ninput layer size       , ', model.layers[0].input_shape)
# #         print('first hidden layer size, ', model.layers[0].output_shape)

#         callbacks = [keras.callbacks.EarlyStopping(
#             # Stop training when `val_loss` is no longer improving
#             monitor="val_loss",
#             # "no longer improving" being defined as "no better than 1e-5 less"
#             min_delta=1e-5,
#             # "no longer improving" being further defined as "for at least 3 epochs"
#             patience=3,
#             verbose=1,
#                 )
#             ]

#         model.fit(train_X, train_y, epochs=size_epochs, batch_size=size_batch, verbose=1, callbacks=callbacks, validation_split=val_split)
#         y_pred = model.predict(test_X)

#         keras.backend.clear_session()

#         time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
#         print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
#         print('\n')

#         return model, y_pred
#     except :
#         print(ex)