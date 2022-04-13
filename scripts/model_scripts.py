#!/usr/bin/env python
# coding: utf-8

# Required libraries
# import os
# import pyreadr
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import KFold, train_test_split, LeaveOneGroupOut
from tensorflow import keras # to load model

def loading_model(path_, last_layer=1):
    """
    Loading model in given path
    Parameters
    ----------
    path_ : str
        The location of required model
    last_layer : int
        The layer number which the network ends
    Returns
    -------
    model_full : keras.Model
        The full version of required model 
    model_customize : keras.Model
        The model from input layer through the required layer
    """
    print('Loaded model!!', path_)
    model_load = keras.models.load_model(path_)

    model_full = keras.models.Model(inputs=model_load.layers[0].input
                       , outputs=model_load.layers[-1].output)
    model_customize = keras.models.Model(inputs=model_load.layers[0].input
                            , outputs=model_load.layers[last_layer].input)
    
    return (model_full, model_customize)
