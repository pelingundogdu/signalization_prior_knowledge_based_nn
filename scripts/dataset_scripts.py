#!/usr/bin/env python
# coding: utf-8

# Required libraries
# import os
# import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

def dataframe_modification(experiment_dataset, target_col_index)
    '''
    Data uploading and preprocessing

    (1) applying lowercase opation for gene names
    (2) re-naming the last columns as cell type
    (3) reordering the dataset (cell type information is the last column)
    (4) changing the data type as float of gene expression

    Parameters
    ----------
    experiment_dataset : dataframe
    
    target_col_index : str
        The column number of target features, it differs depending on the using dataset. (-1 is for last column)

    Returns
    -------
    experiment_dataset : dataframe
        Uploaded and preprocessed dataset
    '''
    print('\n****** RAW DATASET ******')
    print('experiment dataset shape     , {0}\n'.format(df_experiment.shape))
    print(df_experiment.head(3))
    print(df_experiment.info())

#     (1) applying lowercase opation for gene names
    experiment_dataset.columns = experiment_dataset.columns.str.lower()
#     (2) re-naming the last columns as cell type
    experiment_dataset = experiment_dataset.rename(columns={experiment_dataset.columns[target_col_index]:'cell_type'})
#     (3) reordering the dataset (cell type information is the last column)
    col_name = list(experiment_dataset.columns)
    target_col_name = col_name.pop(target_col_index)
    col_name.append(target_col)
    experiment_dataset = experiment_dataset[col_name]
#     (4) changing the data type as float of gene expression
    experiment_dataset.iloc[:, :-1] = experiment_dataset.iloc[:, :-1].astype('float32')

    print('\n****** PROCESSED DATASET ******')
    print('experiment dataset shape     , {0}\n'.format(df_experiment.shape))
    print(df_experiment.head(3))
    print(df_experiment.info())
    
    return(df_experiment)


def sample_wise_normalization(dataset, target_col_index):
    '''
    Applying sample-wise normalization into dataset

    Parameters
    ----------
    dataset : dataframe

    Returns
    -------
    df_scaler : dataframe
        Scaled dataset
    '''
    print('    -> sample wise normalization implemented!')
    df_sample_wise = pd.concat([dataset.iloc[:, :target_col_index].div(dataset.iloc[:, :target_col_index].sum(axis=1), axis=0)*1e6
                                , dataset.iloc[:, target_col_index]], axis=1)
    return(df_sample_wise)

def scaler_normalization(dataset, target_col_index, scaler_name):
    '''
    Applying scaler or required mathematical function into dataset

    Parameters
    ----------
    dataset : dataframe
    
    scaler_name : str, [ss, mms, log1p]
        the scaler information

    Returns
    -------
    df_scaler : dataframe
        Scaled dataset
        
    '''
    try :
        if scaler_name == 'ss':
            scaler = StandardScaler()
        elif scaler_name == 'mms':
            scaler = MinMaxScaler()
        elif scaler_name == 'log1p':
            scaler = FunctionTransformer(np.log1p)
        else:
            raise Exception('Please, choose one of the normzalization options --> standard, minmax or log1p !!!')

        print('    -> Normalization implemented! -- {0}'.format(scaler_name))

        df_scaler = pd.concat([pd.DataFrame(scaler.fit_transform(dataset.iloc[: , :target_col_index]) , columns=dataset.columns[:target_col_index]) 
                               ,dataset.iloc[:, target_col_index]], axis=1)

        return(df_scaler)
    
    except Exception as error:
            print('\n{0}'.format(error))    
    except:
        print("Unexpected error:", sys.exc_info()[0]) 