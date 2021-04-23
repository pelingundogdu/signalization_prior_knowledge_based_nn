#!/usr/bin/env python

'''
DESCRIPTION
-----------
    Traning proposed neural network
    
USAGE
-----
    [PROJECT_PATH]/$ python notebooks/04-pg-model-training.py -exp   {EXPERIMENT}
                                                              -loc   {DATASET LOCATION}
                                                              -ds    {DATASET} 
                                                              -pbk   {BIOLOGICAL KNOWLEDGE} 
                                                              -split {DATASET SPLIT OPERATION}
                                                              -nncv  {TRAINING or CROSS-VALIDATION}
RETURN
------
    {MODEL}.h5 : h5 file
        Trained model
    {MODEL-RESULT}.csv : csv file
        The model result with probabilities, prediction label and ground truth

EXPORTED FILE(s) LOCATION
-------------------------
    ./models/CV/{EXPERIMENT}/{MODEL}.h5
    ./models/CV/{EXPERIMENT}/{MODEL-RESULT}.csv
'''

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import settings as ssrp, dataset_scripts as dsrp, path_scripts as psrp, model_scripts as msrp, nn_design_scripts as nnsrp

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split, LeaveOneGroupOut, RepeatedStratifiedKFold

from numba import cuda
import tensorflow as ts
from tensorflow import keras

# DEFAULT VALUES
epochs_default=100
batch_default=10

rand_state = 91
shuffle_=True
n_split = 10 # number of split
n_repeat = 50 # number of repetation

def cross_validation(experiment, location, dataset, bio_knowledge, nn_cv):
    time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time() dt.datetime.now().strftime('%Y%m%d_%I%M%S%p')
    
    if location == 'external':
        loc_ds = ssrp.DIR_DATA_EXTERNAL
    elif location == 'processed':
        loc_ds = ssrp.DIR_DATA_PROCESSED
    else:
        print('please give a valid location!!!')
    
    # the output location
    loc_output = os.path.join(ssrp.DIR_MODELS, nn_cv, experiment)
    psrp.define_folder(loc_=loc_output)
    
    print('FILE FORMAT, ', dataset.split('.')[1])
    
    if dataset.split('.')[1]=='pck':
        df_processed = pd.read_pickle(os.path.join(loc_ds, experiment, dataset))
        df_processed = pd.concat([(df_processed.iloc[:, :-1]).astype(float) ,df_processed.iloc[:, -1]], axis=1)
    else:
        df_processed = pd.read_csv(os.path.join(loc_ds, experiment, dataset))

    sort_genes = sorted(df_processed.columns[:-1])
    sort_genes.extend(df_processed.columns[-1:])
    df_processed = df_processed[sort_genes]
    
    # Importing all prior biological knowledge and combine all genes to create a common gene list
    list_gene = None
    if (bio_knowledge!=None):
        df_bio = pd.DataFrame(pd.read_csv(os.path.join(ssrp.DIR_DATA_PROCESSED, bio_knowledge), index_col=0)).sort_index()
        df_bio_filtered = df_bio.iloc[df_bio.index.isin(df_processed.columns), :]

#     sort_genes = sorted(df_processed.columns[:-1])
#     df_bio_filtered = df_bio.iloc[df_bio.index.isin(sort_genes), :]
#     if sort_genes == list(df_bio_filtered.index):
#         print('Dataset and biological info are same ordered!')

#     sort_genes.extend(df_processed.columns[-1:])
#     df_processed = df_processed[sort_genes]
    
    del(df_bio)
    print('Dataset cell type, ', df_processed['cell_type'].value_counts())
    print('\nDataset shape             , ', df_processed.shape)
    print('Biological knowledge shape, ', df_bio_filtered.shape)

#     print('\nDataset gene order top 10              ,', list(df_processed.columns[:10]))
#     print('Biological knowledge gene order top 10, ', list(df_bio_filtered.index[:10].values))
    
    ohe = OneHotEncoder()
    X = df_processed.iloc[:, :-1]
    if np.all(X.columns == df_bio_filtered.index):
        print('Genes are at the same ordered!')
    else :
        print('WARNING!! - To check the gene order!!!')
    X = X.values
    y = df_processed.iloc[:, -1:].values
    y_ohe = ohe.fit_transform(y).toarray()
    groups = y.reshape(1,-1)[0]
    
    X_train, y_train, X_test, y_test = [], [], [], []
    
    rskf = RepeatedStratifiedKFold(n_splits=n_split, n_repeats=n_repeat, random_state=rand_state)
    for i, indexes in enumerate(rskf.split(X, y, groups=y)):

        train_index=indexes[0]
        test_index=indexes[1]
        print(i, " - TRAIN:", train_index[:10], "TEST:", test_index[:10])
#         print(i, " - TRAIN:", train_index[-3:], "TEST:", test_index[-3:])
#         print(len(test_index), len(train_index))

        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y_ohe[train_index])
        y_test.append(y_ohe[test_index])


    START_TRAINING = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    df_nn = pd.DataFrame()
    for i in range(len(X_train)):
        
        print('EXPERIMENT --- '+str(i+1)+'/'+str(len(X_train)))
#         1-Layer with biological layer design    
        model_a1, y_pred_a1 = nnsrp.NN_design(train_X=X_train[i]
                                              , train_y=y_train[i]
                                              , test_X=X_test[i]
                                              , groups=groups
                                              , bio_layer=df_bio_filtered
                                              , size_epochs=epochs_default
                                              , size_batch=batch_default
                                              , design_type='bio'
                                              , val_split=0.1)

        df_proba = pd.DataFrame(y_pred_a1, columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
        df_pred = pd.DataFrame(ohe.inverse_transform(y_pred_a1).reshape(1,-1)[0], columns=['prediction'])
        df_ground_truth = pd.DataFrame(ohe.inverse_transform(np.array(y_test[i])).reshape(1,-1)[0], columns=['ground_truth'])
        df_nn_a1 = pd.concat([df_proba, df_pred, df_ground_truth], axis=1)
        df_nn_a1['design'] ='a1'
        df_nn_a1['index_split'] = i

        df_nn = pd.concat([df_nn, df_nn_a1])
#         2-Layer with biological layer design
        model_a2, y_pred_a2 = nnsrp.NN_design(train_X=X_train[i]
                                              , train_y=y_train[i]
                                              , test_X=X_test[i]
                                              , groups=groups
                                              , bio_layer=df_bio_filtered
                                              , size_epochs=epochs_default
                                              , size_batch=batch_default
                                              , design_type='bio'
                                              , val_split=0.1
                                              , second_layer=True)

        df_proba = pd.DataFrame(y_pred_a2, columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
        df_pred = pd.DataFrame(ohe.inverse_transform(y_pred_a2).reshape(1,-1)[0], columns=['prediction'])
        df_ground_truth = pd.DataFrame(ohe.inverse_transform(np.array(y_test[i])).reshape(1,-1)[0], columns=['ground_truth'])
        df_nn_a2 = pd.concat([df_proba, df_pred, df_ground_truth], axis=1)
        df_nn_a2['design'] ='a2'
        df_nn_a2['index_split'] = i

        df_nn = pd.concat([df_nn, df_nn_a2])
        
    df_nn.to_pickle(os.path.join(loc_output,'cv_result_'+dataset.split('.')[0]+'.pck'))

    END_TRAINING  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    print('ended in ', END_TRAINING)
    print('EXPERIMENT COMPLETED! Total time is, ', (dt.datetime.strptime(END_TRAINING,'%H:%M:%S') - dt.datetime.strptime(START_TRAINING,'%H:%M:%S')))
    print('file is exported in ', os.path.join(loc_output,'cv_result_'+dataset.split('.')[0]+'.pck'))
    
    cuda.select_device(0)
    cuda.close()
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment', help='specify the experiment, the experiment name should be located in ./data/external')
    parser.add_argument('-loc', '--location', help='specify the dataset location in data folder, in ./data/{LOCATION}')
    parser.add_argument('-ds' , '--dataset', help='specify the dataset, in ./data/external/{FILE NAME}')
    parser.add_argument('-pbk', '--bio_knowledge', help='specifying the biological knowledge dataset. None for keeping all genes')
    parser.add_argument('-nncv', '--nn_cv', help='specifying the model NN-training or cross-validation')
    
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    cross_validation(args.experiment, args.location, args.dataset, args.bio_knowledge, args.nn_cv)