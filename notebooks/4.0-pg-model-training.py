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
    ./models/NN/{EXPERIMENT}/{MODEL}.h5
    ./models/NN/{EXPERIMENT}/{MODEL-RESULT}.csv
'''

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import config as src

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split, LeaveOneGroupOut

from numba import cuda
import tensorflow as ts
from tensorflow import keras

# DEFAULT VALUES
epochs_default=100
batch_default=10
val_split=0.1

rand_state = 91
shuffle_=True
test_size = 0.3 # train_test_split
kf_split = 5 # KFold

def NN_training(experiment, location, dataset, bio_knowledge, split, nn_cv, save_model):
    time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time() dt.datetime.now().strftime('%Y%m%d_%I%M%S%p')
    print(type(save_model))
    print(save_model)
    
    if location == 'external':
        loc_ds = src.DIR_DATA_EXTERNAL
    elif location == 'processed':
        loc_ds = src.DIR_DATA_PROCESSED
    else:
        print('please give a valid location!!!')
    
    # the output location
    loc_output = os.path.join(src.DIR_MODELS, nn_cv, experiment)
    src.define_folder(loc_=loc_output)
    
    print('FILE FORMAT, ', dataset.split('.')[1])
    if dataset.split('.')[1]=='pck':
        df_raw = pd.read_pickle(os.path.join(loc_ds, experiment, dataset))
    else:
        df_raw = pd.read_csv(os.path.join(loc_ds, experiment, dataset))
    
    # SORTING GENE LIST
    df_raw = pd.concat([(df_raw.iloc[:, :-1]).astype(float) ,df_raw.iloc[:, -1]], axis=1)
    sort_genes = sorted(df_raw.columns[:-1])
    sort_genes.extend(df_raw.columns[-1:])
    df_raw = df_raw[sort_genes]
    
    # Importing all prior biological knowledge and combine all genes to create a common gene list
    list_gene = None
    if (bio_knowledge!=None):
        df_bio = pd.DataFrame(pd.read_csv(os.path.join(src.DIR_DATA_PROCESSED, bio_knowledge), index_col=0)).sort_index()
        df_bio_filtered = df_bio.iloc[df_bio.index.isin(df_raw.columns), :]

    del(df_bio)
    print('Dataset cell type, ', df_raw.groupby('cell_type').size())
    print('\nDataset shape             , ', df_raw.shape)
    print('Biological knowledge shape, ', df_bio_filtered.shape)

    print('\nDataset gene order top 10           ,', list(df_raw.columns[:10]))
    print('Biological knowledge gene order top 10, ', list(df_bio_filtered.index[:10].values))
    
    ohe = OneHotEncoder()
    X = df_raw.iloc[:, :-1].values
    y = df_raw.iloc[:, -1:].values
    y_ohe = ohe.fit_transform(y).toarray()
#     groups = y.reshape(1,-1)[0]
    
    X_train, y_train, X_test, y_test = [], [], [], []

    if split == 'KFold':
        kf = KFold(n_splits=kf_split, shuffle=shuffle_ , random_state=rand_state)
        print('KFold split applied!! The number of KFold is {}'.format(kf.get_n_splits()))
        for train_index, test_index in kf.split(X, y): # so.split(X, y)
            print(train_index, len(train_index))

            X_train.append(X[train_index])
            X_test.append(X[test_index])
            y_train.append(y_ohe[train_index])
            y_test.append(y_ohe[test_index])
#         print(np.array(X_train).shape)

    elif split=='train_test_split':
        print('train_test_split split applied! Test size is, ', test_size)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size
                                                        , shuffle=shuffle_
                                                        , random_state=rand_state
                                                        , stratify=y_ohe)

        X_train.append(Xtrain)
        X_test.append(Xtest)
        y_train.append(ohe.transform(ytrain).toarray())
        y_test.append(ohe.transform(ytest).toarray())
#         print(np.array(X_train).shape)

    elif split=='None':
        print('Full dataset!!')
        X_train.append(X)
        y_train.append(y_ohe)

    START_TRAINING = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    df_nn = pd.DataFrame()
    
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" # Stop training when `val_loss` is no longer improving
                                           , min_delta=1e-5   # "no longer improving" being defined as "no better than 1e-5 less"
                                           , patience=3       # "no longer improving" being further defined as "for at least 3 epochs"
                                           , verbose=1 ) ]
    
    for i in range(len(X_train)):
        
        print('EXPERIMENT --- '+str(i+1)+'/'+str(len(X_train)))
#         1-Layer with biological layer design
        time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        model_a1 = src.proposed_NN(X=X, y=y, bio_layer=df_bio_filtered, design_type='bio')
        model_a1.fit(X_train[i], y_train[i]
                  , epochs=epochs_default
                  , batch_size=batch_default
                  , verbose=1
                  , callbacks=callbacks
                  , validation_split=val_split)
        
        if split!='None':
            y_pred_a1 = model_a1.predict(X_test[i])
            df_proba = pd.DataFrame(y_pred_a1, columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
            df_pred = pd.DataFrame(ohe.inverse_transform(y_pred_a1).reshape(1,-1)[0], columns=['prediction'])
            df_ground_truth = pd.DataFrame(ohe.inverse_transform(np.array(y_test[i])).reshape(1,-1)[0], columns=['ground_truth'])
            df_nn_a1 = pd.concat([df_proba, df_pred, df_ground_truth], axis=1)
            df_nn_a1['design'] ='a1'
            df_nn_a1['index_split'] = i
            df_nn_a1['split'] = split
            df_nn = pd.concat([df_nn, df_nn_a1])
            
        if save_model==False:
            print('model_a1 deleted!!')
            del(model_a1)
            
        time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
        
#         2-Layer with biological layer design
        time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        model_a2 = src.proposed_NN(X=X, y=y, bio_layer=df_bio_filtered, design_type='bio', second_layer=True)
        model_a2.fit(X_train[i], y_train[i]
                  , epochs=epochs_default
                  , batch_size=batch_default
                  , verbose=1
                  , callbacks=callbacks
                  , validation_split=val_split)
        
        if split!='None':
            y_pred_a2 = model_a2.predict(X_test[i])
            df_proba = pd.DataFrame(y_pred_a2, columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
            df_pred = pd.DataFrame(ohe.inverse_transform(y_pred_a2).reshape(1,-1)[0], columns=['prediction'])
            df_ground_truth = pd.DataFrame(ohe.inverse_transform(np.array(y_test[i])).reshape(1,-1)[0], columns=['ground_truth'])
            df_nn_a2 = pd.concat([df_proba, df_pred, df_ground_truth], axis=1)
            df_nn_a2['design'] ='a2'
            df_nn_a2['index_split'] = i
            df_nn_a2['split'] = split
            df_nn = pd.concat([df_nn, df_nn_a2])
        
        if save_model==False:
            print('model_a2 deleted!!')
            del(model_a2)
            
        time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
        
        if save_model==True:
            model_a1.save(os.path.join(loc_output, 'model_a1_'+dataset.split('.')[0]+'_'+split+'_trained.h5'))
            model_a2.save(os.path.join(loc_output, 'model_a2_'+dataset.split('.')[0]+'_'+split+'_trained.h5'))
    
    END_TRAINING  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    print('ended in ', END_TRAINING)
    print('EXPERIMENT COMPLETED! Total time is, ', (dt.datetime.strptime(END_TRAINING,'%H:%M:%S') - dt.datetime.strptime(START_TRAINING,'%H:%M:%S')))
    
    if split!='None':
        df_nn.to_pickle(os.path.join(loc_output,'model_result_'+dataset.split('.')[0]+'_'+split+'.pck'))
        print('file is exported in ', os.path.join(loc_output,'model_result_'+dataset.split('.')[0]+'_'+split+'.pck'))
        
    cuda.select_device(0)
    cuda.close()
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment', help='specify the experiment, the experiment name should be located in ./data/external')
    parser.add_argument('-loc', '--location', help='specify the dataset location in data folder, in ./data/{LOCATION}')
    parser.add_argument('-ds' , '--dataset', help='specify the dataset, in ./data/external/{FILE NAME}')
    parser.add_argument('-pbk', '--bio_knowledge', help='specifying the biological knowledge dataset. None for keeping all genes')
    parser.add_argument('-split', '--split', help='specifying dataset split, etc, train_test_split or KFold')
    parser.add_argument('-nncv', '--nn_cv', help='specifying the model NN-training or cross-validation')
    parser.add_argument('-model', '--save_model', help='exporting the trained model')
    
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    NN_training(args.experiment
                , args.location
                , args.dataset
                , args.bio_knowledge
                , args.split
                , args.nn_cv
                , eval(args.save_model))