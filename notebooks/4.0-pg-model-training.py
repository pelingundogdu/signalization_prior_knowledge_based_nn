#!/usr/bin/env python

'''
DESCRIPTION
-----------
    Traning proposed neural network
    
USAGE
-----
    [PROJECT_PATH]/$ python notebooks/04-pg-model-training.py -desing              {DESIGN NAME}
                                                              -second_hidden_layer {SECOND HIDDEN LAYER}
                                                              -ds                  {DATASET}
                                                              -pbk                 {BIOLOGICAL KNOWLEDGE} 
                                                              -dense               {NUMBER of DENSE LAYER}
                                                              -split               {DATASET SPLIT OPERATION}
                                                              -training_or_cv      {TRAINING or CROSS-VALIDATION}
                                                              -optimizer           {OPTIMIZER}
                                                              
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

# to get reproducible results
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import config as src

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneGroupOut

from numba import cuda
from tensorflow import keras

# DEFAULT VALUES
epochs_default=100 # for model design
batch_default=10   # for model design
val_split=0.1      # for model design
rand_state = 91    # dataset split
test_size = 0.3    # train_test_split
kf_split = 5       # KFold

def NN_training(design_name, second_hidden_layer, dataset, bio_knowledge, dense_nodes, split, training_or_cv, optimizer, target_column):
    time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time() dt.datetime.now().strftime('%Y%m%d_%I%M%S%p')
        
    # the output location
    loc_output = os.path.join(src.DIR_MODELS, training_or_cv, dataset.split('/')[1])
    src.define_folder(loc_=loc_output)
    df_bio_org, df_dense = pd.DataFrame(), pd.DataFrame()
    split_index=''
    print(type(second_hidden_layer))
    print(second_hidden_layer)
    
    print('FILE FORMAT, ', dataset.split('.')[1])
    if dataset.split('.')[1]=='pck':
        df_org = pd.read_pickle(os.path.join(src.DIR_DATA, dataset))
    else:
        df_org = pd.read_csv(os.path.join(src.DIR_DATA, dataset))
    
    print('df_org, \n', df_org)
    # SORTING GENE LIST
    df_org = pd.concat([(df_org.iloc[:, :-1]).astype(float) ,df_org.iloc[:, -1]], axis=1)
    sort_genes = sorted(df_org.columns[:-1])
    sort_genes.extend(df_org.columns[-1:])
    df_org = df_org[sort_genes]
        
        
    print('df_org, \n', df_org)
    print('Dataset cell type         , ', df_org.groupby(target_column).size().index.values)
    print('Dataset shape             , ', df_org.shape)
    # Importing all prior biological knowledge and combine all genes to create a common gene list
    if (bio_knowledge != 'None'):
        df_bio_org = pd.DataFrame(pd.read_csv(os.path.join(src.DIR_DATA_PROCESSED, bio_knowledge), index_col=0)).sort_index()
        df_bio = df_bio_org.iloc[df_bio_org.index.isin(df_org.columns), :]
        del(df_bio_org)
        print(df_bio)
        print('Biological knowledge shape, ', df_bio.shape)        
        df_first_hidden_layer = pd.merge(left=pd.DataFrame(sorted(df_org.columns[:-1])).set_index(0)
                                         , right=df_bio
                                         , left_index=True
                                         , right_index=True
                                         , how='left').fillna(0.0)

    
    if dense_nodes != 0:
        df_dense = pd.DataFrame(df_org.columns[:-1]).set_index('Sample')
        for i in range(dense_nodes):
            df_dense['dense'+str(i)] = 1.0
        df_first_hidden_layer = df_dense.copy()
    print(df_first_hidden_layer)
    if ( ( bio_knowledge != 'None' ) and ( dense_nodes != 0 ) ):
        df_first_hidden_layer = pd.merge(df_dense, df_bio, left_index=True, right_index=True, how='left').fillna(0)
        
    print(df_first_hidden_layer)

    print('df_org shape              , ', df_org.shape)
#     print('df_features_filtered shape, ', df_features_filtered.shape)

    
    print('First hidden layer shape  , ', df_first_hidden_layer.shape)
    print('**** The gene order in biological source and dataset is ordered!! -> {} '.format( np.all(df_org.columns[:-1] == df_first_hidden_layer.index.values) ))
    
    ohe = OneHotEncoder()
    X = df_org.iloc[:, :-1].values
    y = df_org.iloc[:, -1:].values
    y_ohe = ohe.fit_transform(y).toarray()
#     groups = y.reshape(1,-1)[0]
    
    X_train, y_train, X_test, y_test = [], [], [], []

    if split == 'StratifiedKFold':
        kf = StratifiedKFold(n_splits=kf_split, shuffle=True, random_state=rand_state)
        print('KFold split applied!! The number of KFold is {}'.format(kf.get_n_splits()))
        for train_index, test_index in kf.split(X, y):
            print(train_index, len(train_index))

            X_train.append(X[train_index])
            X_test.append(X[test_index])
            y_train.append(y_ohe[train_index])
            y_test.append(y_ohe[test_index])
#         print(np.array(X_train).shape)

    elif split=='train_test_split':
        print('train_test_split split applied! Test size is, ', test_size)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size
                                                        , shuffle=True
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
        split_index='with_full_dataset'
        split=''

    START_TRAINING = dt.datetime.now().time().strftime('%H:%M:%S')
    df_nn = pd.DataFrame()
    
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" # Stop training when `val_loss` is no longer improving
                                           , min_delta=1e-5   # "no longer improving" being defined as "no better than 1e-5 less"
                                           , patience=3       # "no longer improving" being further defined as "for at least 3 epochs"
                                           , verbose=1 ) ]
    
    
    for i in range(len(X_train)):
        
        if len(X_train)>1:
            split_index='_'+str(i+1)
        print('EXPERIMENT --- '+str(i+1)+'/'+str(len(X_train)))
#         1-Layer with biological layer design
        time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        model_trained = src.proposed_NN(X=X, y=y, bio_layer=df_first_hidden_layer, design_type='bio', select_optimizer=optimizer, second_layer=second_hidden_layer)
        model_trained.fit(X_train[i], y_train[i]
                  , epochs=epochs_default
                  , batch_size=batch_default
                  , verbose=1
                  , callbacks=callbacks
                  , validation_split=val_split)
        
        if split_index!='with_full_dataset':
            y_pred_a1 = model_trained.predict(X_test[i])
            df_proba = pd.DataFrame(y_pred_a1, columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
            df_pred = pd.DataFrame(ohe.inverse_transform(y_pred_a1).reshape(1,-1)[0], columns=['prediction'])
            df_ground_truth = pd.DataFrame(ohe.inverse_transform(np.array(y_test[i])).reshape(1,-1)[0], columns=['ground_truth'])
            df_nn_a1 = pd.concat([df_proba, df_pred, df_ground_truth], axis=1)
            df_nn_a1['design'] =design_name
            df_nn_a1['index_split'] = i
            df_nn_a1['split'] = split
            df_nn = pd.concat([df_nn, df_nn_a1])
            
        time_end  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
        print('\nELAPSED TIME, ', (dt.datetime.strptime(time_end,'%H:%M:%S') - dt.datetime.strptime(time_start,'%H:%M:%S')))
        
        print('Model exported!! - '+str(i+1)+'/'+str(len(X_train)))
        model_trained.save(os.path.join(loc_output, 'model_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+split+split_index+'_'+optimizer+'.h5'))
    
        print('Model deleted!! - '+str(i+1)+'/'+str(len(X_train)))
        del(model_trained)
        
    END_TRAINING  = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time()
    print('ended in ', END_TRAINING)
    print('EXPERIMENT COMPLETED! Total time is, ', (dt.datetime.strptime(END_TRAINING,'%H:%M:%S') - dt.datetime.strptime(START_TRAINING,'%H:%M:%S')))
    
    if split_index!='with_full_dataset':
        df_nn.to_pickle(os.path.join(loc_output,'model_result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+split+'_'+optimizer+'.pck'))
        print('file is exported in ', os.path.join(loc_output,'model_result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+split+'_'+optimizer+'.pck'))
        
    cuda.select_device(0)
    cuda.close()
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-design', '--design_name', help='name of design')
    parser.add_argument('-second_hidden_layer', '--second_hidden_layer', help='second layer exist or not')
    parser.add_argument('-ds', '--dataset', help='the experiment dataset')
    parser.add_argument('-pbk', '--bio_knowledge', help='integrated prior biologicl knowledge')
    parser.add_argument('-dense', '--dense_nodes', help='integrated dense node into first hidden layer')
    parser.add_argument('-split', '--split', help='specifying dataset split, etc, train_test_split or KFold')
    parser.add_argument('-training_or_cv', '--nn_cv', help='specifying the model NN-training or cross-validation')
    parser.add_argument('-optimizer', '--optimizer', help='selecting the optimizer')
    parser.add_argument('-target_column', '--target_column', help='target column')
    
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    NN_training(args.design_name
                , eval(args.second_hidden_layer)
                , args.dataset
                , args.bio_knowledge
                , int(args.dense_nodes)
                , args.split
                , args.nn_cv
                , args.optimizer
                , args.target_column
               )