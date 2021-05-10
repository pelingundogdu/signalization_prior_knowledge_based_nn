#!/usr/bin/env python

'''
DESCRIPTION
-----------
    Traning proposed neural network
    
USAGE
-----
    [PROJECT_PATH]/$ python notebooks/04-pg-model-training.py -design                   {DESIGN NAME}
                                                              -first_hidden_layer_pbk   {BIOLOGICAL KNOWLEDGE}
                                                              -first_hidden_layer_dense {NUMBER of DENSE LAYER}
                                                              -second_hidden_layer      {SECOND HIDDEN LAYER}
                                                              -optimizer                {OPTIMIZER}
                                                              -ds                       {DATASET}
                                                              -split                    {DATASET SPLIT OPERATION}
                                                              -filtering_gene_space     {GENE SPACE FILTERING}
                                                              
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
seed(91)
import tensorflow as tf
tf.random.set_seed(91)

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import config as src

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, LeaveOneGroupOut, LeavePGroupsOut

from numba import cuda
from tensorflow import keras

# DEFAULT VALUES
n_p_leave_out =[2,4,6,8]
p_out_iteration = 20
epochs_default=100 # for model design
batch_default=10   # for model design
val_split=0.1      # for model design
rand_state = 91    # dataset split
test_size = 0.3    # train_test_split
skf_split = 5      # number of split StratifiedKFold
rskf_split = 10    # number of split for
rskf_repeat = 50   # number of iteration for RepeatedStratifiedKFold

def NN_training(design_name, bio_knowledge, dense_nodes, second_hidden_layer, optimizer, activation, dataset, split, filtering_gene_space):
    
#     printing information
    print('design_name: {0}\n, bio_knowledge: {1}\n, dense_nodes: {2}\n, second_hidden_layer: {3}\n, optimizer: {4}\n, dataset: {5}\n, split: {6}\n, filter_gene_space: {7}'.format(design_name, bio_knowledge, dense_nodes, second_hidden_layer, optimizer, dataset, split, filtering_gene_space))
    time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time() dt.datetime.now().strftime('%Y%m%d_%I%M%S%p')
        
        
    if filtering_gene_space==True and bio_knowledge!=None:
        print('INFO, design is fully connected. Cannot filter the gene space. all the genes in dataset is using!!' )
#     the output location
    loc_output = os.path.join(src.DIR_MODELS, dataset.split('/')[1], split)
    src.define_folder(loc_=loc_output)
    
    df_bio, df_dense = pd.DataFrame(), pd.DataFrame()
    split_index='' # for output text information
    how_join='left' # Filtering gene space with given bio knowledge
#     Reading dataset
    df = pd.read_pickle(os.path.join(src.DIR_DATA, dataset))  
    print('Dataset cell type         , ', df.groupby('cell_type').size().index.values)
    print('Dataset shape             , ', df.shape)
#     Creating dense layer
    if dense_nodes != 0:
        df_dense = pd.DataFrame(df.columns[:-1]).set_index(0)
        print('***** DENSE LAYER ADDED - {}!!'.format(dense_nodes))
        for i in range(dense_nodes):
            df_dense['dense'+str(i)] = 1.0
            
    # Importing all prior biological knowledge and combine all genes to create a common gene list
    if (bio_knowledge != 'None'):
        df_bio = pd.DataFrame(pd.read_csv(os.path.join(src.DIR_DATA_PROCESSED, bio_knowledge), index_col=0))
        print('Imported bio layer, ', df_bio.shape)
        
        if  filtering_gene_space==True:
            print('***** GENE SPACE FILTERED!!')
            how_join = 'right'
           
    df_bio_with_df = pd.merge(left=pd.DataFrame(df.columns[:-1]).set_index(0)
                                     , right=df_bio
                                     , left_index=True
                                     , right_index=True
                                     , how=how_join).fillna(0.0)
      
    df_first_hidden_layer = pd.merge(left=df_bio_with_df
                                     , right=df_dense
                                     , left_index=True
                                     , right_index=True
                                     , how='left').fillna(0.0)
    
#     df = df[df_first_hidden_layer.index]
    
    print('First hidden layer shape, ', df_first_hidden_layer.shape)      
    
    ohe = OneHotEncoder()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    y_ohe = ohe.fit_transform(y).toarray()
    groups = df.iloc[:, -1].values
    
    START_TRAINING = dt.datetime.now().time().strftime('%H:%M:%S')
    df_nn = pd.DataFrame()
    
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" # Stop training when `val_loss` is no longer improving
                                           , min_delta=1e-5   # "no longer improving" being defined as "no better than 1e-5 less"
                                           , patience=3       # "no longer improving" being further defined as "for at least 3 epochs"
                                           , verbose=1 ) ]
    
    df_nn = pd.DataFrame()
    
    if split == 'StratifiedKFold':
        stratified = StratifiedKFold(n_splits=skf_split
                                     , shuffle=True
                                     , random_state=rand_state)
    elif split=='RepeatedStratifiedKFold':
        stratified = RepeatedStratifiedKFold(n_splits=rskf_split
                                             , n_repeats=rskf_repeat
                                             , random_state=rand_state)
        
    elif split=='LeaveOneGroupOut':
        logo = LeaveOneGroupOut()
        for i, indexes in enumerate(logo.split(X, y, groups)):
            print(split+' --- '+str(i+1)+'/'+str(logo.get_n_splits(X, y, groups)))
            train_index=indexes[0]
            print(train_index[0])
            test_index=indexes[1]
                
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_ohe[train_index], y_ohe[test_index]

            keras.backend.clear_session()
            model = src.proposed_NN(X=X, y=y
                            , bio_layer=df_first_hidden_layer
                            , select_optimizer=optimizer
                            , select_activation=activation
                            , second_layer=second_hidden_layer)
            print(model.summary)

            model.fit(X_train, y_train
                      , epochs=epochs_default
                      , batch_size=batch_default
                      , verbose=1
                      , callbacks=callbacks
                      , validation_split=val_split)
            
            y_pred = model.predict(X_test)
            df_split = src.generate_pred_result(y_pred, y_test, ohe)
            df_split['design'] = design_name
            df_split['index_split'] = i
            df_split['split'] = split
            df_nn = pd.concat([df_nn, df_split])

            model.save(os.path.join(loc_output, 'design_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+str(i)+'_'+optimizer+'.h5'))

        df_nn.to_pickle(os.path.join(loc_output, 'result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.pck'))
        print('file is exported in ', os.path.join(loc_output, 'result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.pck'))

            
    elif split=='LeavePGroupsOut':
        for i_p_out in n_p_leave_out:
            src.define_folder(loc_=os.path.join(loc_output, 'cell_out_'+str(i_p_out)))
            lpgo = LeavePGroupsOut(n_groups=i_p_out)
            ids = np.random.choice(len(list(lpgo.split(X, y, groups))), p_out_iteration).tolist()
            lpgo_split_random_selection = [list(lpgo.split(X, y, groups))[i] for i in ids]

            for i, indexes in enumerate(lpgo_split_random_selection):
                print(split+' --- '+str(i+1)+'/'+str(p_out_iteration))
                train_index=indexes[0]
                print(train_index[:50])
                test_index=indexes[1]
            
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_ohe[train_index], y_ohe[test_index]

                keras.backend.clear_session()
                model = src.proposed_NN(X=X, y=y
                                , bio_layer=df_first_hidden_layer
                                , select_optimizer=optimizer
                                , select_activation=activation
                                , second_layer=second_hidden_layer)
                print(model.summary)

                model.fit(X_train, y_train
                          , epochs=epochs_default
                          , batch_size=batch_default
                          , verbose=1
                          , callbacks=callbacks
                          , validation_split=val_split)

                y_pred = model.predict(X_test)
                df_split = src.generate_pred_result(y_pred, y_test, ohe)
                df_split['design'] = design_name
                df_split['index_split'] = i
                df_split['split'] = split
                df_nn = pd.concat([df_nn, df_split])

                model.save(os.path.join(loc_output, 'cell_out_'+str(i_p_out), 'design_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+str(i)+'_'+optimizer+'.h5'))
            
            df_nn.to_pickle(os.path.join(loc_output, 'cell_out_'+str(i_p_out), 'result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.pck'))
            print('file is exported in ', os.path.join(loc_output, 'cell_out'+str(i_p_out), 'result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.pck'))

    elif split=='train_test_split':
        print('train_test_split split applied! Test size is, ', test_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=test_size
                                                            , shuffle=True
                                                            , random_state=rand_state
                                                            , stratify=y_ohe)
        keras.backend.clear_session()
        model = src.proposed_NN(X=X, y=y
                        , bio_layer=df_first_hidden_layer
                        , select_optimizer=optimizer
                        , select_activation=activation
                        , second_layer=second_hidden_layer)
        print(model.summary)

        model.fit(X_train, y_train
                  , epochs=epochs_default
                  , batch_size=batch_default
                  , verbose=1
                  , callbacks=callbacks
                  , validation_split=val_split)
        
        y_pred = model.predict(X_test)
        df_split = src.generate_pred_result(y_pred, y_test, ohe)
        df_split['design'] = design_name
        df_split['index_split'] = 0
        df_split['split'] = split
        df_nn = pd.concat([df_nn, df_split])
        
        model.save(os.path.join(loc_output, 'design_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.h5'))

    elif split=='None':
        print('Full dataset!!')
        split='with_full_dataset'
        
        keras.backend.clear_session()
        
        model = src.proposed_NN(X=X, y=y
                        , bio_layer=df_first_hidden_layer
                        , select_optimizer=optimizer
                        , select_activation=activation
                        , second_layer=second_hidden_layer)
        print(model.summary)
        model.fit(X, y_ohe
                  , epochs=epochs_default
                  , batch_size=batch_default
                  , verbose=1
                  , callbacks=callbacks
                  , validation_split=val_split)
        
        model.save(os.path.join(loc_output, 'design_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.h5'))

    if split =='StratifiedKFold' or split =='RepeatedStratifiedKFold':
        
        print('{0} split applied!! The number of split is {1}'.format(split, stratified.get_n_splits()))

        for i, indexes in enumerate(stratified.split(X, y)):
            print(split+' --- '+str(i+1)+'/'+str(stratified.get_n_splits()))
            train_index=indexes[0]
            test_index=indexes[1]

            print(train_index, len(train_index))

            X_train, X_test = X[train_index] , X[test_index]
            y_train, y_test = y_ohe[train_index], y_ohe[test_index]

            keras.backend.clear_session()
            model = src.proposed_NN(X=X, y=y
                        , bio_layer=df_first_hidden_layer
                        , select_optimizer=optimizer
                        , select_activation=activation
                        , second_layer=second_hidden_layer)
            print(model.summary)
            model.fit(X_train, y_train
                      , epochs=epochs_default
                      , batch_size=batch_default
                      , verbose=1
                      , callbacks=callbacks
                      , validation_split=val_split)

            y_pred = model.predict(X_test)
            df_split = src.generate_pred_result(y_pred, y_test, ohe)
            df_split['design'] = design_name
            df_split['index_split'] = i
            df_split['split'] = split
            df_nn = pd.concat([df_nn, df_split])
            
            if split =='StratifiedKFold':
                model.save(os.path.join(loc_output, 'design_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+str(i)+'_'+optimizer+'.h5'))
  

    if split!='LeavePGroupsOut' and split!='LeaveOneGroupOut':
        df_nn.to_pickle(os.path.join(loc_output,'result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.pck'))
        print('file is exported in ', os.path.join(loc_output,'result_'+design_name+'_'+dataset.split('.')[0].split('/')[-1]+'_'+optimizer+'.pck'))
    
    cuda.select_device(0)
    cuda.close()
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-design'                  , '--design_name', help='name of design')
    parser.add_argument('-first_hidden_layer_pbk'  , '--bio_knowledge', help='integrated prior biologicl knowledge')
    parser.add_argument('-first_hidden_layer_dense', '--dense_nodes', help='integrated dense node into first hidden layer')
    parser.add_argument('-second_hidden_layer'     , '--second_hidden_layer', help='second layer exist or not')
    parser.add_argument('-optimizer'               , '--optimizer', help='selecting the optimizer')
    parser.add_argument('-activation'              , '--activation', help='selecting the actibation')
    parser.add_argument('-ds'                      , '--dataset', help='the experiment dataset')
    parser.add_argument('-split'                   , '--split', help='specifying dataset split, etc, train_test_split or KFold')
    parser.add_argument('-filter_gene_space'       , '--filter_space', help='filtering gene space with given bio knowledge set')
    
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    NN_training(args.design_name
                , args.bio_knowledge
                , int(args.dense_nodes)
                , eval(args.second_hidden_layer)
                , args.optimizer
                , args.activation
                , args.dataset
                , args.split
                , eval(args.filter_space)
               )
    
