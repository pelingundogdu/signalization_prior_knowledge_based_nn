SEED = 91 # for reproducible result

import os
import sys
import glob
import random
import argparse
import traceback
import numpy as np
import pandas as pd
import datetime as dt

import tensorflow as tf
from tensorflow import keras

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter
from imblearn.datasets import make_imbalance

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit

import helper

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from scripts import config as src

# print(f'\n\nProject parent directory {ROOT_DIR}')

# def set_seeds(seed=SEED):
# #     tf.keras.backend.clear_session()
#     os.environ['HOROVOD_FUSION_THRESHOLD']='0'
#     os.environ['TF_DETERMINISTIC_OPS'] = '1'
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     tf.random.set_seed(seed)
#     np.random.seed(seed)
    
def NN_model_training(design_name# = 'circuits_signaling_1_layer'
                      , dataset_name# = 'processed/exper_pbmc/pbmc_sw_log1p.pck'
                      , bio_knowledge# = 'pbk_circuit_hsa_sig.txt'
                      , stratified_split# = 10
                      , stratified_repeat# = 2
                      , activation# = 'Relu'
                      , optimizer# = 'Adam'
                      , epochs# = 100
                      , batch_size# = [10]):
                      , second_hidden_layer=False
                     ):
    try:
        helper.set_seeds(seed=SEED)
        time_start = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time() dt.datetime.now().strftime('%Y%m%d_%I%M%S%p')
        print(f'Executing started at {time_start}')
        
        # Network design
#         epochs_default = 100       # the number of epoch
#         batch_default = 10         # the size of batch
        val_split = 0.1            # the percentage of validation split
#         optimizer='Adam'
#         activation='relu'
        filter_space='False'
#         dataset_name='processed/exper_pbmc/pbmc_sw_log1p.pck'
#         bio_knowledge='pbk_circuit_hsa_sig.txt'
#         stratified_split = 10
#         stratified_repeat = 2
#         second_hidden_layer = False
#         design_name = 'circuits_signaling_1_layer'
        split = 'RepeatedStratifiedKFold'
    
        experiment_name = dataset_name.split('/')[1]
        data_name = dataset_name.split('/')[-1].split('.')[0]
        print(experiment_name,'----', data_name)
        
        df_dense = pd.DataFrame()
        # reading dataset
        df = pd.read_pickle(os.path.join(src.DIR_DATA, dataset_name))
#         print(f'Dataset shape, {df.shape}')
        # importing prior knowledge
        df_bio = pd.DataFrame(pd.read_csv(os.path.join(src.DIR_DATA_PROCESSED, bio_knowledge), index_col=0))
#         print(f'Prior knowledge shape, {df_bio.shape}')

        # creating infromed layer
        df_bio_with_df = pd.merge(left=pd.DataFrame(df.columns[:-1]).set_index(0)
                                         , right=df_bio
                                         , left_index=True
                                         , right_index=True
                                         , how='left').fillna(0.0)

        df_first_hidden_layer = pd.merge(left=df_bio_with_df
                                         , right=df_dense
                                         , left_index=True
                                         , right_index=True
                                         , how='left').fillna(0.0)
#         df_first_hidden_layer.head()

        print('\n****SCRIPT INFORMATION****\n')
        print(f'Design name: {design_name}')
        print(f'Dataset shape, {df.shape}')
        print(f'bio_knowledge: {bio_knowledge}')
        print(f'Prior knowledge shape, {df_bio.shape}')
        print(f'Dense_nodes: {len(df_dense)}')
        print(f'First hidden layer shape, {df_first_hidden_layer.shape}')
        print(f'second_hidden_layer: {second_hidden_layer}')
        print(f'Epoch size: {epochs}')
        print(f'Batch size: {batch_size}')
        print(f'optimizer: {optimizer}')
        print(f'Activation: {activation}')
        print(f'dataset: {dataset_name}')
        print(f'split: {split}')
        print(f'filter_gene_space: {filter_space}')
        print(f'stratified_split: {stratified_split}')
        print(f'stratified_repeat: {stratified_repeat}')
        print(f'Total number of iteration is (stratified_split * stratified_repeat): {stratified_split*stratified_repeat}')

        # Loading dataset
#         ohe = OneHotEncoder()
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1:].values
#         y_category = df.iloc[:, -1].astype('category').cat.codes
#         y_ohe = ohe.fit_transform(y).toarray()
#         # Defining the order or the label via cell type value
#         order_plot = list(np.unique(pd.DataFrame(y)[0]))
#         all_label = dict(zip(order_plot, range(len(order_plot))))
#         print(f'Cell type values and the label order, {all_label}')

#         stratified = RepeatedStratifiedKFold(n_splits=stratified_split
#                                              , n_repeats=stratified_repeat
#                                             , random_state=SEED)


        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss" # Stop training when `val_loss` is no longer improving
                                                   , min_delta=1e-5   # "no longer improving" being defined as "no better than 1e-5 less"
                                                   , patience=3       # "no longer improving" being further defined as "for at least 3 epochs"
                                                   , verbose=0 ) ]

        df_result_similarity, df_result_pred_truth = pd.DataFrame(), pd.DataFrame()
        dict_threshold = dict()
        
        
        multipliers = [.2, .4, .6, .8]
#         multipliers = [.2, .4]
        print(f'multipliers, {multipliers}')
        for i_multiplier in multipliers:
            for i_cell_type in np.unique(df.iloc[:, -1:].values):#[:1]:            
                ohe = OneHotEncoder()
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1:].values
                print(f'\n\n\ncell type {i_cell_type}, multipliers {i_multiplier}\n\n')
                X, y = make_imbalance(X
                                      , y.ravel()
                                      , random_state=SEED
                                      , sampling_strategy=helper.ratio_func
                                      , **{'multiplier': i_multiplier, 'effected_class': i_cell_type},
                                      )
                y = y.reshape(-1, 1)
                print(pd.DataFrame(y).groupby(0).size())
                y_ohe = ohe.fit_transform(y).toarray()
                # Defining the order or the label via cell type value
                order_plot = list(np.unique(pd.DataFrame(y)[0]))
                all_label = dict(zip(order_plot, range(len(order_plot))))
                print(f'Cell type values and the label order, {all_label}')

                stratified = RepeatedStratifiedKFold(n_splits=stratified_split
                                                     , n_repeats=stratified_repeat
                                                    , random_state=SEED)

                for i, (train_index, test_index) in enumerate(stratified.split(X, y)):
                    print(f'\n\n\n{i+1}/{stratified_split * stratified_repeat} - training-validation-testing split!!!')
                    X_train_val, X_test = X[train_index] , X[test_index]
                    y_train_val_ohe, y_test_ohe = y_ohe[train_index], y_ohe[test_index]
                    y_train_val, y_test = y[train_index], y[test_index]
                    print('testing split completed!!')
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=SEED)
                    for j, (train_index_sss, test_index_sss) in enumerate(sss.split(X_train_val, y_train_val)):
                        print('training - validation split completed!!')
                        X_train, X_val = X_train_val[train_index_sss] , X_train_val[test_index_sss]
                        y_train_ohe, y_val_ohe = y_train_val_ohe[train_index_sss], y_train_val_ohe[test_index_sss]
                        y_train, y_val = y_train_val[train_index_sss], y_train_val[test_index_sss]

                    print(f'INFO - Training set  , X shape {X_train.shape}, y shape {y_train.shape}, y_ohe shape {y_train_ohe.shape}')
                    print(f'INFO - Validation set, X shape {X_val.shape} , y shape {y_val.shape} , y_ohe shape {y_val_ohe.shape}')
                    print(f'INFO - Testing set   , X shape {X_test.shape} , y shape {y_test.shape} , y_ohe shape {y_test_ohe.shape}\n')

                    keras.backend.clear_session()
                    helper.set_seeds(seed=SEED)
                    model = src.proposed_NN(X=X, y=y
                                            , bio_layer=df_first_hidden_layer
                                            , select_optimizer=optimizer
                                            , select_activation=activation
                                            , second_layer=second_hidden_layer)
                    print('\n\nModel summary')
                    model.summary()

                    print('INFO - Model is training!!!')
                    model.fit(X_train, y_train_ohe          
                              , epochs=epochs
                              , batch_size=batch_size
                              , verbose=2
                              , callbacks=callbacks
                              , validation_data=(X_val, y_val_ohe)
                             )
                    #     pd.DataFrame(model.history.history)[['loss','val_loss']].plot(figsize=(12,6))
                    print('INFO - Model training is completed!!')

                    model_encoding = keras.models.Model(inputs=model.layers[0].input
                                                    , outputs=model.layers[-1].input)
                    print('Model - encoding summary\n\n')
                    model_encoding.summary()

                    ############################### LOCAL OUTLIER FACTOR ###############################
                    # getting encoding information
                    encoding_training = model_encoding.predict(X_train)
                    encoding_validation = model_encoding.predict(X_val)
                    np.random.seed(SEED)

                    # calculating similarity score
                    lof = LocalOutlierFactor(novelty=True)
                    lof.fit(encoding_training)

                    df_similarity_training = pd.concat([ pd.DataFrame(y_train, columns=['cell_type'])
                                                        , pd.DataFrame(lof.score_samples(encoding_training), columns=['score'])], axis=1)
                    df_similarity_validation = pd.concat([ pd.DataFrame(y_val, columns=['cell_type'])
                                                          , pd.DataFrame(lof.score_samples(encoding_validation), columns=['score'])], axis=1)

#                     df_similarity_training['cell_multiplier'] = i_cell_type
#                     df_similarity_training['ratio'] = i_multiplier

#                     df_similarity_validation['cell_multiplier'] = i_cell_type
#                     df_similarity_validation['ratio'] = i_multiplier

                    print(f'Encoding training data shape, {encoding_training.shape}')
                    print(f'Encoding validation data shape, {encoding_validation.shape}')
                    print(f'Similarity training shape, {df_similarity_training.shape}')
                    print(f'Similarity validation shape, {df_similarity_validation.shape}')
                    # Calculated threshold value
                    threshold = np.mean(df_similarity_training.groupby('cell_type').aggregate(['mean', 'std'])['score']['mean'] 
                                        - 1.5*df_similarity_training.groupby('cell_type').aggregate(['mean', 'std'])['score']['std'])
                    print('Threshold value from reference dataset, ', threshold)
                    dict_threshold[f'{i}_{i_cell_type}_{i_multiplier}'] = threshold
        #             df_threshold_score = pd.DataFrame(threshold , columns=['threshold'])
        #             df_threshold_score['index_split'] = i
        #             df_result_threshold_score = pd.concat([df_result_threshold_score, df_threshold_score])

                    df_similarity_training['source']='training'
                    df_similarity_validation['source']='validation'
                    df_similarity = pd.concat([df_similarity_training, df_similarity_validation], axis=0).reset_index(drop=True)
                    df_similarity['index_split'] = i #index_split
                    df_similarity['multiplier_cell'] = i_cell_type
                    df_similarity['multiplier_ratio'] = i_multiplier
                    df_result_similarity = pd.concat([df_result_similarity, df_similarity])
                    print(f'INFO - Similarity score for training-validation added into result dataset!! - iteration {i}')

                    print('INFO - LOF and prediction values are calculating for testing dataset')
                    prediction= model_encoding.predict(X_test)

                    df_prediction = pd.DataFrame(model.predict(X_test), columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
                    df_prediction['index_split'] = i
                    df_prediction = pd.concat([df_prediction, pd.DataFrame(y_test, columns=['cell_type'])
                                            , pd.DataFrame(lof.score_samples(prediction), columns=['score'])], axis=1)

                    df_prediction['pred'] = pd.DataFrame([ohe.categories_[0][i] for i in np.argmax(model.predict(X_test), axis=-1)])
                    df_prediction['threshold'] = 'above'
                    df_prediction['multiplier_cell'] = i_cell_type
                    df_prediction['multiplier_ratio'] = i_multiplier
                    df_prediction['threshold_score'] = threshold

                    df_below_threshold = np.where(df_prediction['score'] <= threshold)[0]
                    df_prediction.loc[df_prediction.index.isin(df_below_threshold), 'pred'] = 'unassigned'
                    df_prediction.loc[df_prediction.index.isin(df_below_threshold), 'threshold'] = 'below'

                    print(f'Threshold statistics - iteration {i}')
                    print(f'Total of the unassigned samples, {len(df_below_threshold)}')
                    print(df_prediction.groupby('threshold').size() / len(df_prediction), df_prediction.groupby('threshold').size())

                    df_result_pred_truth = pd.concat([df_result_pred_truth, df_prediction])
                    print(f'INFO - Prediction, LOF, ground truth values are added into result dataset!! - iteration {i}\n')
        
        df_result_threshold_score = pd.DataFrame(dict_threshold.items(), index=dict_threshold.keys(), columns=['index_split', 'threshold'])
        df_result_pred_truth['design_name'] = design_name
        df_result_similarity['design_name'] = design_name
        df_result_threshold_score['design_name'] = design_name

        df_result_pred_truth.reset_index(drop=True, inplace=True)
        df_result_similarity.reset_index(drop=True, inplace=True)
        df_result_threshold_score.reset_index(drop=True, inplace=True)

        #         df_result_pred_truth.to_csv(f'./experiment_{dataset_name}_testing_result_{stratified_split * stratified_repeat}.csv')
        #         df_result_similarity.to_csv(f'./experiment_{dataset_name}_lof_training_val_result_{stratified_split * stratified_repeat}.csv')
        #         df_result_threshold_score.to_csv((f'./experiment_{dataset_name}_lof_training_val_threshold_scores.csv'))
        print(df_result_pred_truth.head())
        print(df_result_similarity.head())
        print(df_result_threshold_score.head())
        
        df_result_pred_truth.to_parquet(f'./paper2_figures_for_Review/report/fake_experiment_{data_name}_testing_result_{stratified_split * stratified_repeat}_shl{second_hidden_layer}.parquet')
        df_result_similarity.to_parquet(f'./paper2_figures_for_Review/report/fake_experiment_{data_name}_lof_training_val_result_{stratified_split * stratified_repeat}_shl{second_hidden_layer}.parquet')
        df_result_threshold_score.to_parquet((f'./paper2_figures_for_Review/report/fake_experiment_{data_name}_lof_training_val_threshold_scores_{stratified_split * stratified_repeat}_shl{second_hidden_layer}.parquet'))
        
        df_result_pred_truth.to_csv(f'./paper2_figures_for_Review/report/fake_experiment_{data_name}_testing_result_{stratified_split * stratified_repeat}_shl{second_hidden_layer}.csv')
        df_result_similarity.to_csv(f'./paper2_figures_for_Review/report/fake_experiment_{data_name}_lof_training_val_result_{stratified_split * stratified_repeat}_shl{second_hidden_layer}.csv')
        df_result_threshold_score.to_csv((f'./paper2_figures_for_Review/report/fake_experiment_{data_name}_lof_training_val_threshold_scores_{stratified_split * stratified_repeat}_shl{second_hidden_layer}.csv'))
        
        print('INFO - RESULTS ARE EXPORTED!!')
        
        time_end = dt.datetime.now().time().strftime('%H:%M:%S') # = time.time() dt.datetime.now().strftime('%Y%m%d_%I%M%S%p')
        print(f'Executing started at {time_start}')
        print(f'Executing finished! {time_end}')

    except ValueError as e:
        print(e)
        
    except UnboundLocalError as e:
        print(e)
        
    except TypeError as e:
        print(e)
        
    except AttributeError:
        traceback.print_exc()
        
    except NameError:
        traceback.print_exc()
        
    except KeyError:
        traceback.print_exc()

    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        
        
        
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-design_name'           , '--design_name', help='design_name')
    parser.add_argument('-dataset_name'          , '--dataset_name', help='dataset name, [string]')
    parser.add_argument('-bio_knowledge'         , '--bio_knowledge', help='prior biolgical knowledge detail, [string]')
    parser.add_argument('-stratified_split'      , '--stratified_split', help='K-fold size, [integer]')
    parser.add_argument('-stratified_repeat'     , '--stratified_repeat', help='iteration value, [integer]')
    parser.add_argument('-activation'            , '--activation', help='activation value, [string]')
    parser.add_argument('-optimizer'             , '--optimizer', help='optimizer value, [string]')
    parser.add_argument('-epochs'                , '--epochs', help='epoch size, [integer]')
    parser.add_argument('-batch_size'            , '--batch_size', help='batch size value, [integer]')
    parser.add_argument('-second_hidden_layer'   , '--second_hidden_layer', help='second layer, [integer]')
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    NN_model_training(args.design_name
                      , args.dataset_name
                      , args.bio_knowledge
                      , int(args.stratified_split)
                      , int(args.stratified_repeat)
                      , args.activation
                      , args.optimizer
                      , int(args.epochs)
                      , int(args.batch_size)
                      , eval(args.second_hidden_layer)
                     )