#!/usr/bin/env python

import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

# from scripts import config as src
import pandas as pd
import numpy as np
from scipy.stats import rankdata
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain
import warnings
warnings.filterwarnings('ignore')

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rand_state = 91
dataset              = 'immune_new.pck'
experiment           = 'exper_immune'
analysis             = 'activation'
split                = 'train_test_split_tuningFalse_filtergeneFalse'
model_detail         = 'circuits_1_layer'
bio_knowledge        = 'pbk_circuit_hsa_sig.txt'
# model_detail         = 'pathways_1_layer'
# bio_knowledge        = 'pbk_layer_hsa_sig.txt'

df = pd.read_pickle(os.path.join('./data/processed/', experiment, dataset))
# df = df.sample(n=20, replace=True, random_state=1) # sampling
ohe = OneHotEncoder()
X_data = df.iloc[:, :-1].values
y_data = df.iloc[:, -1:].values
y_ohe = ohe.fit_transform(y_data).toarray()
print(X_data.shape)
print(y_data.shape)

print(df.groupby('cell_type').size())

df_pbk = pd.read_csv(f'./data/processed/{bio_knowledge}').set_index('symbol')

df_pbk_final = pd.merge(left=pd.DataFrame(df.columns[:-1]).set_index(0)
                        , right=df_pbk
                        , left_index=True
                        , right_index=True
                        , how='left').fillna(0)
# df_pbk_final = df_pbk_final.iloc[:, :10]
df_pbk_final

        
for experiment_index in range(0,100):
    sess = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    print(f'iteration no, {experiment_index}')
    model_name =f'design_{model_detail}_immune_new_Adam_relu_{experiment_index}.h5'
    with sess.graph.as_default():

#         tf.compat.v1.keras.backend.set_session(sess)
        tf1.keras.backend.set_session(sess)

        model = tf.compat.v1.keras.models.load_model(filepath=f'./models/{experiment}/{split}/{model_name}')

        model.summary()
        # predictions = model.predict(X_data)
        time_start = dt.datetime.now().time().strftime('%H:%M:%S') 
        print(f'start time, {time_start}')
        
        circuit_gene_index = dict()
        method_name = 'elrp'

        with DeepExplain(session=tf1.keras.backend.get_session() ) as de:
        #     defining the input size which is the number of gene in given dataset, input_tensor--> X
            X = model.layers[0].input
        #     the target layer size which shows the biological layer, target_tensor --> T
            T = model.layers[1].input
        #     the mask for the filtering the target node in for analysis
            mask_ys = np.zeros((1, T.shape[1]))
            xs = X_data
            print(f'input_tensor, {X}\ntarget_tensor, {T}\nmask_ys, {mask_ys.shape}\nxs, {xs.shape}')

        #     iterating the filtering mask for each node in the biological layer which defined as T
            for i_circuit in range(T.shape[1]):
                node_name = df_pbk_final.columns[i_circuit]
                print(i_circuit, node_name)
                if i_circuit % 500 == 0:
                    print(i_circuit)
        #         filtering the target node in target layer with assigning value as 1
                mask_ys[:, i_circuit] = 1.0
        #         compute attributions of each genes ('X') for each sample defined as 'xs' with defined mask 'T * mask_ys'
                explain = de.explain(method_name, T * mask_ys, X, xs)
            
        #         concat cell type with explain matrix
                df_explain = pd.concat([pd.DataFrame(explain, columns=df.columns[:-1]), pd.DataFrame(y_data, columns=['cell_type']) ], axis=1)
        #         The mean score of each cell type
                df_explain_cell_type_mean = df_explain.groupby('cell_type').mean()
        #         Converting dataframe  updating index and column value with cell type and gene names, respectively, 
                df_explain_rankdata = pd.DataFrame(rankdata(df_explain_cell_type_mean, axis=1, method="min").astype('float32')
                                   , columns=df_explain_cell_type_mean.columns
                                   , index=df_explain_cell_type_mean.index)
        #         Exporting ranked order of eadc genes for each cell type
        #         The order is lowest to highest (0 -> lowest)
                df_explain_rankdata.to_csv(f'./reports/deepexplain/cell_type_summary_with_rank/{model_detail}_{method_name}_{experiment_index}_{node_name}.csv.gzip'
                                           , compression='gzip')
            
        time_end = dt.datetime.now().time().strftime('%H:%M:%S') 
        print(f'end time, {time_end}')
        