#!/usr/bin/env python
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

# from scripts import config as src
import pandas as pd
import numpy as np

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

rand_state = 91
dataset              = 'immune_new.pck'
experiment           = 'exper_immune'
analysis             = 'activation'
split='train_test_split_tuningFalse_filtergeneFalse'
model_detail = 'circuits_with_weights_2_layer'
bio_knowledge        = 'pbk_circuit_hsa_sig.txt'

df = pd.read_pickle(os.path.join('./data/processed/', experiment, dataset))
# df = df.sample(n=20, replace=True, random_state=1) # sampling
ohe = OneHotEncoder()
X_data = df.iloc[:, :-1].values
y_data = df.iloc[:, -1:].values
y_ohe = ohe.fit_transform(y_data).toarray()
# groups = y.reshape(1,-1)[0]
# y_cat = df['cell_type'].astype('category').cat.codes
print(X_data.shape)
print(y_data.shape)

print(df.groupby('cell_type').size())
X_data

df_pbk = pd.read_csv(f'./data/processed/{bio_knowledge}').set_index('symbol')

df_pbk_final = pd.merge(left=pd.DataFrame(df.columns[:-1]).set_index(0)
                        , right=df_pbk
                        , left_index=True
                        , right_index=True
                        , how='left').fillna(0)
# df_pbk_final = df_pbk_final.iloc[:, :10]
print(df_pbk_final)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    
    for experiment_index in range(0,101):
        print(f'iteration no, {experiment_index}')
        
        with tf1.Session(graph=tf1.Graph()) as sess:
            model_name =f'design_{model_detail}_immune_new_Adam_relu_{experiment_index}.h5'
            model = tf1.keras.models.load_model(filepath=f'./models/{experiment}/{split}/{model_name}')
            print(model.summary())
            
        #     model.summary()
            # predictions = model.predict(X_data)
            time_start = dt.datetime.now().time().strftime('%H:%M:%S') 
            print(f'start time, {time_start}')
            df_circuits = []
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
                    if i_circuit % 500 == 0:
                        print(f'iteration{i_circuit} - {dt.datetime.now().time().strftime("%H:%M:%S") }')
                        print(i_circuit)
            #         filtering the target node in target layer with assigning value as 1
                    mask_ys[:, i_circuit] = 1.0
            #         compute attributions of each genes ('X') for each sample defined as 'xs' with defined mask 'T * mask_ys'
                    explain = de.explain(method_name, T * mask_ys, X, xs)
            #         getting the mean score of full dataset ( by given in xs) of each genes for filtered target node 
                    df_circuits.append(np.mean(explain, axis=0))

            #         additional information
            #         the gene index information, it shows which genes are included for selected circuit
                    gene_index = [i for i,x in enumerate(df_pbk_final.iloc[:, i_circuit].values) if x == 1]
                    circuit_gene_index.update({i_circuit : gene_index})

            time_end = dt.datetime.now().time().strftime('%H:%M:%S') 
            print(f'end time, {time_end}')
            pd.DataFrame(df_circuits, columns=df.columns[:-1], index=df_pbk_final.columns).to_csv(f'./reports/deepexplain/{experiment}/{model_detail}_{method_name}_{experiment_index}.csv')
            pd.DataFrame(df_circuits, columns=df.columns[:-1], index=df_pbk_final.columns).to_pickle(f'./reports/deepexplain/{experiment}/{model_detail}_{method_name}_{experiment_index}.pck')
