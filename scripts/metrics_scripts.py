#!/usr/bin/env python
# coding: utf-8

# Required libraries
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score

def calculate_clustering_metrics(dataframe):
    list_homo, list_comp, list_vmes, list_ari, list_ami, list_fm, list_acc, list_mean = [],[],[],[],[],[],[],[]
    for i_cell_out in dataframe['cell_out'].unique():
        for i_design in dataframe['design'].unique():
            for i_exp in dataframe['index_split'].unique():
                df_temp = dataframe[(dataframe['design'] == i_design ) 
                                    & (dataframe['index_split'] == i_exp) 
                                    & (dataframe['cell_out'] == i_cell_out)]
#                 print('cell_out -{0}\n desing {1}\n index_split {2}\n len {3}'.format(i_cell_out, i_design, i_exp, len(df_temp)))
                list_homo.append([ homogeneity_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, '1-homogeneity', i_design, i_cell_out])
                list_comp.append([ completeness_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, '2-completeness', i_design, i_cell_out])
                list_vmes.append([ v_measure_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, '3-v_measure', i_design, i_cell_out])
                list_ari.append([ adjusted_rand_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, '4-ari', i_design, i_cell_out])
                list_ami.append([ adjusted_mutual_info_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, '5-ami', i_design, i_cell_out])
                list_fm.append([ fowlkes_mallows_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, '6-fowlkes_mallows', i_design, i_cell_out])
                list_mean.append([ np.mean( [homogeneity_score(df_temp['ground_truth'], df_temp['prediction'])
                                        , completeness_score(df_temp['ground_truth'], df_temp['prediction'])
                                        , v_measure_score(df_temp['ground_truth'], df_temp['prediction'])
                                        , adjusted_rand_score(df_temp['ground_truth'], df_temp['prediction'])
                                        , adjusted_mutual_info_score(df_temp['ground_truth'], df_temp['prediction'])
                                        , fowlkes_mallows_score(df_temp['ground_truth'], df_temp['prediction'])]), i_exp, '7-mean', i_design, i_cell_out])

        result = [element for lis in [list_homo, list_comp, list_vmes, list_ari, list_ami, list_fm, list_mean] for element in lis]
        print(len(result))
        df_metric = pd.DataFrame(result, columns=['score','expr','metric','design','cell_out'])
    return(df_metric)