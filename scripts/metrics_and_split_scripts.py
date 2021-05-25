#!/usr/bin/env python
# coding: utf-8

# Required libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, train_test_split, LeaveOneGroupOut, LeavePGroupsOut
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

def calculate_clustering_metrics(dataframe):
    print('calculate_clustering_metrics')
    list_homo, list_comp, list_vmes, list_ari, list_ami, list_fm, list_acc, list_mean = [],[],[],[],[],[],[],[]
    for i_cell_out in dataframe['cell_out'].unique():
        for i_exp in dataframe['index_split'].unique():
            df_temp = dataframe[(dataframe['index_split'] == i_exp) 
                                & (dataframe['cell_out'] == i_cell_out)]
#                 print('cell_out -{0}\n desing {1}\n index_split {2}\n len {3}'.format(i_cell_out, i_design, i_exp, len(df_temp)))
            list_homo.append([ homogeneity_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'homogeneity', i_cell_out])
            list_comp.append([ completeness_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'completeness', i_cell_out])
            list_vmes.append([ v_measure_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'v_measure', i_cell_out])
            list_ari.append([ adjusted_rand_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'ari', i_cell_out])
            list_ami.append([ adjusted_mutual_info_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'ami', i_cell_out])
            list_fm.append([ fowlkes_mallows_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'fowlkes_mallows', i_cell_out])
            list_mean.append([ np.mean( [homogeneity_score(df_temp['ground_truth'], df_temp['prediction'])
                                    , completeness_score(df_temp['ground_truth'], df_temp['prediction'])
                                    , v_measure_score(df_temp['ground_truth'], df_temp['prediction'])
                                    , adjusted_rand_score(df_temp['ground_truth'], df_temp['prediction'])
                                    , adjusted_mutual_info_score(df_temp['ground_truth'], df_temp['prediction'])
                                    , fowlkes_mallows_score(df_temp['ground_truth'], df_temp['prediction'])]), i_exp, 'mean', i_cell_out])

    result = [element for lis in [list_homo, list_comp, list_vmes, list_ari, list_ami, list_fm, list_mean] for element in lis]
    print(len(result))
    df_metric = pd.DataFrame(result, columns=['score','index_split','metric','cell_out'])
    return(df_metric)


# def generate_pred_result(y_pred, y_test, ohe):
    
#     df_split = pd.DataFrame(y_pred, columns=list(pd.DataFrame(ohe.categories_).iloc[0,:]))
#     df_split['prediction'] = ohe.inverse_transform(y_pred).reshape(1, -1)[0]
#     df_split['ground_truth'] = ohe.inverse_transform(y_test).reshape(1, -1)[0]

#     return df_split

def calculate_f1_recall_precision_metrics_overall(dataframe):
    print('calculate_f1_recall_precision_metrics_overall')
    list_f1, list_precision, list_recall, list_acc, list_bacc = [],[],[],[],[]

    for i_exp in dataframe['index_split'].unique():
        df_temp = dataframe[dataframe['index_split']==i_exp]

        for i_average in ['micro','macro','weighted']:
            list_f1.append([f1_score(df_temp['ground_truth'], df_temp['prediction'], average=i_average), i_exp, 'f1-'+i_average])
            list_precision.append([precision_score(df_temp['ground_truth'], df_temp['prediction'], average=i_average), i_exp, 'precision-'+i_average])
            list_recall.append([recall_score(df_temp['ground_truth'], df_temp['prediction'], average=i_average), i_exp, 'recall-'+i_average])
        list_acc.append([accuracy_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'accuracy'])
        list_bacc.append([balanced_accuracy_score(df_temp['ground_truth'], df_temp['prediction']), i_exp, 'balanced accuracy'])

    result = [element for lis in [list_f1, list_precision, list_recall, list_acc, list_bacc] for element in lis]

    df_metric = pd.DataFrame(result, columns=['score','index_split','metric'])
    return(df_metric)
   

def calculate_f1_recall_precision_metrics_cell_type_detail(dataframe):
    print('calculate_f1_recall_precision_metrics_cell_type_detail')
    list_f1, list_precision, list_recall= [],[],[]
    cell_types = sorted(dataframe['ground_truth'].unique())
    cell_types.extend(['index_split','metric'])
    for i_exp in dataframe['index_split'].unique():
        df_temp = dataframe[dataframe['index_split']==i_exp]
        list_f1.append(list(f1_score(df_temp['ground_truth'], df_temp['prediction'], average=None)) + [i_exp, 'f1'])
        list_precision.append(list(precision_score(df_temp['ground_truth'], df_temp['prediction'], average=None )) + [i_exp, 'precision'])
        list_recall.append(list(recall_score(df_temp['ground_truth'], df_temp['prediction'], average=None )) + [i_exp, 'recall'])
# , labels=np.unique(df_temp['prediction']
        result = [element for lis in [list_f1, list_precision, list_recall] for element in lis]

    df_metric = pd.DataFrame(result, columns=cell_types)
    df_label_melt = pd.melt(frame=df_metric, id_vars=['index_split','metric'])
    return(df_label_melt)

def generate_training_testing_samples(X, y, y_ohe, y_category, groups, SEED
                                      , split
                                      , stratified_split, stratified_repeat
                                      , n_p_leave_out, p_out_iteration
                                      , test_size
                                      , export_to_text
                                      , train_test_repeat):

#         Creating empty training and testing list, will use in following steps to store the index values of splits
    X_train_list, y_train_list, X_test_list, y_test_list, split_index_list, co_list = [], [], [], [], [], []
    
    if split =='StratifiedKFold' or split =='RepeatedStratifiedKFold':
        if split =='StratifiedKFold' :
            stratified_repeat = 1
            
        stratified = RepeatedStratifiedKFold(n_splits=stratified_split
                                             , n_repeats=stratified_repeat
                                             , random_state=SEED)

        export_to_text.save(text=f'{split} split applied!! The number of split is {stratified_split}, and number of repeat is {stratified_repeat}. The total iteration is {stratified_split * stratified_repeat}')
        print(f'{split} split applied!! The number of split is {stratified_split}, and number of repeat is {stratified_repeat}. The total iteration is {stratified_split * stratified_repeat}')

        for i, indexes in enumerate(stratified.split(X, y)):
            train_index=indexes[0]
            test_index=indexes[1]
            print(f'{i+1}/{stratified_split * stratified_repeat}')
            export_to_text.save(text=f'{i+1}/{stratified_split * stratified_repeat}\ntest_index[:15]; {test_index[:15]}\nlen(test_index); {len(test_index)}')

            X_train, X_test = X[train_index] , X[test_index]
            y_train, y_test = y_ohe[train_index], y_ohe[test_index]

            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            split_index_list.append(i)

            del(X_train, X_test, y_train, y_test, i)

    elif split =='LeaveOneGroupOut' or split =='LeavePGroupsOut':
        if split =='LeaveOneGroupOut':
            n_p_leave_out = [1]

        export_to_text.save(text=f'{split} split applied!! split will be perform for leaving {n_p_leave_out} group(s) out!!')
        print(f'{split} split applied!! split will be perform for leaving {n_p_leave_out} group(s) out!!')

        for i_p_out in n_p_leave_out:
            lgo = LeavePGroupsOut(n_groups=i_p_out)
            if len(list(lgo.split(X, y, groups))) < p_out_iteration:
                print(f'WARNING - The total number of combination ({len(list(lgo.split(X, y, groups)))}) when leaving {i_p_out} cell out is less than given iteration ({p_out_iteration}) number!!')
                n_iteration = len(list(lgo.split(X, y, groups)))
            else :
                n_iteration = p_out_iteration
            
            ids = np.random.choice(len(list(lgo.split(X, y, groups))), n_iteration, replace=False).tolist()
            lgo_split_random_selection = [list(lgo.split(X, y, groups))[i] for i in ids]
            for i, indexes in enumerate(lgo_split_random_selection):
                train_index=indexes[0]
                test_index=indexes[1]
                print(f'{i+1}/{n_iteration} -- cell_out_{i_p_out}')
                export_to_text.save(text=f'{i+1}/{n_iteration} -- cell_out_{i_p_out}\ntest_index[:15]; {test_index[:15]}\nlen(test_index); {len(test_index)}')

                X_train, X_test = X[train_index], X[test_index]
                y_train = y_ohe[train_index]
                if split =='LeaveOneGroupOut':
                    y_test = y_ohe[test_index]
                else:
                    y_test = y_category[test_index]

                X_train_list.append(X_train)
                X_test_list.append(X_test)
                y_train_list.append(y_train)
                y_test_list.append(y_test)
                co_list.append(i_p_out)
                split_index_list.append(i)

        del(X_train, X_test, y_train, y_test, i_p_out, i)

    elif split =='train_test_split':
        export_to_text.save(text=f'{split} split applied!! Test size is {test_size} and the iteration is {train_test_repeat}!!')
        print(f'{split} split applied!! Test size is {test_size} and the iteration is {train_test_repeat}!!')
        for i in range(train_test_repeat):
            print(f'train_test_split iteration {i+1} / {train_test_repeat}')
            X_train, X_test, y_train, y_test = train_test_split(X, y_ohe
                                                                , test_size=test_size
                                                                , shuffle=True
                                                                , random_state=SEED+i
                                                                , stratify=y_ohe)
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
            split_index_list.append(i)

            del(X_train, X_test, y_train, y_test)

    else:
        export_to_text.save(text=f'{split} split applied!! Full dataset will use in fitting step!!')
        print(f'{split} split applied!! Full dataset will use in fitting step!!')
        X_train_list.append(X)
        y_train_list.append(y_ohe)
        

    return (X_train_list, y_train_list, X_test_list, y_test_list, split_index_list, co_list)


def calculate_threshold(encoding_with_seen, encoding_with_unseen, y_with_seen, y_with_unseen):
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(encoding_with_seen)

    df_score_unseen = pd.concat([ y_with_unseen, pd.DataFrame(lof.score_samples(encoding_with_unseen), columns=['score'])], axis=1)
    df_score_seen = pd.concat([ y_with_seen, pd.DataFrame(lof.score_samples(encoding_with_seen), columns=['score'])], axis=1)

    # Calculated threshold value
    threshold = np.mean(df_score_seen.groupby('cell_type').aggregate(['mean', 'std'])['score']['mean'] 
                        - 0.5*df_score_seen.groupby('cell_type').aggregate(['mean', 'std'])['score']['std'])
    print('Threshold value from reference dataset, ', threshold)
    return threshold, df_score_seen, df_score_unseen