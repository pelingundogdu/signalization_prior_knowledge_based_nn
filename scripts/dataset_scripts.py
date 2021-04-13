#!/usr/bin/env python
# coding: utf-8

# Required libraries
# import os
# import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import *

class ExperimentDataSet:
    '''
    Data uploading and preprocessing
    
    Parameters
    ----------
    experiment_dataset : str
        The reading operation for given dataset.
        ex. pd.read_csv/pd.read_pck/...('{LOCATION OF SELECTED DATASET}')
    
    target_col_index : str
        The column number of target features, it may differ depending on the using dataset. (-1 is for last column)
    
    filter_gene : list, default=None
        The common gene list to filter the experiment dataset. If it is None, the all genes in dataset are keeping.
    
    scaler : str
        The definition of preferred mathematical function or scaler algorithm.
    '''
    
    def __init__(self, experiment_dataset, target_col_index, filter_gene=None, scaler=None):
        
        self.experiment_dataset = experiment_dataset
        self.target_col_index = target_col_index
        self.filter_gene = filter_gene
        self.scaler = scaler
    
    def get_experiment_dataset(self):#(target_feature):
        '''
        Reading required dataset and applying lowercase opation for gene names.
        
        Parameters
        ----------
        experiment_dataset : str
            The reading operation for given dataframe.
        
        Returns
        -------
        experiment_df : dataframe
            The experiment dataset with lower gene names
        '''
        
        experiment_df = pd.DataFrame(self.experiment_dataset)
        experiment_df = self.__dataset_gene_name_lower(experiment_df)
        
        print('\n********* RAW DATASET *********')
        print('experiment dataset shape     , {0}\n'.format(experiment_df.shape))

        return(experiment_df)
    
    def __dataset_scaler(self, dataset):
        '''
        Applying scaler or required mathematical function into dataset
        
        Parameters
        ----------
        dataset : dataframe
        
        Returns
        -------
        df_scaler : dataframe
            Scaled dataset
        '''
        print('   -> {0} operation implemented!'.format(self.scaler))
        scaler = self.scaler
        df_scaler = pd.DataFrame()
        df_scaler['cell_type'] = dataset['cell_type'].reset_index(drop=True)
        df_scaler=pd.concat([pd.DataFrame(scaler.fit_transform(dataset.iloc[: , :-1])
                                          , columns=dataset.columns[:-1]).reset_index(drop=True), df_scaler], axis=1)
        return(df_scaler)
    
    def __dataset_filter_genes(self, dataset):
        '''
        Dataset filtering based on given gene list.
        
        Parameters
        ----------
        dataset : dataframe
        
        Returns
        -------
        dataset : dataframe
            Filtered dataset
        '''
        
        list_gene = self.filter_gene.copy()
        list_gene.extend(['cell_type'])
        print('len', len(list_gene))
        dataset = dataset.iloc[:, dataset.columns.isin(list_gene)]
        print('   -> Common genes are filtered! {0}'.format(len(list_gene)))
        return(dataset, list_gene)
    
    def __dataset_gene_name_lower(self, dataset):
        '''
        Dataframe operation. Renaming and lower case operation for columns in dataset.
        
        Parameters
        ----------
        dataset : dataframe
        
        Returns
        -------
        dataset : dataframe
            Processed dataset
        '''
#         pd.DataFrame(loc_ref)[[pd.DataFrame(loc_ref).columns[-1]]].columns[0]
        
        dataset = dataset.rename(columns={dataset[[dataset.columns[self.target_col_index]]].columns[0]:'cell_type'})
        dataset.columns = dataset.columns.str.lower()
        return(dataset)
    
    def __remove_empty_nodes(self, dataset):
        '''
        Removing if there is NO information in gene.
        
        Parameters
        ----------
        dataset : dataframe
        
        Returns
        -------
        dataset : dataframe
            Processed dataset
        
        NOTE. The biological information will be defining as neuron in this project. If dataset filtered based on given gene list, 
        this means that each neurons should be connected with at least one gene.
        
        '''
        print('   -> There is no information for {0} genes in dataset.'.format(len(dataset.sum()[dataset.sum() == 0].index)))
        dataset.drop(columns=(dataset.sum()[dataset.sum() == 0].index), inplace=True)        
        return dataset
    
    
    def run(self):
        '''
        Applying the data processes based on the parameters in __init__.
        
        Returns
        -------
        df_experiment : dataframe
            Uploaded and proprocessed dataset
        list_gene : dataframe
            the gene list which used for filtering dataset
        '''
        df_experiment = self.get_experiment_dataset()
        
        if (self.filter_gene != None):
            df_experiment, gene_list = self.__dataset_filter_genes(df_experiment)
        
#         df_experiment = self.__remove_empty_nodes(df_experiment)
        
        if (self.scaler != None):
            df_experiment = self.__dataset_scaler(df_experiment)
        
        print('\n****** PROCESSED DATASET ******')
        print('experiment dataset shape     , {0}\n'.format(df_experiment.shape))
        
        return(df_experiment)


#     if __name__ == "__main__":
#         dataset = ExperimentDataSet()
#         dataset.run()


def sample_wise_normalization(dataset, target_col_index):
    '''
    Applying scaler or required mathematical function into dataset

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


def scaler_normalization(dataset, scaler, target_col_index):
    '''
    Applying scaler or required mathematical function into dataset

    Parameters
    ----------
    dataset : dataframe

    Returns
    -------
    df_scaler : dataframe
        Scaled dataset
    '''
    print('    -> Normalization implemented! -- {0}'.format(scaler))
#     df_scaler = pd.DataFrame()
#     df_scaler['cell_type'] = dataset['cell_type'].reset_index(drop=True)
#     df_scaler=pd.concat([pd.DataFrame(scaler.fit_transform(dataset.iloc[: , :target_col_index])
#                                       , columns=dataset.columns[:target_col_index]).reset_index(drop=True), df_scaler], axis=1)
    df_scaler = pd.concat([pd.DataFrame(scaler.fit_transform(dataset.iloc[: , :target_col_index]) , columns=dataset.columns[:target_col_index]) 
                           ,dataset.iloc[:, target_col_index]], axis=1)
    
    return(df_scaler)