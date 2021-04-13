#!/usr/bin/env python

'''
DESCRIPTION
-----------
    Preprocessing raw dataset -> applying scaling, filtering genes.
    
USAGE
-----
    [PROJECT_PATH]/$ python notebooks/3.0-pg-preprocessing-experiment-dataset.py -exp {EXPERIMENT}
                                                                                 -loc {DATASET LOCATION}
                                                                                 -ds  {DATASET} 
                                                                                 -pbk {BIOLOGICAL KNOWLEDGE} 
                                                                                 -sc  {SCALER} 
                                                                                 -tci {TARGET COLUMN INDEX}

RETURN
------
    {DATASET}_{SCALE}.pck : pck file
        Preprocessed file

EXPORTED FILE(s) LOCATION
-------------------------
    ./data/processed/{EXPERIMENT}/{DATASET}_{SCALE}.pck
'''

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import settings as ssrp, dataset_scripts as dsrp, path_scripts as psrp, model_scripts as msrp
import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import pyreadr # imported to read .rds files

def preprocessing_dataset(experiment, location, dataset, bio_knowledge, scale, target_column_index):

    # Importing all prior biological knowledge and combine all genes to create a common gene list
    list_gene = None
    if (bio_knowledge!=None):
        df_bio = pd.DataFrame(pd.read_csv(os.path.join(ssrp.DIR_DATA_PROCESSED, bio_knowledge)))
        list_gene = list(df_bio['symbol'])
    
    if location == 'external':
        loc_ds = ssrp.DIR_DATA_EXTERNAL
    elif location == 'processed':
        loc_ds = ssrp.DIR_DATA_PROCESSED
    else:
        print('please give a valid location!!!')
    # the output location
    loc_output = os.path.join(ssrp.DIR_DATA_PROCESSED, experiment)
    psrp.define_folder(loc_=loc_output)
    
    if scale == None:
        sc_text = 'no_scale'
    else:
        # getting scaling information to include into name of the file
        sc_text = str(type(scale)).split("'")[1].split(".")[-1]
        for k, i in scale.get_params().items():
            if k=='func':
                sc_text = scale.func.__name__

    file_name_output = '.'.join(dataset.split('.')[:-1])+'_'+sc_text+'.pck'
    print('FILE FORMAT, ', dataset.split('.')[-1])
    
    if dataset.split('.')[-1]=='rds':
        df_raw = pyreadr.read_r(os.path.join(loc_ds, experiment, dataset))[None]
    elif dataset.split('.')[-1]=='pck':
        df_raw = pd.read_pickle(os.path.join(loc_ds, experiment, dataset))
    else:
        df_raw = pd.read_csv(os.path.join(loc_ds, experiment, dataset))

    # Importing experiment dataset
    df = dsrp.ExperimentDataSet(experiment_dataset=df_raw
                                , target_col_index=target_column_index
                                , filter_gene=list_gene
                                , scaler=scale).run()

    print('prior bio shape, ', df_bio.shape)

    # Exporting preprocessed dataset
    df.to_pickle(os.path.join(loc_output,  file_name_output ))
    print('Experiment datasets are exported into ', os.path.join(loc_output, file_name_output))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment', help='specify the experiment, the experiment name should be located in ./data/external')
    parser.add_argument('-loc', '--location', help='specify the dataset location in data folder, in ./data/{LOCATION}')
    parser.add_argument('-ds' , '--dataset', help='specify the dataset, in ./data/external/{FILE NAME}')
    parser.add_argument('-pbk', '--bio_knowledge', help='specifying the biological knowledge dataset. None for keeping all genes' )
    parser.add_argument('-sc' , '--scale', help='defining scaling operation, None for keeping raw samples')
    parser.add_argument('-tci', '--target_column_index', help='index number of target column, -1 for last column', type=int)
    
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    preprocessing_dataset(args.experiment, args.location,  args.dataset, args.bio_knowledge, eval(args.scale), args.target_column_index)