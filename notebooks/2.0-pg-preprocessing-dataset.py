#!/usr/bin/env python

'''
DESCRIPTION
-----------
    Preprocessing raw dataset -> applying scaling, filtering genes.
    
USAGE
-----
    [PROJECT_PATH]/$ python notebooks/2.0-pg-preprocessing-dataset.py -exp {EXPERIMENT}
                                                                      -ds  {DATASET}
                                                                      -sc  {SCALER}
                                                                      -sw  {SAMPLE-WISE}
                                                                      -tci {TARGET COLUMN INDEX}
                                                                      -ofn {OUTPUT FILE NAME}

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
from scripts import config as src
import pandas as pd
import numpy as np

def preprocessing_dataset(experiment, dataset, sample_wise, gene_wise, scale, target_column_index, ofn):
    
    try:
        
        print(type(sample_wise))
        if os.path.exists(os.path.join(src.DIR_DATA_EXTERNAL, experiment, dataset)) ==False:
            print(os.path.join(src.DIR_DATA_EXTERNAL, experiment, dataset))
            raise Exception('******* INVALID dataset path')
            
#         print(experiment, dataset, scale, target_column_index)    
        print('READING DATASET --> ', os.path.join(src.DIR_DATA_EXTERNAL, experiment, dataset) )
        
        # the output location
        loc_output = os.path.join(src.DIR_DATA_PROCESSED, experiment)
        src.define_folder(loc_=loc_output)
        
        read_df_loc = os.path.join(src.DIR_DATA_EXTERNAL, experiment, dataset)

        df = pd.read_pickle(read_df_loc)
        df = src.dataframe_modification(df, target_column_index)
        
#         SAMPLE-WISE NORMALIZATION
        if sample_wise==True:
            df = src.sample_wise_normalization(df)
            ofn = ofn+'_sw'

#         GENE-WISE NORMALIZATION
        if gene_wise==True:
            df = src.gene_wise_normalization(df)
            ofn = ofn+'_gw'
            
#         APPLYING DEFINED NORMALIZATION
        if scale != 'None':
            df = src.scaler_normalization(df, scale)
            ofn = ofn+'_'+scale
            
        print('******* DATASET INFO *******')
        print(df.info())
        print('******* DATASET INFO *******')
        
#         Exporting preprocessed dataset
        df.to_pickle( os.path.join(src.DIR_DATA_PROCESSED, experiment, ofn+'.pck') )
        print('Experiment datasets are exported into ', os.path.join(src.DIR_DATA_PROCESSED, experiment, ofn+'.pck'))
        
    except Exception as error:
        print('\n{0}'.format(error))
    except:
        print("Unexpected error:", sys.exc_info()[0])

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp', '--experiment', help='specify the experiment, the experiment name should be located in ./data/external')
    parser.add_argument('-ds' , '--dataset', help='specify the dataset, in ./data/external/{FILE NAME}')
    parser.add_argument('-sw' , '--sample_wise', help='applying sample-wise normalization')
    parser.add_argument('-gw' , '--gene_wise', help='applying gene-wise normalization')
    parser.add_argument('-sc' , '--scale', help='defining scaling operation, None for keeping raw samples')
    parser.add_argument('-tci', '--target_column_index', help='index number of target column, -1 for last column')
    parser.add_argument('-ofn', '--output_file_name', help='name of the preprocessed dataset')

#     parser.add_argument('-loc', '--location', help='specify the dataset location in data folder, in ./data/{LOCATION}')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    preprocessing_dataset(args.experiment
                          , args.dataset
                          , eval(args.sample_wise)
                          , eval(args.gene_wise)
                          , args.scale
                          , int(args.target_column_index)
                          , args.output_file_name)