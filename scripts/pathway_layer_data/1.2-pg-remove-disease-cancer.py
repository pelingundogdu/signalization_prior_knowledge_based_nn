#!/usr/bin/python

'''
DESCRIPTION
-----------
    Removing disease related pathways from pathways which are obtained via hipathia package.
    
USAGE
-----
    [PROJECT_PATH]/$ python scripts/pathway_layer_data/1.2-pg-remove-disease-cancer.py -sp {SPECIES} -src {SOURCE}

RETURN
------
    pathway_ids_and_names.csv : csv file
        Final version after removed disease related pathways 

EXPORTED FILE(s) LOCATION
-------------------------
    ./data/processed/hsa/hipathia/pathway_ids_and_names.csv
'''

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import config as src
import pandas as pd
import numpy as np

def remove_disease_from_dataset(species, source):

    # defining output folder
    output_folder = src.define_folder( os.path.join(src.DIR_DATA_PROCESSED, species, source ) )
    
    # importing raw dataset which is imported by hipathia
    df_hp = pd.read_csv(os.path.join(src.DIR_DATA_RAW, species, source, 'pathway_ids_and_names.csv'))
    print('RAW dataset,')
    print('.head()', df_hp.head())
    print('Shape,', df_hp.shape)


    # FILTERING #1
    # filtering raw dataset according to keyword, shared in below
    keywords_ = ['disease', 'cancer', 'leukemia', 'infection', 'virus','addiction', 'anemia', 'cell carcinoma', 'diabet', 'Hepatitis']
    df_hp = df_hp.loc[~df_hp['path.name'].str.contains('|'.join(keywords_))]
    print('RAW dataset is filtered by "keywords" list!')
    print('Shape,', df_hp.shape)

    # FILTERING #2
    # filtering again according to remained disease name, shared in below
    additional_disease = ['Long-term depression', 'Insulin resistance', 'Amyotrophic lateral sclerosis (ALS)', 'Alcoholism', 'Shigellosis'
                          , 'Pertussis', 'Legionellosis', 'Leishmaniasis', 'Toxoplasmosis', 'Tuberculosis', 'Measles', 'Influenza A'
                          , 'Glioma', 'Melanoma']

    df_hp = df_hp.loc[~df_hp['path.name'].isin(additional_disease)]
    print('RAW dataset is filtered by "additional_disease" list!')
    print('Shape,', df_hp.shape)

    # exporting processed dataset
    df_hp.to_csv(os.path.join(output_folder, 'pathway_ids_and_names.csv'), index=False)
    print('FILE exported in {}'.format(os.path.join(output_folder, 'pathway_ids_and_names.csv')))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--species', help='specify the species, the location of species in ./data/raw/{SPECIES}')
    parser.add_argument('-src', '--source', help='specify the source, the location of source in ./data/raw/{SPECIES}/{SOURCE}')
    parser.add_argument('-ga', '--genome_annotation', help='specify genome wide annotition package', default=None)

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    remove_disease_from_dataset(args.species, args.source)