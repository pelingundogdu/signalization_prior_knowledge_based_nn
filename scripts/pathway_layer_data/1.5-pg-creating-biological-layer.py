#!/usr/bin/python

'''
DESCRIPTION 
-----------
    Creating prior biological knowledge matrix. The matrix will implemented into NN design in first hidden layer.

USAGE
-----
    [PROJECT_PATH]/$ python scripts/pathway_layer_data/1.5-pg-creating-biological-layer.py -sp {SPECIES} -src {SOURCE}

RETURN
------
    bio_layer.txt : txt file
        The gene x pathway matrix which stores the prior biological knowledge
        
EXPORTED FILE(s) LOCATION
-------------------------
    ./data/processed/hsa/hipathia/bio_layer.txt
'''

import os, argparse, sys, glob
import pandas as pd
import numpy as np

def create_biological_layer(species, source):
    sys.path.append('./')
    from scripts import config as src

    # defining output folder
    output_folder = src.define_folder( os.path.join(src.DIR_DATA_PROCESSED, species, source ) )
    print(output_folder)
    # importing raw dataset which is imported by hipathia
    df_entrez_symbol = pd.read_csv(os.path.join(src.DIR_DATA_PROCESSED, species, source, 'entrez_and_symbol.csv'))
    print(df_entrez_symbol.shape)
    print(df_entrez_symbol.head())
    
    print('Checking NA values, number of NA values, ', len(df_entrez_symbol.loc[df_entrez_symbol['symbol'].isna()]))
    
    
    for i_pathway in sorted(glob.glob(os.path.join(src.DIR_DATA_PROCESSED, species, source, 'details/*gene_list.txt'))):
        # Reading selected pathways which shows the gene relation for each circuits
        df_i_pathway = pd.read_csv(i_pathway, index_col=0).fillna(value=0)
        # Replace columns names which is circuit name with pathway name
        df_i_pathway.columns = [pw[1] for pw in df_i_pathway.columns.str.split('-')]
        # Grouping all circuit as pathway representation
        df_i_pathway = df_i_pathway.groupby(df_i_pathway.columns, axis=1).max()
        # Merging entrez_and_symbol dataset with pathway information dataset
        df_entrez_symbol = pd.merge(left=df_entrez_symbol, right=df_i_pathway, left_on='gene_id', right_index=True, how='left')
        df_entrez_symbol.fillna(value=0, inplace=True)

    # Updating 'symbol' values as lowercase
    df_entrez_symbol['symbol'] = df_entrez_symbol['symbol'].str.lower()    
    df_entrez_symbol.drop(columns=['gene_id'], inplace=True)

    # EXPORTING - the prior biological knowledge layer
    print('The prior biological knowledge layer EXPORTED!! - {}'.format(os.path.join(src.DIR_DATA_PROCESSED, species, source, 'bio_layer_'+species+'.txt')))
    df_entrez_symbol.to_csv(os.path.join(src.DIR_DATA_PROCESSED, species, source, 'bio_layer_'+species+'.txt'), index=False)
    print(df_entrez_symbol.shape)
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-sp', '--species', help='specify the species, the location of species in ./data/raw/{SPECIES}')
    parser.add_argument('-src', '--source', help='specify the source, the location of source in ./data/raw/{SPECIES}/{SOURCE}')
    parser.add_argument('-ga', '--genome_annotation', help='specify genome wide annotition package', default=None)

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    create_biological_layer(args.species, args.source)