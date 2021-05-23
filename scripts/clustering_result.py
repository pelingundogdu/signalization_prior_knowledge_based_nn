#!/usr/bin/env python

'''
DESCRIPTION
-----------
    Exporting dataset into pck format
    
RETURN
------
    {DATASET}.pck : pck file
        pck version of file

EXPORTED FILE(s) LOCATION
-------------------------
    ./data/external/{EXPERIMENT}/{DATASET}.pck
'''

# importing default libraries
import os, argparse, sys
sys.path.append('./')
# importing scripts in scripts folder
from scripts import config as src
import pandas as pd
import numpy as np
import glob 
import warnings
warnings.filterwarnings('ignore')

loc_output = './reports/clustering/exper_mouse/'
df_result = pd.DataFrame()
for i in sorted(glob.glob('./reports/clustering/exper_mouse/metrics*')):
    df_result = pd.concat([df_result, pd.read_csv(i)])
    
df_result = df_result[['cell_out','design','homogeneity','completeness','v_measure','ari','ami','fowlkes_mallows','mean']] 

for i_co in sorted(set(df_result['cell_out'])):
    df_temp = df_result[(df_result['cell_out']==i_co)]
    df_temp.to_csv(os.path.join(loc_output, f'metrics_{i_co}.csv'), index=False)