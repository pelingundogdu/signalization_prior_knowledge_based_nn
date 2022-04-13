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
import matplotlib.pyplot as plt
import seaborn as sns

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

loc_output = './reports/clustering/exper_mouse/'
df_result = pd.DataFrame()
for i in sorted(glob.glob('./reports/clustering/exper_mouse/metrics*')):
    df_result = pd.concat([df_result, pd.read_csv(i)])

df_result = df_result[['cell_out','design','homogeneity','completeness','v_measure','ari','ami','fowlkes_mallows','mean']] 
    
df_metrics_detail = pd.DataFrame()
for i in sorted(glob.glob(f'./reports/clustering/exper_mouse/detail_*.csv')):
    clustering_metrics = src.calculate_clustering_metrics(pd.read_csv(i))
    clustering_metrics['design'] = i.split('/')[-1].split('mouse')[0].split('detail')[-1][1:-1]
    df_metrics_detail = pd.concat([df_metrics_detail, clustering_metrics ], axis = 0)

    
for i_co in sorted(set(df_result['cell_out'])):
    df_temp = df_result[(df_result['cell_out']==i_co)]
    df_temp.to_csv(os.path.join(loc_output, f'results_{i_co}.csv'), index=False)
    df_metrics_co = df_metrics_detail[df_metrics_detail['cell_out']==i_co]

    sns.set_palette("tab10")
    plt.figure(figsize=(20,6))
    sns.boxplot(data=df_metrics_co, x='metric', y='score', hue='design');
    plt.xticks(rotation=5)
    plt.xlabel('')
    # legend = plt.legend(title='Proposed network', loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=False, shadow=False, ncol=2)
    legend = plt.legend(title='Design', loc='upper center', bbox_to_anchor=(1.15, 0.75))
    legend.get_title().set_fontsize(SMALL_SIZE) #legend 'Title' fontsize
    plt.xticks(rotation=45)
    plt.title(f'Clustering Analysis - {i_co}')
    plt.tight_layout();
    plt.savefig(os.path.join(loc_output,(f'1_metrics_clustering_{i_co}.png')), dpi=300, bbox_inches = 'tight')
    plt.savefig(os.path.join(loc_output,(f'1_metrics_clustering_{i_co}.pdf')), dpi=300, bbox_inches = 'tight')
    plt.savefig(os.path.join(loc_output,(f'1_metrics_clustering_{i_co}.svg')), dpi=300, bbox_inches = 'tight')