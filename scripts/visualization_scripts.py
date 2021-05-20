#!/usr/bin/env python
# coding: utf-8

# Required libraries
import os
# import pyreadr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

TINY_SIZE = 8
SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('legend', title_fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def heatmap_cv(dataframe_crosstab, title):
    '''
    Visualization of heatmap for giving pandas.crosstab(ground_truth, pred_label)
    Parameters
    ----------
    dataframe_crosstab : dataframe (integer)
        Dataset values for ground truth vs predicted label 
    Returns
    -------
    g : seaborn.heatmap
        The heatmap visualization
    '''
#     cmap = mpl.cm.Blues(np.linspace(0,1,350))
#     cmap = mpl.colors.ListedColormap(cmap[10:,:-1])
    
    heatmap_cv_hex = ['#D53E4F','#f46d43','#fdae61', '#F4E88F','#9BD690','#74add1','#3188BD']
#     sns.set_palette(heatmap_cv_hex)
    heatmap_cv_rgb = list()
    for i in heatmap_cv_hex[::-1]:
#         print(mpl.colors.hex2color(i))
        heatmap_cv_rgb.append(mpl.colors.hex2color(i))
    
    # heatmap_cv_rgb = [(e[0] / 255.0, e[1] / 255.0, e[2] / 255.0) for e in heatmap_cv_rgb]
    nc = len(heatmap_cv_rgb)
    c = np.zeros((3, nc, 3))
    rgb = ['red', 'green', 'blue']
    for idx, e in enumerate(heatmap_cv_rgb):
        for ii in range(3):
            c[ii, idx, :] = [float(idx) / float(nc - 1), e[ii], e[ii]]

    cdict = dict(zip(rgb, c))
    cmap = mpl.colors.LinearSegmentedColormap('heatmap_cv', cdict)
    
    mpl.pyplot.figure(figsize=(12,6))
    g = sns.heatmap(dataframe_crosstab
                    , cmap=cmap#flatui[::-1]
                    , annot=True, fmt='.3f'
                    , linewidths=0.1
                    , linecolor='white');
#     g = sns.heatmap(dataframe_crosstab, cmap=cmap, annot=True, fmt='.2f');
    g.set(xlabel='PREDICTION', ylabel='GROUND TRUTH', title=title)
    
    return g.get_figure()


def plot_expression_and_sum(df, title, random_genes, png=None, output=None):
#     df['sum'] = df.iloc[:, :-1].sum(axis=1)
    fig, axes = plt.subplots(ncols=2, figsize=(15,5))
#     plt.figure(figsize=(10,5))
    sns.distplot(df.iloc[:, :-1].sum(axis=1), ax=axes[0])
#     sns.displot(pd.melt(df[random_genes]), x='value', hue='variable', kind='kde', legend=False)
    df_melt = pd.melt(df[random_genes])
    my_order = df_melt.groupby(by=['variable'])['value'].median().sort_values(ascending=False).index
    sns.boxplot(x="variable", y="value", data=df_melt, order=my_order, ax=axes[1])
#     plt.title('randomly selected 50 genes - ordered by mean')
    axes[0].set_title('distribution of gene expression of each samples')
    axes[0].set_yticks([])
    
    axes[1].set_title('randomly selected 50 genes - ordered by mean')
    axes[1].set_xlabel('genes')
    axes[1].set_ylabel('expression value')
    axes[1].set_xticks([])
    
    plt.tight_layout(pad=3, w_pad=0.5, h_pad=2.0)
    fig.suptitle(title)
    if png != None:
        plt.savefig(os.path.join(output, png), dpi=300, bbox_inches = 'tight');
        print('EXPORTED!!, ', os.path.join(output, png))
    
def scibet_confusion_matrix(dataframe):
    cmap = mpl.cm.Blues(np.linspace(0,1,350))
    cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

    mpl.pyplot.figure(figsize=(12,6))

    g = sns.heatmap(dataframe.div(dataframe.sum(axis=1).values, axis='rows'), cmap=cmap, annot=True);
    g.set(xlabel='PREDICTION', ylabel='GROUND TRUTH');
    
    return g.get_figure()