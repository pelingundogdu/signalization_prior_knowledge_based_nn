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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def scibet_compare(y_with_seen, y_with_unseen, X_with_unseen, lof_unseen, model, threshold):
    
    ohe = OneHotEncoder()
    ohe.fit_transform(y_with_seen)

    y_ohe_query = OneHotEncoder().fit_transform(y_with_unseen).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_with_unseen, y_with_unseen
                                                        , test_size=0.3
                                                        , shuffle=True
                                                        , random_state=91
                                                        , stratify=y_ohe_query)
    
    df_below_threshold = np.where(lof_unseen['score'] <= threshold)[0]
    y_truth_pred = y_train.copy()#.reset_index(drop=True)
    y_truth_pred.loc[y_truth_pred.index.isin(df_below_threshold), 'threshold'] = 'unassigned'
    y_truth_pred.reset_index(drop=True, inplace=True)
    y_truth_pred['pred'] = pd.DataFrame([ohe.categories_[0][i] for i in np.argmax(model.predict(X_train), axis=-1)])
    y_truth_pred.loc[y_truth_pred['threshold']=='unassigned', 'pred']='unassigned'
    
    return y_truth_pred
