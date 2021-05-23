#!/usr/bin/env python
# coding: utf-8

# Required libraries
import os
import numpy as np
import pandas as pd
import datetime as dt
from scripts import config as src

def define_folder(loc_):
    """
    Creating folder based on the giving location information. 
    If the given information is not folder, it gives error message.
    
    Parameters
    ----------
    loc_ : str
        The location of folder
    Returns
    -------
    path_ : str
        It gives the created location.
    """
#     checking if there is any mistake with '/' sign
#     print(loc_)
    prefix = ''
    if loc_[0]=='/':
        prefix='/'
    loc_ = [x for x in loc_.split('/') if x != '']
    loc_ = '/'.join(loc_)
    loc_ = prefix+loc_
#     returning message if the given information is different than folder.
    if (loc_.split('/')[-1].find('.')>0==False):
        print('PLEASE ENTER FOLDER PATH!!, given information is ',loc_)
    else:
        path_ = ''
        count = 0
        for s_ in loc_.split('/'):
            path_ = path_+s_+'/'
#             checking the existence of location, if it does not exist, it creates the folder in order to given loc_ information
            if os.path.exists(path_)==False:
                count=count+1
                os.mkdir(path_)
        
        if count >0:
            print('PATH created!!')
        
        print('FOLDER information, ', path_)
    
    return(path_)


class Export_to_text:
    def __init__(self, experiment, detail, loc):
        self.experiment = experiment
        self.detail = detail
        self.loc = loc

    def save(self, text, file_operation='a+'):
        
        info_text = os.path.join(self.loc, 'info_'+self.detail+'.txt')
        
        f=open(info_text, file_operation)
        f.write(text)
        f.write('\n\n')
        f.close()