#!/usr/bin/env python
# coding: utf-8

# Required libraries
import os
import numpy as np
import pandas as pd

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
<<<<<<< HEAD
    prefix=''
    if loc_[0] == '/':
=======
    prefix = ''
    if loc_[0]=='/':
>>>>>>> develop
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
<<<<<<< HEAD
                os.mkdir(path_)
                count = count+1
            
        if count>0:
            print('PATH created!!')
        
        print('FOLDER information!, ', path_)
=======
                count=count+1
                os.mkdir(path_)
        
        if count >0:
            print('PATH created!!')
        
        print('FOLDER information, ', path_)
>>>>>>> develop
    
    return(path_)


# def define_folder(loc_):
#     """
#     Creating folder based on the giving location information. 
#     If the given information is not folder, it gives error message.
    
#     Parameters
#     ----------
#     loc_ : str
#         The location of folder
#     Returns
#     -------
#     path_ : str
#         It gives the created location.
#     """
# #     checking if there is any mistake with '/' sign
# #     print(loc_)
#     loc_ = [x for x in loc_.split('/') if x != '']
#     loc_ = '/'.join(loc_)
# #     returning message if the given information is different than folder.
#     if (loc_.split('/')[-1].find('.')>0==False):
#         print('PLEASE ENTER FOLDER PATH!!, given information is ',loc_)
#     else:
#         path_ = ''
#         for s_ in loc_.split('/'):
#             path_ = path_+s_+'/'
# #             checking the existence of location, if it does not exist, it creates the folder in order to given loc_ information
#             if os.path.exists(path_)==False:
#                 print('PATH created!!, ', path_)
#                 os.mkdir(path_)
    
#     return(path_)