'''
    The path information of folders file
'''
import os, sys

DIR_ROOT = os.path.dirname(os.path.abspath('__file__'))

os.chdir(DIR_ROOT)
sys.path.append(DIR_ROOT)
# DIR_CONFIG = os.path.dirname(os.path.abspath('__file__'))
DIR_CONFIG = os.path.join('.','scripts')
DIR_ROOT_BACK = os.path.dirname(DIR_ROOT)
DIR_DATA = os.path.join(DIR_ROOT_BACK,'00_data',os.path.basename(DIR_ROOT))
DIR_MODELS = os.path.join('.','models')
DIR_REPORTS = os.path.join('.','reports')

DIR_DATA_EXTERNAL = os.path.join(DIR_DATA,'external')
DIR_DATA_INTERIM = os.path.join(DIR_DATA,'interim')
DIR_DATA_PROCESSED = os.path.join(DIR_DATA,'processed')
DIR_DATA_RAW = os.path.join(DIR_DATA,'raw')

DIR_SRC = os.path.join('.','src')
SRC_DATA = os.path.join(DIR_SRC, 'data/')
SRC_FEATURES = os.path.join(DIR_SRC, 'features/')
SRC_MODELS = os.path.join(DIR_SRC, 'models')
SRC_VISUALIZATION = os.path.join(DIR_SRC, 'visualization')


print('**** PROJECT FOLDER     , ', DIR_ROOT)
print('**** PROJECT DATA FOLDER, ', DIR_DATA)
print('**** scripts/settings.py - PATHS IMPORTED!!!')
# print(os.path.dirname(os.path.dirname(DIR_ROOT)))
# os.system('python visualization_scripts.py')
from scripts.dataset_scripts import *
from scripts.model_scripts import *
import scripts.nn_design_scripts as nn
from scripts.path_scripts import *
from scripts.visualization_scripts import *

# import inspect

# if not hasattr(sys.modules[__name__], '__file__'):
#     __file__ = inspect.getfile(inspect.currentframe())
    
# print(__file__)
