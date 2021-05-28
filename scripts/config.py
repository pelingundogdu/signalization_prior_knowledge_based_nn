'''
    The path information of folders file
'''
import os, sys

DIR_ROOT = os.path.dirname(os.path.abspath('__file__'))

os.chdir(DIR_ROOT)
sys.path.append(DIR_ROOT)
DIR_CONFIG = os.path.join(DIR_ROOT,'scripts')
# DIR_ROOT_BACK = os.path.dirname(DIR_ROOT)
# DIR_DATA = os.path.join(DIR_ROOT_BACK,'00_data',os.path.basename(DIR_ROOT))
DIR_DATA = os.path.join(DIR_ROOT,'data')
DIR_MODELS = os.path.join(DIR_ROOT,'models')
DIR_REPORTS = os.path.join(DIR_ROOT,'reports')
DIR_NOHUP = os.path.join(DIR_ROOT,'nohup')

DIR_DATA_EXTERNAL = os.path.join(DIR_DATA,'external')
DIR_DATA_INTERIM = os.path.join(DIR_DATA,'interim')
DIR_DATA_PROCESSED = os.path.join(DIR_DATA,'processed')
DIR_DATA_RAW = os.path.join(DIR_DATA,'raw')

print('**** scripts/config.py IMPORTED!!!')
print('**** PROJECT FOLDER , ', DIR_ROOT)
# print('**** PROJECT DATA FOLDER    , ', DIR_DATA)
# print('**** PROJECT DATA/RAW FOLDER, ', DIR_DATA_RAW)

from scripts.dataset_scripts import *
from scripts.model_scripts import *
from scripts.nn_design_scripts import *
from scripts.path_scripts import *
from scripts.visualization_scripts import *
from scripts.metrics_and_split_scripts import *
from scripts.scibet_compare import *
from scripts.autoencoder import *