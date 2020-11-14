# import system and file handling packages
import os
import sys
# set path to working directory
path = os.getcwd()
rootpath = os.path.join(path, os.pardir)
roothpath = os.path.abspath(rootpath)
sys.path.insert(0, path)

DATA_DIR = os.path.join(path, 'dataset')
DATA_DIR = os.path.join(DATA_DIR, 'raw')
CKPT_DIR = os.path.join(path, 'checkpoints')
LOG_DIR  = os.path.join(path, 'logs')
