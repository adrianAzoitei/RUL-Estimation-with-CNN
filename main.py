import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import system and file handling packages
import os
import sys
# set path to working directory
path = os.getcwd()
rootpath = os.path.join(path, os.pardir)
roothpath = os.path.abspath(rootpath)
sys.path.insert(0, path)

#import defined variables and methods
from utils import DATA_DIR, CKPT_DIR, LOG_DIR
from data_loader.data_prep import prepare_sub_dataset, split_timeseries_per_feature
from trainers.trainer_per_variable import train_per_variable
# from trainers.trainer_original import train

# take files sepparately
for i in range(3, 4):
    train_file = 'train_FD00{}.txt'.format(i)
    test_file  = 'test_FD00{}.txt'.format(i)
    RUL_file = 'RUL_FD00{}.txt'.format(i)
    print('Training on FD00{}'.format(i))

    if i == 1 or i == 3:
        window_size = 30
    elif i == 2:
        window_size = 20
    elif i == 4:
        window_size = 15

    # load training data
    [X, y] = prepare_sub_dataset(DATA_DIR, train_file, window_size=window_size)
    n_features = len(X[1,1,:])
    # min-max normalize labels
    min_y = min(y)
    max_y = max(y)
    # y = (y - min_y) / (max_y - min_y)

    # load test data
    [X_val, y_val] = prepare_sub_dataset(DATA_DIR, 
                                        test_file, 
                                        RUL_file, 
                                        test=True,
                                        window_size=window_size)
    # min-max normalize labels
    min_yv = min(y_val)
    max_yv = max(y_val)
    # y_val = (y_val - min_yv) / (max_yv - min_yv)

    # checkpoints and logs for TensorBoard
    ckpt_file ="weights_FD00{}.hdf5".format(i)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train per-variable CNN
    model = train_per_variable(X, y, X_val, y_val, ckpt_path, logdir, window_size)

    # train normal CNN
    # model = train(X, y, X_val, y_val, ckpt_path, logdir, window_size)

    X_val = split_timeseries_per_feature(X_val, n_features)
    predictions = model.predict(X_val)
    # # reconstruct predictions from normalized values
    # # predictions = predictions * (max_y - min_y) + min_y
    # print(X_val[0].shape)
    print("Ground truth vs prediction on test data:{} - {}".format(y_val[1], predictions[1]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_val[60], predictions[60]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_val[10], predictions[10]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_val[7], predictions[7]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_val[3], predictions[3]))