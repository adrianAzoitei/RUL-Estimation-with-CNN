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
from data_loader.data_prep import prepare_sub_dataset, add_RUL_linear, add_RUL_piecewise, normalize_data, sliding_window, split_timeseries_per_feature
from trainers.trainer_per_variable import train_per_variable
# from trainers.trainer_original import trainc

# take files sepparately
for i in range(4, 5):
    train_file = 'train_FD00{}.txt'.format(i)
    test_file  = 'test_FD00{}.txt'.format(i)
    RUL_file = 'RUL_FD00{}.txt'.format(i)
    print('Training on FD00{}'.format(i))

    # load data and prepare for training
    if i == 1 or i == 3:
        window_size = 30
    elif i == 2:
        window_size = 20
    elif i == 4:
        window_size = 15
    
    [X, y] = prepare_sub_dataset(DATA_DIR, train_file, window_size=window_size)
    # y = 2 * (y - min(y)) / (max(y) - min(y)) - 1
    min_y = min(y)
    max_y = max(y)
    y = (y - min_y) / (max_y - min_y)
    print("X.shape:{}, y.shape:{} ".format(X.shape, y.shape))
    print("y[-3] reconstructed: ", y[-3] * (max_y - min_y) + min_y)
    # load validation data
    [X_val, y_val] = prepare_sub_dataset(DATA_DIR, 
                                        test_file, 
                                        RUL_file, 
                                        test=True,
                                        window_size=window_size)

    # take a couple of values from the test set to run some predictions
    X_predict = X_val[-5:]
    X_predict = X_predict.reshape(len(X_predict), window_size, 14)
    y_predict = y_val[-5:]

    # min-max normalize labels
    # store min, max to reconstruct RUL values for predictions
    min_yv = min(y_val)
    max_yv = max(y_val)
    y_val = (y_val - min_yv) / (max_yv - min_yv)
    print("y_val[0]: ", y_val[0])
    print("y_val[0] reconstructed: ", y_val[0] * (max_yv - min_yv) + min_yv)

    # checkpoints and logs for TensorBoard
    ckpt_file ="weights_FD00{}.hdf5".format(i)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train per-variable CNN
    model, history = train_per_variable(X, y, X_val, y_val, ckpt_path, logdir, window_size)

    # train normal CNN
    # model = train(X, y, X_val, y_val, ckpt_path, logdir)

    # split input training data into separate time series, per feature
    X_predict = split_timeseries_per_feature(X_predict, 14)
    predictions = model.predict(X_predict)

    # reconstruct predictions from normalized values
    # predictions = (predictions + 1) * (max_yv - min_yv) / 2 + min_yv
    predictions = predictions * (max_yv - min_yv) + min_yv
    print("Predictions:{}, True:{}".format(predictions, y_predict))