import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import math
import matplotlib.pyplot as plt
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
for i in range(1, 2):
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

    # choose piecewise or linear RUL function
    piecewise = True
    # load training data
    [X_train, y_train] = prepare_sub_dataset(DATA_DIR, 
                                 train_file, 
                                 window_size=window_size,
                                 piecewise=piecewise)
    n_features = len(X_train[1,1,:])
    
    # checkpoints and logs for TensorBoard
    ckpt_file ="weights_FD00{}_piecewiseRUL2.hdf5".format(i)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = train_per_variable(X_train, y_train, ckpt_path, logdir, window_size, train=False)

    # plot train predictions
    print(X_train[0])
    print(y_train[0])
    X_train = split_timeseries_per_feature(X_train, n_features)
    predictions = model.predict(X_train)
    for i in range(100):
        print("Ground truth vs prediction on train data:{} - {}".format(y_train[i], predictions[i]))
    predictions = predictions.reshape(len(predictions),)
    testRMSE = math.sqrt(sum((predictions - y_train) ** 2)/len(y_train))
    print('Test set RMSE:{}'.format(testRMSE))
    unit = np.arange(0, len(y_train))
    plt.figure(1,figsize=(7,5))
    plt.plot(unit[0:162], predictions[0:162], 'r--')
    plt.plot(unit[0:162], y_train[0:162], 'b-')
    plt.xlabel('Test units')
    plt.ylabel('RUL')
    plt.grid(True)
    plt.style.use(['seaborn-ticks'])

    # load test data
    [X_test, y_test] = prepare_sub_dataset(DATA_DIR, 
                                        test_file, 
                                        RUL_file, 
                                        test=True,
                                        window_size=window_size,
                                        piecewise=piecewise)
    
    # run predictions on test set and compute rmse
    X_test = split_timeseries_per_feature(X_test, n_features)
    predictions = model.predict(X_test)
    # reconstruct predictions from normalized values
    # predictions = predictions * (max_y - min_y) + min_y
    for i in range(100):
        print("Ground truth vs prediction on test data:{} - {}".format(y_test[i], predictions[i]))

    # plot test units predictions
    predictions = predictions.reshape(len(predictions),)
    testRMSE = math.sqrt(sum((predictions - y_test) ** 2)/len(y_test))
    print('Test set RMSE:{}'.format(testRMSE))
    unit = np.arange(0, len(y_test))
    plt.figure(2,figsize=(7,5))
    plt.plot(unit, predictions, 'r--')
    plt.plot(unit, y_test, 'b-')
    plt.xlabel('Test units')
    plt.ylabel('RUL')
    plt.grid(True)
    plt.style.use(['seaborn-ticks'])
    plt.show()