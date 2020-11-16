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
for i in range(4, 5):
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

    # decide if using piecewise or linear RUL function for training data
    piecewise = True
    # load training data
    [X, y] = prepare_sub_dataset(DATA_DIR, 
                                 train_file, 
                                 window_size=window_size, 
                                 piecewise=piecewise)
    # min-max normalize labels
    # min_y = min(y)
    # max_y = max(y)
    # y = (y - min_y) / (max_y - min_y)

    # split data into training and validation
    split = int(len(X[:, 1, 1])*.7)
    X_train = X[:split]
    y_train = y[:split]
    X_val = X[split:]
    y_val = y[split:]

    # load test data
    [X_test, y_test] = prepare_sub_dataset(DATA_DIR, 
                                        test_file, 
                                        RUL_file, 
                                        test=True,
                                        window_size=window_size,
                                        piecewise=piecewise)

    # checkpoints and logs for TensorBoard
    ckpt_file ="weights_FD00{}.hdf5".format(i)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    # train per-variable CNN
    model, history = train_per_variable(X_train, y_train, X_val, y_val, ckpt_path, logdir, window_size)

    # import matplotlib.pyplot as plt
    # nb_epoch = len(history.history['loss'])
    # learning_rate=history.history['lr']
    # xc=range(nb_epoch)
    # plt.figure(3,figsize=(7,5))
    # plt.plot(xc,learning_rate)
    # plt.xlabel('num of Epochs')
    # plt.ylabel('learning rate')
    # plt.title('Learning rate')
    # plt.grid(True)
    # plt.style.use(['seaborn-ticks'])
    # plt.show()
    # # train normal CNN
    # model = train(X, y, X_val, y_val, ckpt_path, logdir, window_size)

    n_features = len(X[1,1,:])
    X_test = split_timeseries_per_feature(X_test, n_features)
    predictions = model.predict(X_test)
    # predictions = y * (max_y - min_y) + min_y
    # # reconstruct predictions from normalized values
    print("Ground truth vs prediction on test data:{} - {}".format(y_test[1], predictions[1]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_test[60], predictions[60]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_test[10], predictions[10]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_test[7], predictions[7]))
    print("Ground truth vs prediction on test data:{} - {}".format(y_test[3], predictions[3]))