import pandas as pd
import numpy as np
import os
from data_loader.read_data import read_data

# add RUL column
# copyright: https://www.kaggle.com/vinayak123tyagi/damage-propagation-modeling-for-aircraft-engine
def add_RUL_linear(data, factor = 0):
    """
    This function appends a RUL column to the df by means of a linear function.
    """
    df = data.copy()
    fd_RUL = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['unit_number','max']
    df = df.merge(fd_RUL, on=['unit_number'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]

def add_RUL_piecewise(data, factor = 0):
    """
    This function appends a RUL column to the df by means of a piece-wise function.
    """
    # TODO
    return None

# def add_RUL_mockup(array):
#     RUL_list = []
#     for i in range(len(array[:,1])):
#         RUL = array[i][0] + 4*array[i][1] + 2*array[i][2] + 3*array[i][3] + 7*array[i][4]
#         RUL_list.append(RUL)
#     RUL_list = np.asarray(RUL_list)
#     RUL_list = RUL_list.reshape(len(RUL_list),1)
#     array = np.hstack((array, RUL_list))
#     return array

def normalize_data(array, test):
    """
    This function normalizes the data with min-max normalization as specified in the scientific paper.

    Input: numpy array of shape (rows, columns)
    Output: normalized numpy array of shape (rows, columns).
    """
    norm_array = np.ones((len(array[:,1]), 1))
    for i in range(len(array[1,:])):
        if ((i == len(array[1,:])-1) and not test) or (i < 5):
            original_col = array[:,i].reshape((len(array[:,i]), 1))
            norm_array = np.hstack((norm_array, original_col))
        else:
            norm_array_i = (2*(array[:,i] - min(array[:,i])) / (max(array[:,i]) - min(array[:,i])) - 1).reshape((len(array[:,1]), 1))
            norm_array = np.hstack((norm_array,norm_array_i))
    norm_array = np.delete(norm_array, 0, 1)
    return norm_array

def sliding_window(sequence, window_size):
    X = []
    y = []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the dataset
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    return X,y

def test_samples(sequence, window_size):
    # sequence is the array of values associated with ONE engine unit
    X = []
    # find the start of this pattern
    start_ix = len(sequence) - window_size
    seq_x = sequence[start_ix:len(sequence), :]
    X.append(seq_x)
    X = np.array(X)
    return X

def prepare_sub_dataset(data_dir, filename, validation_RUL_file="", test=False, window_size=30):
    """
    This function does the following:
    1) Reads a FD00X file associated with one sub-dataset into a pandas DataFrame.
    2) Drops the unnecessary sensor reading columns as specified in the scientific paper.
    4) Converts the DataFrame to a numpy array.
    3) Appends the piece-wise RUL values.
    5) Normalizes the features, except the unit id, cycles, and RUL.
    6) Samples the sub-dataset with a sliding time window strategy for each engine unit.

    Inputs: the location of the dataset files.
    Returns: A numpy array of dimensions (samples, window_length, features)
    """
    # 1)
    df = read_data(data_dir, filename)

    # 2)
    df.drop(columns=['s1','s5','s6','s10','s16','s18','s19'], inplace=True)

    if not test:
        # 3)
        df = add_RUL_linear(df)    

    # 4)
    array = df.to_numpy()

    # 5)
    array = normalize_data(array, test)

    units = int(df['unit_number'].max())
    # 6) Apply sliding window on EACH engine unit, if train unit
    if not test:
        X = np.empty((1,window_size,19))
        y = np.empty((1,))
        for i in range(1, units + 1):
            idx = array[:,0] == i
            X_unit, y_unit = sliding_window(array[idx], window_size)
            X = np.concatenate((X, X_unit), axis=0)
            y = np.concatenate((y, y_unit), axis=0)
        # discard first elements (which are empty)
        X = X[1:]
        y = y[1:]
    # 6) OR take last samples of each unit, if test unit
    else:
        X = np.empty((1, window_size, 19))
        for i in range(1, units + 1):
            idx = array[:,0] == i
            X_unit = test_samples(array[idx], window_size)
            X = np.concatenate((X, X_unit), axis=0)
        # discard first elements (which are empty)
        X = X[1:]
        # add RUL column
        data_path = os.path.join(data_dir, validation_RUL_file)
        test_RUL = pd.read_csv(data_path, header=None, dtype='float')
        y = test_RUL.to_numpy()

    # 7) Remove unit_id, time and the three settings from data
    X = X[:,:,5:]

    return X, y


def split_timeseries_per_feature(data, n_features):
    # split input training data into separate time series, per feature
    data_split = []
    for i in range(n_features):
        data_feature = data[:,:, i].reshape(data.shape[0], data.shape[1], 1)
        data_split.append(data_feature)
    return data_split
