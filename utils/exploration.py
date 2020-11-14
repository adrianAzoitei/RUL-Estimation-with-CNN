# import visualization and numerical packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import system and file handling packages
import os
import sys

# set path to working directory
path = os.getcwd()
rootpath = os.path.join(path, os.pardir)
roothpath = os.path.abspath(rootpath)

# # if in interactive (jupyter) mode
# sys.path.insert(0, rootpath)

# if in local (terminal) mode
sys.path.insert(0, path)

#import defined variables
from utils import DATA_DIR

from data_loader.read_data import read_data
from data_loader.data_prep import add_RUL_linear, add_RUL_piecewise, df_to_array

# call read_data fn to read data from file
train_FD004 = read_data(DATA_DIR, 'train_FD004.txt')

# add RUL column
train_FD004 = add_RUL_linear(train_FD004)

# known faults present in the HPC module
plt.title('Static pressure at HPC outlet (Ps30)')
legend = []
legendNames = ['Unit 1', 'Unit 50', 'Unit 100']
units = [1, 50, 100]

# Plot behaviour of HPC outlet temperature w.r.t. time for some engines
plt.figure(1)
for i in units:
    unit,  = plt.plot(train_FD004[train_FD004['unit_number']==i]['time_in_cycles'].values,
        train_FD004[train_FD004['unit_number']==i]['Ps30'].values, )
    legend.append(unit)
plt.xlabel('Time in cycles')
plt.legend(legend, legendNames)


# find correlations
plt.figure(2)
sns.heatmap(train_FD004.corr(), annot=True, cmap='RdYlGn')

# scatter plots to see operating condition clusters in FD002 and FD004
ax = plt.figure(3).add_subplot(projection='3d')
ax.scatter(train_FD004['altitude'].values,
            train_FD004['MachNo'].values,
            train_FD004['TRA'].values)

ax.set_xlabel('altitude')
ax.set_ylabel('MachNo')
ax.set_zlabel('TRA')

# plot RUL of some engine against time to check function
plt.figure(4)
i = 1 # engine number
plt.plot(train_FD004[train_FD004['unit_number']==i]['time_in_cycles'].values,
         train_FD004[train_FD004['unit_number']==i]['RUL'].values)
plt.xlabel('Time in cycles')
plt.ylabel('RUL of engine {}'.format(i))

plt.show()

