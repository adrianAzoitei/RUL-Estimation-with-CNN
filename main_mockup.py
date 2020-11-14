import numpy as np
import pandas as pd
import tensorflow as tf

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

#import defined variables and methods
from utils import DATA_DIR
from data_loader.data_prep import prepare_sub_dataset, add_RUL_linear, add_RUL_piecewise, normalize_data, sliding_window
from models.cnn import model

[X, y] = prepare_sub_dataset(DATA_DIR, 'mock-up.csv')

# split into train and test data (70, 30 %)
split = int(len(X) * 0.7)
X_train = X[:split]
y_train = y[:split]
print(X_train.shape)
print(X_train)
X_val = X[split:-100]
y_val = y[split:-100]

X_test = X[-100:-1]
X_test = X_test.reshape(len(X_test), 3, 5)
y_test = y[-100:-1]
print(X_test.shape)
print(X_test)

# train
history = model.fit(X_train,
                    y_train,
                    validation_data= (X_val, y_val),
                    epochs=100,
                    batch_size=15,
                    verbose=1)


# Plot the chart for accuracy and loss on both training and validation
import matplotlib.pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

prediction = model.predict(X_test)
print(val_loss)
