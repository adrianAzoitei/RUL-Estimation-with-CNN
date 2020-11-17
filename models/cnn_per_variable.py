import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, schedules
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def step_decay(epoch):
   initial_lrate = 0.001
   if epoch > 200:
       lrate = 0.0001
   else:
       lrate = initial_lrate
   return lrate

def build_model(n_steps, n_features):
    cnns = []
    inputs = []
    for i in range(1, n_features + 1):
        inp = Input(shape=(n_steps, 1))
        inputs.append(inp)
        conv1 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation='tanh', name='conv1.{}'.format(i))(inp)
        conv2 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation='tanh', name='conv2.{}'.format(i))(conv1)
        conv3 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation='tanh', name='conv3.{}'.format(i))(conv2)
        conv4 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation='tanh', name='conv4.{}'.format(i))(conv3)
        conv5 = Conv1D(filters=1, kernel_size=3,
                                    strides=1, padding="same",
                                    activation='tanh', name='conv5.{}'.format(i))(conv4)
        cnns.append(conv5)
    concat = concatenate(cnns)
    flatten = Flatten()(concat)
    dropout = Dropout(0.5)(flatten)
    dense = Dense(100, activation='tanh')(dropout)
    output = Dense(1)(dense) 

    model = Model(inputs=inputs, outputs=output)
    model.summary()
    model.compile(optimizer='adam',
                  loss='mse')
    model.summary()
    return model