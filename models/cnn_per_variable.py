import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.initializers import GlorotNormal

# COPYRIGHT
# https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
opt = Adam(learning_rate=0.001)

def build_model(n_steps, n_features):
    cnns = []
    inputs = []
    for i in range(1, n_features + 1):
        inp = Input(shape=(n_steps, 1))
        inputs.append(inp)
        conv1 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation="tanh", name='conv1.{}'.format(i))(inp)
        conv2 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation="tanh", name='conv2.{}'.format(i))(conv1)
        conv3 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation="tanh", name='conv3.{}'.format(i))(conv2)
        conv4 = Conv1D(filters=10, kernel_size=10,
                                    strides=1, padding="same",
                                    activation="tanh", name='conv4.{}'.format(i))(conv3)
        conv5 = Conv1D(filters=1, kernel_size=3,
                                    strides=1, padding="same",
                                    activation="tanh", name='conv5.{}'.format(i))(conv4)
        cnns.append(conv5)
    concat = concatenate(cnns)
    flatten = Flatten()(concat)
    dropout = Dropout(0.5)(flatten)
    dense = Dense(100, activation="tanh")(flatten)
    output = Dense(1)(dense)          
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=opt,
                loss=root_mean_squared_error)
    model.summary()
    return model