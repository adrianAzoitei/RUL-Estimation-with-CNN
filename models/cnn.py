import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

opt = Adam(learning_rate=0.001)

def build_model(window_size):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv1D(filters=10, kernel_size=10,
                                        strides=1, padding="same",
                                        activation="tanh",
                                        input_shape=(window_size, 14)),
                tf.keras.layers.Conv1D(filters=10, kernel_size=10,
                                        strides=1, padding="same",
                                        activation="tanh"),
                tf.keras.layers.Conv1D(filters=10, kernel_size=10,
                                        strides=1, padding="same",
                                        activation="tanh"),
                tf.keras.layers.Conv1D(filters=10, kernel_size=10,
                                        strides=1, padding="same",
                                        activation="tanh"),
                tf.keras.layers.Conv1D(filters=1, kernel_size=3,
                                        strides=1, padding="same",
                                        activation="tanh"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(100, activation="tanh"),
                tf.keras.layers.Dense(1)])
        model.compile(optimizer=opt,
                loss=root_mean_squared_error)
        model.summary()
        return model


