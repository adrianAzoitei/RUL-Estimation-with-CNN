from models.cnn_per_variable import build_model, step_decay
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from data_loader.data_prep import split_timeseries_per_feature
import numpy as np

def train_per_variable(X, y, ckpt_path, log_dir, window_size, train=True):
# def train_per_variable(X, y, X_val, y_val, ckpt_path, log_dir, window_size, train=True):
    # prepare callbacks for training
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    lrate = LearningRateScheduler(step_decay)

    # build model
    n_features = len(X[1,1,:])
    model = build_model(window_size, n_features)
    # split input validation training into separate time series, per feature
    X = split_timeseries_per_feature(X, n_features)
    # X_val = split_timeseries_per_feature(X_val, n_features)
    if train:
        history = model.fit(X,
                            y,
                            validation_split=0.05,
                            # validation_data= (X_val, y_val),
                            epochs=250,
                            batch_size=512,
                            verbose=2,
                            shuffle=True,
                            callbacks=[checkpoint, tensorboard_callback, lrate])
        return model, history
    else:
        model.load_weights(ckpt_path)
        return model

    