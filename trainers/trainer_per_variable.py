from models.cnn_per_variable import build_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from data_loader.data_prep import split_timeseries_per_feature

def train_per_variable(X, y, X_val, y_val, ckpt_path, log_dir, window_size):

    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # build model
    model = build_model(window_size, 14)

    # split input validation data into separate time series, per feature
    X = split_timeseries_per_feature(X, 14)
    # print(X_separated[-1][0])
    # print(y[0])
    X_val = split_timeseries_per_feature(X_val, 14)

    # train
    # model.load_weights(ckpt_path)
    # history = model.fit(X,
    #                     y,
    #                     validation_data=(X_val, y_val),
    #                     epochs=250,
    #                     batch_size=512,
    #                     verbose=1,
    #                     callbacks=[checkpoint, tensorboard_callback])
    history = []
    return model, history

    