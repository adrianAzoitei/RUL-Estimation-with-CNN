import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from models.cnn import build_model

def train(X, y, X_val, y_val, ckpt_path, log_dir, window_size):
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    model = build_model(window_size)
    # CNNmodel.load_weights(ckpt_path)
    # train
    history = model.fit(X,
                        y,
                        validation_data=(X_val, y_val),
                        epochs=250,
                        batch_size=512,
                        verbose=1,
                        callbacks=[checkpoint, tensorboard_callback])
    return model