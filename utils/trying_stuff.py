import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(10))

for window in dataset:
    print(window.numpy())