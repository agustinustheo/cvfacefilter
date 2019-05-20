import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),  (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

print(x_train[0])