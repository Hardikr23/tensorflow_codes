import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist_data = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist_data.load_data()

x_train = x_train/255
x_test = x_test/255

model = keras.Sequential([
                          keras.layers.Flatten(),
                          keras.layers.Dense(128, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=["accuracy"])

class my_call_back(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epochs, logs={}):
    if(logs.get("accuracy") > 0.98):
      print("\n80% Acc reached. Stopping Training\n")
      self.model.stop_training = True

callback = my_call_back()
model.fit(x_train, y_train, epochs=5, callbacks=[callback])

model.evaluate(x_test, y_test)