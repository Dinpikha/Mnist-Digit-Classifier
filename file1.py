from tensorflow.keras.datasets import mnist
import pandas as pd
import seaborn as sns

(x,y),(X,Y)=mnist.load_data()
# print(x)

import matplotlib.pyplot as plt


x = x.reshape(-1, 28, 28, 1) / 255.0
X = X.reshape(-1, 28, 28, 1) / 255.0

x_train=x
y_train=y

from keras import layers 
import keras
from keras import layers
model = keras.Sequential()
model.add(keras.Input(shape=(28,28,1)))
model.add(layers.Conv2D(32,(3,3),activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

# ans=model.summary()
# print(ans)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=10,validation_split=0.5)

model.save("mnist_model.keras")

import pickle

with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)
