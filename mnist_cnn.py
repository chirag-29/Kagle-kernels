#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:14:16 2018

@author: chintanpatel
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


from keras.datasets import mnist

(X_train, y_train),(X_test,y_test) = mnist.load_data()


plt.imshow(X_train[0])

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)


#one hot encoding


from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_train[0]


model = Sequential()


model.add(Conv2D(64,kernel_size=3, activation = 'relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32,kernel_size=3, activation = 'relu'))


model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(10,activation = 'softmax'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,validation_split = 0.2, epochs = 3)

model.predict(X_test[:4])
y_test[:4]