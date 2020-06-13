#IMPORT PACKAGES
import tensorflow as tf
import keras
keras.__version__
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.utils import to_categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#LOAD THE DATASET
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#NORMIZE THE IMAGES. Normalize the pixel values from [0,255] to [-0.5,0.5] to make the network easier to train.
train_images= (train_images/255)-0.5
test_images= (test_images/225)-0.5

#FLATTEN THE IMAGES. Flatten each 28*28 images into a 28*28=784 dimensional vector to pass into the neural network.
train_images= train_images.reshape((-1,784))
test_images= test_images.reshape((-1,784))

#PRINT THE SHAPE OF IMAGES.
print(train_images.shape)
print(test_images.shape)

#BUILD THE MODEL WITH 3 LAYERS
#2 layers with 64 neurons and the relu function
#1 layer with 10 neurons and softmax function
model= Sequential()
model.add(Dense(64, activation='relu',input_dim=784))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#COMPILE THE MODEL
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#TRAIN THE MODEL
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=10,
    batch_size=64
)

#EVALUATE THE MODEL
test_loss, test_acc = model.evaluate(test_images, to_categorical(test_labels))
print('test_acc:', test_acc)

#PREDICT ON THE FIRST 5 TEST IMAGES
predictions= model.predict(test_images[:5])

#PRINT MODELS PREDICTION
print(np.argmax(predictions, axis=1))
print(test_labels[:5])

#VISUALIZE THE IMAGES
for i in range(0,5):
    first_image= test_images[i]
    first_image= np.array(first_image, dtype='float')
    pixels= first_image.reshape((28,28))
    plt.imshow(pixels,cmap='gray')
    plt.show()




