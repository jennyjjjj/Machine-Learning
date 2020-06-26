import numpy
import pandas
from PIL import Image
from keras import backend as K
from keras.utils import np_utils

def load_data(dataset_path):
img = Image.open(dataset_path)
img_ndarray = numpy.asarray(img, dtype='float64') / 256
print(img_ndarray.shape)
faces = numpy.empty((400,57,47))
for row in range(20):
for column in range(20):
faces[row * 20 + column] = img_ndarray[row * 57:(row + 1) * 57, column * 47:(column + 1) * 47]

label = numpy.empty(400)
for i in range(40):
label[i * 10:i * 10 + 10] = i
label = label.astype(numpy.int)
label = np_utils.to_categorical(label, 40)

train_data = numpy.empty((320, 57,47))
train_label = numpy.empty((320,40))
valid_data = numpy.empty((40, 57,47))
valid_label = numpy.empty((40,40))
test_data = numpy.empty((40, 57,47))
test_label = numpy.empty((40,40))

train_data[i * 8:i * 8 + 8] = faces[i * 10:i * 10 + 8]
train_label[i * 8:i * 8 + 8] = label[i * 10:i * 10 + 8]
valid_data[i] = faces[i * 10 + 8]
valid_label[i] = label[i * 10 + 8]
test_data[i] = faces[i * 10 + 9]
test_label[i] = label[i * 10 + 9]

return [(train_data, train_label), (valid_data, valid_label),(test_data, test_label)]

if __name__ == '__main__':
[(train_data, train_label), (valid_data, valid_label), (test_data, test_label)] = load_data('olivettifaces.gif')
oneimg = train_data[0]*256
print(oneimg)
im = Image.fromarray(oneimg)
im.show()

#CNN

import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from PIL import Image
import FaceData

batch_size = 128
nb_classes = 40
epochs = 600
img_rows, img_cols = 57, 47
nb_filters = 32
pool_size = (2, 2)
kernel_size = (5, 5)
input_shape = (img_rows, img_cols,1)
[(X_train, Y_train), (X_valid, Y_valid),(X_test, Y_test)] =FaceData.load_data('olivettifaces.gif')

X_train=X_train[:,:,:,np.newaxis]
X_valid=X_valid[:,:,:,np.newaxis]
X_test=X_test[:,:,:,np.newaxis]

model = Sequential()
model.add(Conv2D(6,kernel_size,input_shape=input_shape,strides=1))
model.add(AveragePooling2D(pool_size=pool_size,strides=2))
model.add(Conv2D(12,kernel_size,strides=1))
model.add(AveragePooling2D(pool_size=pool_size,strides=2))
model.add(Flatten())
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
for i in range(len(y_pred)):
oneimg = X_test[i,:,:,0]*256
im = Image.fromarray(oneimg)
im.show()















