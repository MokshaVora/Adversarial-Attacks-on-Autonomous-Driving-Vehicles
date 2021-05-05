# keras imports for the dataset and building our neural network
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import keras
import tensorflow as tf

# loading the dataset
classes = 4

# path to traffic sign dataset
cur_path = '/Dataset'


X, y = [], []
#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path, str(i))
    images = os.listdir(path)
    
    for a in images:
        try:
            temp = os.path.join(path, a)
            image = tf.keras.preprocessing.image.load_img(temp, grayscale=False, color_mode='rgb', target_size=(30, 30, 3), interpolation='nearest')
            image = keras.preprocessing.image.img_to_array(image)
            image = np.array(image)
            X.append(image)
            y.append(i)
        except Exception as e:
            print(e)

X_tra, X_tes, y_tra, y_tes = train_test_split(X, y, test_size = 0.2, shuffle=True)

X_train = np.array(X_tra)
y_train = np.array(y_tra)
X_test = np.array(X_tes)
y_test = y_tes
# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], 30, 30, 3)
X_test = X_test.reshape(X_test.shape[0], 30, 30, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 4 
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(30, 30, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(4, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

start = time.time()
# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=4, epochs=10, validation_data=(X_test, Y_test))
end = time.time()

print(end-start)

model.save('model.h5')

