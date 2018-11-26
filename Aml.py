import numpy as np
import keras.utils
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers import Conv2D,MaxPooling2D#use 2d convo layers because filter goes horizontal and vertical
from keras.activations import relu
import keras.losses
import os
import glob
import cv2
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD



train_folder = '/home/aditya/Desktop/Vidya-AML/testData'

train_set = []
train_y = []
for filename in os.listdir(train_folder):
	#female -1 and male - 0
	if(filename[-34] == '1'):
		train_y.append(1)
	else:
		train_y.append(0)
	filename = train_folder+'/'+filename
	train_set.append(cv2.imread(filename))

xTrain, xTest, yTrain, yTest = train_test_split(train_set, train_y, test_size = 0.2, random_state = 0)

epochs = 100
train_set = np.array(xTrain)
train_y  = np.array(yTrain)

test_set = np.array(xTest)
test_y  = np.array(yTest)


model = Sequential()


model.add(Conv2D(96,kernel_size=7,strides=(4,4),input_shape=(227,227,3),padding='valid',data_format="channels_last"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(256,kernel_size=5,padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(384,kernel_size=3,padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))
model.summary()



#model.compile(loss = keras.losses.binary_crossentropy,optimizer = keras.optimizers.Adam(),metrics=['accuracy'])

model.compile(loss = keras.losses.binary_crossentropy,optimizer = keras.optimizers.SGD(lr=0.002),metrics=['accuracy'])



train = model.fit(train_set,train_y,epochs = epochs,verbose = 1,validation_data = (test_set,test_y))

test = model.evaluate(test_set,test_y,verbose=0)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")









