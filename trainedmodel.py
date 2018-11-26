from keras.models import model_from_json
import keras.losses
from keras.optimizers import SGD
import os 
import cv2
import numpy as np

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


directory = '/home/vidya/Desktop/aml/resize'

TEST = []
TESTY = [0,0,0,1,0,1,1,1,0,0,1,0,0]
for filename in os.listdir(directory):
	filename = directory+'/'+filename
	TEST.append(cv2.imread(filename))
	


TEST = np.array(TEST)
TESTY = np.array(TESTY)

# evaluate loaded model on test data
loaded_model.compile(loss = keras.losses.binary_crossentropy,optimizer = keras.optimizers.SGD(lr=0.002),metrics=['accuracy'])
test = loaded_model.evaluate(TEST,TESTY,verbose=0)

print("evaluate",test)
