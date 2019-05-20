# with open("/data/fer2013.csv") as f:
# content = f.readlines()
 
# lines = np.array(content)
 
# num_of_instances = lines.size
# print("number of instances: ",num_of_instances)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

X = x / 255.0

model = Sequential()
model.add(Conv2D(645), (3,3), input_shape = X.shape[1:])
