#!/usr/bin/python3
"""
This file sets up the model for the neural network using keras and compiles it
"""

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
import json


print("Initializing Model")

model = Sequential()

print("Setting Up Model")

in_dim = (1, 100)
out_dim = (1, 10)
hid_dim = (1, 64)

# Setup first layer (inputs -> hidden layer)
layer1 = Dense(units=hid_dim[1], input_dim=in_dim[1])
model.add(layer1)
model.add(Activation('relu'))

# Setup second layer (hidden -> output layer)
layer2 = Dense(units=out_dim[1], input_dim=hid_dim[1])
model.add(layer2)
model.add(Activation('softmax'))

# print(layer1.get_config())
# print(layer2.get_config())

print("Compiling Model")

model.compile(loss=categorical_crossentropy, 
        optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))




ARCH_FILE_EXTENSION = ".arch"
WEIGHTS_FILE_EXTENSION = ".wei"

def saveModel(model, fileName="modelConfig"):
    """
    This save the model as two separate files: 
        The """ + ARCH_FILE_EXTENSION + """ file encodes the architecture 
        The """ + WEIGHTS_FILE_EXTENSION + """  file encodes the weights
    """
    with open(fileName + ARCH_FILE_EXTENSION, "w") as outfile:
        json.dump(model.to_json(), outfile)

    
    model.save_weights(fileName + WEIGHTS_FILE_EXTENSION)


def loadModel(fileName="modelConfig"):
    """
    This loads the model from the .arc and .wei files:
        The """ + ARCH_FILE_EXTENSION + """ file encodes the architecture 
        The """ + WEIGHTS_FILE_EXTENSION + """  file encodes the weights
    """
    global model
    with open(fileName + ARCH_FILE_EXTENSION, "w") as infile:
        model = model_from_json( json.load(infile) )

    model.load_weights(fileName + WEIGHTS_FILE_EXTENSION, by_name=False)
