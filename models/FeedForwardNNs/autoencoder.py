#!/usr/bin/python3

"""
The keras model for an autoencoder feedforward neural net (INCOMPLETE).  The autoencoder neural net is generally used for dimensionality reduction and decoding an encoded sequence
"""



from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.losses import mean_squared_error
from keras.optimizers import SGD

def getNumLayers():
    # TODO: change to the correct #
    return 0

def genModel(layerSizes, activationFunctions, 
        optimizer=SGD(), 
        lossFunction=mean_squared_error):
    if (len(layerSizes) != len(activationFunctions) + 1):
        raise RuntimeError("The number of Layers should be 1 greater " +
            "than the number of activation functions")

    if len(layerSizes) % 2 == 0:
            raise RuntimeError("There needs to be an odd number of layers")

    for index in range(len(layerSizes)): 
        if (layerSizes[index] != layerSizes[-(index+1)]):
            raise RuntimeError("There number of nodes per layer need to be " +
                "symmetrical along the middle layer of the neural network")

    model = Sequential()

    # TODO: make use of layerSize input
    for index in range(len(layerSizes) - 1): 
        model.add( Dense(layerSizes[index+1], 
                input_dim=layerSizes[index]) )
                #input_shape=(1, layerSizes[index])) )
        model.add( Activation(activationFunctions[index]) )

    # TODO: Compile
    model.compile(optimizer, lossFunction)

    return model
