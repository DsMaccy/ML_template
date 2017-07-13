#!/usr/bin/python3
"""
This module compiles the keras model
"""


from setupModel import model, saveModel, loadModel 
from setupData import x_train, y_train, x_test, y_test, batch_size
from train import train, evaluate, predict


if __name__ == "__main__":
    neuralNetData = {
        "model": model, 
        "x_train": x_train, 
        "y_train": y_train,
        "y_test": y_test,
        "x_test": x_test,
        "batch_size": batch_size,
        "epochs": 5 
    }
    train(**neuralNetData)
    print(evaluate(**neuralNetData))
    print(predict(**neuralNetData))

    saveModel(model)    


