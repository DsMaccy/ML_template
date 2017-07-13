#!/usr/bin/python3
"""
This module compiles the keras model
"""

DEFAULT_EPOCHS = 100

def train(**args):
    """
    Provide "model", "x_train", and "y_train"
    """
    model = args["model"]
    x = args["x_train"]
    y = args["y_train"]
    if "epochs" in args:
        epochs = args["epochs"]
    else:
        epochs = DEFAULT_EPOCHS
    batch_size = args["batch_size"]

    model.fit(x, y, epochs=epochs, batch_size=batch_size)

def evaluate(**args):
    """
    Provide "model", "x_test", "y_test", and "batch_size" 
    """
    model = args["model"]
    x = args["x_test"]
    y = args["y_test"]
    batch_size = args["batch_size"]
    return model.evaluate(x, y, batch_size=batch_size)

def predict(**args):
    """
    Provide "model", and "x_test"
    """
    model = args["model"]
    x = args["x_test"]
    batch_size = args["batch_size"]
    return model.predict(x, batch_size=batch_size) 
