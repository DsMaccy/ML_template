#!/usr/bin/python3
"""
This module loads the training data and configures the input and output batches.
It also conglomerates the test data 
"""

import numpy as np
from setupModel import in_dim, out_dim


# This is the training data: used to train the Neural Net
x_train = np.zeros(in_dim) 
y_train = np.zeros(out_dim)

# This is the data that the model tries to solve w/o prior training
x_test = np.zeros(in_dim)
y_test = np.zeros(out_dim)
batch_size = 1
