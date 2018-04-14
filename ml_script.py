#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

def load_data(csv_filename):
    """
    Returns a numpy ndarray in which each row represents
    a MNIST digit and each column is the result for each of the 21 algorithms.

    """

    data = np.genfromtxt(csv_filename, skip_header=1)

    data = data[:,:]

    return data

def pc(training_data):

    number_predictions = 21
    correct_predictions = 0

    for row in training_data:

        correct_predictions = row.sum()
        row(23) = correct_predictions


if __name__ == "__main__":

    data = load_data('MNIST_train.csv')

    ndarray = pc(load_data(data))

    print(ndarray)
