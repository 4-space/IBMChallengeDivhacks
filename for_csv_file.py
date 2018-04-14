#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


def load_data(csv_filename):
    """
    Returns a numpy ndarray in which each row represents
    a MNIST digit and each column is the result for each of the 21 algorithms.

    """

    data = np.genfromtxt(csv_filename, delimiter=',', skip_header=1)

    data = data[:,:25]

    return data

def pc(td, th):

    number_predictions = 21
    correct_predictions = 0
    counter = 0

    avg_data = np.zeros((10, 2))

    zeros = np.zeros((60000, 1))

    td = np.append(td, zeros, axis=1)


    for row in td:
        current_row = td[counter,2:]
        correct_predictions = current_row.sum()
        accuracy = correct_predictions / 21

        if accuracy > th:
            label = 0
        else:
            label = 1

        td[counter,23] = label


        for i in range(10):
            if td[counter,1] == i:
                avg_data[i,0] += 1
                avg_data[i,1] += accuracy

        counter += 1

    return td
"""
def hist(counts):


    temp_array = np.zeros((10, 2))

    t_r_c = 0
    for row in counts:
        temp_array[t_r_c,0] = t_r_c
        temp_array[t_r_c,1] = t_r_c


        t_r_c += 1
    plt.hist(gaussian_numbers)


"""

if __name__ == "__main__":

    data = load_data('test.csv')

    ndarray = pc(data,0.5)

    print(ndarray)

    np.savetxt('test5.csv', ndarray, delimiter=',')
