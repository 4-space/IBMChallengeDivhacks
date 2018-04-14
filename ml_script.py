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


    for row in td:
        current_row = td[counter,2:]
        correct_predictions = current_row.sum()
        accuracy = correct_predictions / 21
        ##td[counter,23] = accuracy ##store accuracy


        for i in range(10):
            if td[counter,1] == i:
                avg_data[i,0] += 1

                if accuracy < th:

                    avg_data[i,1] += 1 ##adds how many data poins are "hard"

        counter += 1

    return avg_data

def graph(modified_file):

    hist_array = np.zeros((10, 2))

    array_counter1 = 0
    for row in modified_file:

        hist_array[array_counter1,0] = array_counter1
        hist_array[array_counter1,1] = (modified_file[array_counter1,1]) / (modified_file[array_counter1,0])
        array_counter1 += 1

    objects = (hist_array[:,0]).tolist()
    y_pos = np.arange(len(objects))

    performance = (hist_array[:,1]).tolist()

    print((hist_array[:,1]))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.title("% of digits that are classfied as HARD when T=0.5")
    plt.xlabel("Digit")
    plt.ylabel("Percentage")
    plt.show()




if __name__ == "__main__":

    data = load_data('MNIST_train.csv')

    ndarray = pc(data,0.8)

    print(ndarray)

    graph(ndarray)
    ##np.savetxt('test1.csv', ndarray, delimiter=',')
