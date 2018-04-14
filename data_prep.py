import numpy as np
import numpy as np
from matplotlib import pyplot as plt


def load_data(csv_filename, skip_header=1):
    """
    Returns a numpy ndarray in which each row represents
    a MNIST digit and each column is the result for each of the 21 algorithms.
    """

    data = np.genfromtxt(csv_filename, delimiter=',', skip_header=skip_header)

    data = data[:,:25]

    return data

def pc(td, th):

    number_predictions = 21
    correct_predictions = 0
    counter = 0

    avg_data = np.zeros((10, 2))

    zeros = np.zeros((60000, 1))

    print(td.shape)
    print(zeros.shape)
    
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


if __name__ == "__main__":

    data = load_data('MNIST_train.csv')

    ndarray = pc(data,0.80)

    #print(ndarray)

    np.savetxt('train_81.csv', ndarray, delimiter=',')