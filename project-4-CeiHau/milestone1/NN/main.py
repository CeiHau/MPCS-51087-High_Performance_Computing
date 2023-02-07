import pickle
import gzip
import numpy as np
import pandas as pd
import neural_network

def one_hot_encode(y):
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded

def load_data(filename):
    data = pd.read_csv(filename)
    y = data['label']
    y = np.array([one_hot_encode(y) for y in y])
    # normalize the data from 0-255 to 0-1
    x = data.drop(columns='label').to_numpy().reshape((len(data), 784,1))/255
    data = [[x, y] for (x, y) in zip(x, y)]
    return data

if __name__ == '__main__':

    training_data = load_data('mnist_data/mnist_train.csv')
    testing_data = load_data('mnist_data/mnist_test.csv')
    net = neural_network.NeuralNetwork([784, 20,20, 10])

    net.train(list(training_data), 10, 10, 3.0, list(testing_data))

