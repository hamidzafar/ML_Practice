import utils.dnn_app_utils_v2 as ut
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
from nn import NN

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    np.random.seed(1)
    train_x_orig, train_y, test_x_orig, test_y, classes = ut.load_data()

    index = 10
    # plt.imshow(train_x_orig[index])
    # print ("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
    # plt.show()

    # Explore your dataset
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],
                                           -1).T  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    G = [None, NN.relu, NN.sigmoid]
    G_p = [None, NN.relu_p, NN.sigmoid_p]
    h_u = [0, 7, 1]

    # G = [None, NN.relu, NN.relu, NN.sigmoid]
    # G_p = [None, NN.relu_p, NN.relu_p, NN.sigmoid_p]
    # h_u = [0, 7, 4, 1]

    # G = [None, NN.relu, NN.relu, NN.relu, NN.sigmoid]
    # G_p = [None, NN.relu_p, NN.relu_p, NN.relu_p, NN.sigmoid_p]
    # h_u = [0, 20, 7, 5, 1]

    model = NN(train_x, train_y, G, G_p, h_u, L=len(G) - 1)
    parameters = model.train(learnign_rate=0.0075, iteration=1000, lambd=.7)
    predictions = model.predict(parameters, train_x)
    print (
        'Accuracy: %d' % float(
            (np.dot(train_y, predictions.T) + np.dot(1 - train_y, 1 - predictions.T)) / float(
                train_y.size) * 100) + '%')
    predictions = model.predict(parameters, test_x)
    print (
        'Accuracy: %d' % float(
            (np.dot(test_y, predictions.T) + np.dot(1 - test_y, 1 - predictions.T)) / float(test_y.size) * 100) + '%')
