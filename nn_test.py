import utils.planar_utils as pu
import numpy as np
from nn import NN
import matplotlib.pyplot as plt


def show(x):
    img = x.reshape((16, 16))
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


def load_dataset():
    X = []
    Y = []
    with open("data/semeion.data", "r") as _file:
        lines = _file.readlines()
        for line in lines:
            X.append(map(float, line.split()[:256]))
            Y.append(map(float, line.split()[256:]))
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    np.random.seed(2)
    # X, Y = load_dataset()
    # X = X.T
    # Y = Y[:, 0].reshape((1, X.shape[1]))
    #
    # G = [None, np.tanh, NN.sigmoid]
    # G_p = [None, NN.tanh_p, NN.sigmoid_p]
    #
    # params = model(X_train, Y_train, G, G_p, learnign_rate=3.001, iteration=500)
    #
    # predictions = predict(params, X_train, G)
    # print (
    # 'Accuracy: %d' % float((np.dot(Y_train, predictions.T) + np.dot(1 - Y_train, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    #
    # predictions = predict(params, X_test, G)
    # print (
    # 'Accuracy: %d' % float((np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)) / float(Y.size) * 100) + '%')


    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = pu.load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    ### START CODE HERE ### (choose your dataset)
    dataset = "noisy_moons"
    ### END CODE HERE ###

    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2

    # X, Y = pu.load_planar_dataset()

    v = int(70 * X.shape[1] / 100)
    X_train = X[:, :v]
    Y_train = Y[:, :v]
    X_test = X[:, v:]
    Y_test = Y[:, v:]

    # Visualize the data
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
    # plt.show()


    G = [None, NN.relu, NN.sigmoid]
    G_p = [None, NN.relu_p, NN.sigmoid_p]
    h_u = [0, 5, 1]

    # G = [None,  NN.relu, NN.relu, NN.relu, NN.sigmoid]
    # G_p = [None,  NN.relu_p, NN.relu_p, NN.relu_p, NN.sigmoid_p]
    # h_u = [0, 10, 7, 5, 1]

    model = NN(X_train, Y_train, G, G_p, h_u, L=len(G) - 1)
    parameters = model.train(learnign_rate=.5, iteration=4000)

    # Plot the decision boundary

    pu.plot_decision_boundary(lambda x: model.predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    predictions = model.predict(parameters, X_train)
    print (
        'Accuracy: %d' % float(
            (np.dot(Y_train, predictions.T) + np.dot(1 - Y_train, 1 - predictions.T)) / float(
                Y_train.size) * 100) + '%')
    predictions = model.predict(parameters, X_test)
    print (
        'Accuracy: %d' % float(
            (np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)) / float(Y_test.size) * 100) + '%')
