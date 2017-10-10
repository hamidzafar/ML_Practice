import numpy as np
import utils.dnn_app_utils_v2 as util


class NN:
    @staticmethod
    def sigmoid(Z):
        return 1.0 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_p(A, Z):
        # A == NN.sigmoid(Z)
        return A * (1 - A)

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_p(A, Z):
        return 1. * (Z > 0)

    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_p(A, Z):
        # A ==  NN.tanh(Z)
        return 1 - np.power(A, 2)

    def __init__(self, X, Y, G, G_p, h_u, L=2):
        self.X = X
        self.Y = Y
        self.G = G
        self.G_p = G_p
        self.h_u = h_u
        self.L = L

    def __init_params(self):
        n_x = self.X.shape[0]
        m = self.X.shape[1]
        assert (m == self.Y.shape[1])

        cache = {
            "m": m,
            "L": self.L
        }
        self.h_u[0] = n_x
        for i in range(1, self.L + 1):
            i_str = str(i)
            cache["W" + i_str] = \
                np.random.randn(self.h_u[i], self.h_u[i - 1]) * 0.01
            # np.random.randn(h_u[i], h_u[i - 1]) / np.sqrt(h_u[i - 1])
            cache["b" + i_str] = np.zeros((self.h_u[i], 1))

        return cache

    def __forward(self, params, X=None):
        if X is None:
            X = self.X

        cache = {"A0": X}
        for idx in range(1, params["L"] + 1):
            i = str(idx)
            i_1 = str(idx - 1)
            Z = np.dot(params["W" + i], cache["A" + i_1]) + params["b" + i]
            cache["Z" + i] = Z
            cache["A" + i] = self.G[idx](Z)
        return cache

    def __cost(self, params, cache, lambd=None):
        regularizer = 0
        if lambd is not None:
            regularizer = np.sum([np.sum(params["W" + str(i + 1)]) for i in range(params["L"])]) * (
            lambd / (2.0 * params["m"]))
        l_str = str(params["L"])
        return np.squeeze(np.sum(np.multiply(np.log(cache["A" + l_str]), self.Y) +
                                 np.multiply(np.log(1 - cache["A" + l_str]), 1 - self.Y)) / -params["m"]) \
               + regularizer

    def __backward(self, params, cache, lambd=None):
        l_str = str(params["L"])
        A = cache["A" + l_str]
        cache["dA" + l_str] = - (np.divide(self.Y, A) - np.divide(1 - self.Y, 1 - A))
        m = params["m"]
        for idx in range(params["L"], 0, -1):
            i = str(idx)
            i_1 = str(idx - 1)
            dZ = cache["dA" + i] * self.G_p[idx](cache["A" + i], cache["Z" + i])
            regularizer = 0
            if lambd is not None:
                regularizer = ((lambd / m) * params["W" + i])
            cache["dW" + i] = np.dot(dZ, cache["A" + i_1].T) / params["m"] + regularizer
            cache["db" + i] = np.sum(dZ, axis=1, keepdims=True) / params["m"]
            cache["dA" + i_1] = np.dot(params["W" + i].T, dZ)
            cache["dZ" + i] = dZ

        return cache

    def train(self, learnign_rate=1.2, iteration=100, lambd=None):
        params = self.__init_params()
        for i in range(iteration):
            cache = self.__forward(params)
            gradients = self.__backward(params, cache, lambd)
            for l in range(1, params["L"] + 1):
                l_str = str(l)
                params["W" + l_str] = params["W" + l_str] - learnign_rate * gradients["dW" + l_str]
                params["b" + l_str] = params["b" + l_str] - learnign_rate * gradients["db" + l_str]

            if i % 100 == 0:
                print self.__cost(params, cache, lambd)

        return params

    def predict(self, params, X=None):
        if X is None:
            X = self.X
        l_str = str(params["L"])
        cache = self.__forward(params, X)
        return cache["A" + l_str] > 0.5
