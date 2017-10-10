import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def logistic_gradient(X, y, theta):
    # FORWARD PROPAGATION
    z = X.dot(theta)
    a = sigmoid(z)

    # BACKWARD PROPAGATION
    dz = y - a
    w = dz.dot(X)
    # w = w / N
    return w


def gradient_descent_runner(learning_rate, num_iterations, gradient_func, X, Y, theta):
    for i in range(num_iterations):
        theta = theta + learning_rate * gradient_func(X, Y, theta)
    return theta


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in xrange(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        # if step % 10000 == 0:
        #     print log_likelihood(features, target, weights)

    return weights


g_theta = np.array([0, 0, 0])

if __name__ == '__main__':
    np.random.seed(12)
    num_observations = 5000

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_labels = np.hstack((np.zeros(num_observations),
                                  np.ones(num_observations)))
    N = len(x1)

    data = np.genfromtxt("data/2d_logistic.csv", delimiter=",")
    simulated_separableish_features = data[:, 0:2]
    simulated_labels = data[:, 2]
    N = len(data)


    plt.subplot(411)
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c=simulated_labels, alpha=.4)

    weights = logistic_regression(simulated_separableish_features, simulated_labels,
                                  num_steps=300000, learning_rate=0.00001, add_intercept=True)
    print weights

    features = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)), simulated_separableish_features))
    y_hat = np.dot(features, weights) > 0
    pos = np.where(y_hat == 0)
    neg = np.where(y_hat == 1)

    plt.subplot(412)
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c=y_hat, alpha=.4)

    print 'Accuracy from scratch: {0}'.format((y_hat == simulated_labels).sum().astype(float) / len(y_hat))

    theta = gradient_descent_runner(0.00001, 300000, logistic_gradient, features, y_hat,
                                    g_theta)
    print theta

    y_hat = np.dot(features, weights) > 0
    pos = np.where(y_hat == 0)
    neg = np.where(y_hat == 1)

    plt.subplot(413)
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c=y_hat, alpha=.4)

    print 'Accuracy from scratch: {0}'.format((y_hat == simulated_labels).sum().astype(float) / len(y_hat))

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(fit_intercept=True, C=1e15)
    clf.fit(simulated_separableish_features, simulated_labels)

    print clf.intercept_, clf.coef_

    y_hat = (clf.coef_.dot(simulated_separableish_features.T) + clf.intercept_) > 0
    pos = np.where(y_hat == 0)
    neg = np.where(y_hat == 1)

    plt.subplot(414)
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
                c=y_hat, alpha=.4)

    print 'Accuracy from sk-learn: {0}'.format(clf.score(simulated_separableish_features, simulated_labels))

    plt.show()
