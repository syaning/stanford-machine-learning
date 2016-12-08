import numpy as np

from sigmoid import sigmoid


def costFunction(theta, X, y):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(X.dot(theta))
    J = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = X.T.dot(h - y) / m
    return J, grad.ravel()
