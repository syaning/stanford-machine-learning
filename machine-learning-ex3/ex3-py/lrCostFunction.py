import numpy as np

from sigmoid import sigmoid


def lrCostFunction(theta, X, y, lamda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(X.dot(theta))
    J = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    J += np.square(theta[1:]).sum() * lamda / 2 / m
    grad = X.T.dot(h - y) / m
    grad[1:] = grad[1:] + lamda * theta[1:] / m
    return J, grad.ravel()
