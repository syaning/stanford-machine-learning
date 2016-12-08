import numpy as np

from sigmoid import sigmoid


def costFunctionReg(theta, X, y, lamda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(X.dot(theta))
    J = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    J -= (theta[1:, :]**2).sum() / lamda / 2 / m
    grad = X.T.dot(h - y) / m + lamda / m * theta
    grad[0] = grad[0] - lamda / m * theta[0]
    return J, grad.ravel()
