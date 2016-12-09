import numpy as np


def linearRegCostFunction(X, y, theta, lamda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = X.dot(theta)
    J = ((h - y) ** 2).sum() / 2 / m + (theta[1:] ** 2).sum() * lamda / 2 / m
    grad = X.T.dot(h - y) / m
    grad[1:] = grad[1:] + lamda / m * theta[1:]
    return J, grad.ravel()
