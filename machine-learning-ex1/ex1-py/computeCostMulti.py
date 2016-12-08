import numpy as np


def computeCostMulti(X, y, theta):
    m = y.size
    J = np.power(X.dot(theta) - y, 2).sum() / 2 / m
    return J
