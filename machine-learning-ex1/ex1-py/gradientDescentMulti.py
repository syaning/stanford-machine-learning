import numpy as np

from computeCostMulti import computeCostMulti


def gradientDescentMulti(X, y, theta, alpha, iters):
    m = y.shape[0]
    J_history = list(range(iters))
    for i in range(iters):
        theta = theta - (alpha / m) * X.T.dot(X.dot(theta) - y)
        J_history[i] = computeCostMulti(X, y, theta)
    return theta, J_history
