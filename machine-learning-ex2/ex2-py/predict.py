import numpy as np

from sigmoid import sigmoid


def predict(theta, X):
    theta = theta.reshape((theta.size, 1))
    p = sigmoid(X.dot(theta))
    return np.round(p)
