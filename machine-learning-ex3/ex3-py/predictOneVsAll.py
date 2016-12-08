import numpy as np

from sigmoid import sigmoid


def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    p = np.argmax(sigmoid(X.dot(all_theta.T)), axis=1) + 1
    return p.reshape((p.size, 1))
