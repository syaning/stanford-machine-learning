import numpy as np

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = np.hstack((np.ones((m, 1)), sigmoid(a1.dot(Theta1.T))))
    a3 = sigmoid(a2.dot(Theta2.T))
    p = np.argmax(a3, axis=1) + 1
    return p.reshape((p.size, 1))
