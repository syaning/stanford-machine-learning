import numpy as np

from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros((m, 1))
    h1 = sigmoid(np.hstack((np.ones((m, 1)), X)).dot(Theta1.T))
    h2 = sigmoid(np.hstack((np.ones((m, 1)), h1)).dot(Theta2.T))
    p = np.argmax(h2, axis=1) + 1
    return p.reshape((p.size, 1))
