import numpy as np


# def computeNumericalGradient(J, theta):
def computeNumericalGradient(J, X, Theta):
    theta = np.hstack((X.T.flatten(), Theta.T.flatten()))
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for i in range(theta.size):
        perturb[i] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        numgrad[i] = (loss2 - loss1) / (2 * e)
        perturb[i] = 0
    return numgrad
