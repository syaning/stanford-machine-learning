import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)

    for i in range(lambda_vec.size):
        lamda = lambda_vec[i]
        theta = trainLinearReg(X, y, lamda)
        error_train[i], _ = linearRegCostFunction(X, y, theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)

    return lambda_vec, error_train, error_val
