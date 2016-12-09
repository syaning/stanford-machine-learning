import numpy as np

from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def learningCurve(X, y, Xval, yval, lamda):
    m = y.size
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        Xtrain = X[:i + 1, :]
        ytrain = y[:i + 1, :]
        theta = trainLinearReg(Xtrain, ytrain, lamda)
        error_train[i], _ = linearRegCostFunction(Xtrain, ytrain, theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)

    return error_train, error_val
