import numpy as np
from scipy.optimize import fmin_cg

from linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, lamda):
    initial_theta = np.zeros((X.shape[1], 1))

    cost_function = lambda theta: linearRegCostFunction(X, y, theta, lamda)[0]
    grad_function = lambda theta: linearRegCostFunction(X, y, theta, lamda)[1]

    theta = fmin_cg(cost_function, initial_theta, fprime=grad_function,
                    maxiter=50, disp=False)

    return theta
