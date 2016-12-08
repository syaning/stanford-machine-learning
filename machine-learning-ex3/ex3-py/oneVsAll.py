import numpy as np
from scipy.optimize import fmin_cg

from sigmoid import sigmoid
from lrCostFunction import lrCostFunction


def oneVsAll(X, y, num_labels, lamda):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))

    cost_function = lambda p, y: lrCostFunction(p, X, y, lamda)[0]
    grad_function = lambda p, y: lrCostFunction(p, X, y, lamda)[1]

    for i in range(1, num_labels + 1):
        initial_theta = np.zeros(n + 1)
        all_theta[i - 1, :] = fmin_cg(cost_function, initial_theta, fprime=grad_function,
                                      args=((y == i).astype(int),), maxiter=100, disp=False)
        print('Finish oneVsAll checking number: %d' % i)

    return all_theta
