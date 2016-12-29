import numpy as np
import matplotlib.pyplot as plt
from math import isinf

from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    n = np.arange(0, 35, 0.5)
    X1, X2 = np.meshgrid(n, n)
    Z = multivariateGaussian(np.column_stack(
        (X1.T.flatten(), X2.T.flatten())), mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx', markersize=5)
    if not isinf(np.sum(Z)):
        plt.contour(X1, X2, Z, 10.0**np.arange(-20, 0, 3))
