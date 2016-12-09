import numpy as np
import matplotlib.pyplot as plt

from polyFeatures import polyFeatures


def plotFit(min_x, max_x, mu, sigma, theta, p):
    theta = theta.reshape((theta.size, 1))
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    x = x.reshape(x.size, 1)

    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    plt.plot(x, X_poly.dot(theta), linestyle='--')
