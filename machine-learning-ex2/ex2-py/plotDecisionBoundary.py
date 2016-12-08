import numpy as np
import matplotlib.pyplot as plt

from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y, xlabel='', ylabel='', legends=[]):
    pos = y[:, 0] == 1
    neg = y[:, 0] == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+',
                label=legends[0])
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o',
                label=legends[1], alpha=0.5)

    if X.shape[1] <= 2:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 0]) - 2, max(X[:, 0]) + 2])
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, label=legends[2])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((u.size, v.size))
        for i in range(u.size):
            for j in range(v.size):
                z[i, j] = mapFeature(np.array(u[i]).reshape((1, 1)),
                                     np.array(v[j]).reshape((1, 1))).dot(theta)
        plt.contour(u, v, z.T, levels=[0.0], label=legends[2])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
