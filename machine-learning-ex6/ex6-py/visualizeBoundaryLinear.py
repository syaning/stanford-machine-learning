import numpy as np
import matplotlib.pyplot as plt


def visualizeBoundaryLinear(X, y, model):
    w = model.coef_.ravel()
    b = model.intercept_.ravel()
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0] * xp + b) / w[1]

    pos = y[:, 0] == 1
    neg = y[:, 0] == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')
    plt.plot(xp, yp)
    plt.show()
