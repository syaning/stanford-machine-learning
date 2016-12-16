import numpy as np
import matplotlib.pyplot as plt


def visualizeBoundary(X, y, model):
    pos = y[:, 0] == 1
    neg = y[:, 0] == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')

    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(this_X)

    plt.contour(X1, X2, vals, levels=[0])
    plt.show()
