import matplotlib.pyplot as plt


def plotData(X, y):
    pos = y[:, 0] == 1
    neg = y[:, 0] == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o')
    plt.show()
