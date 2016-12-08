import matplotlib.pyplot as plt


def plotData(X, y, xlabel='', ylabel='', legends=[]):
    pos = y[:, 0] == 1
    neg = y[:, 0] == 0
    plt.scatter(X[pos, 0], X[pos, 1], c='k', marker='+',
                label=legends[0])
    plt.scatter(X[neg, 0], X[neg, 1], c='y', marker='o',
                label=legends[1], alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
