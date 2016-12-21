import numpy as np
import matplotlib.pyplot as plt


def plotDataPoints(X, idx):
    cmap = plt.get_cmap('jet')
    idxn = idx.ravel() / np.max(idx)
    colors = cmap(idxn)
    plt.scatter(X[:, 0], X[:, 1], 15, edgecolors=colors,
                marker='o', facecolors='none', lw=0.5)
