import numpy as np


def computeCentroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i] = np.mean(X[idx[:, 0] == i + 1, :], axis=0)

    return centroids
