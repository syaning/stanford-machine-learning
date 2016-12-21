import numpy as np


def kMeansInitCentroids(X, K):
    randidx = np.random.permutation(range(X.shape[0]))
    centroids = X[randidx[:K], :]
    return centroids
