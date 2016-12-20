import numpy as np


def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros((m, 1))

    for i in range(m):
        min_distance = float('inf')
        for j in range(K):
            distance = np.sum((X[i] - centroids[j])**2)
            if distance < min_distance:
                idx[i, 0] = j + 1
                min_distance = distance

    return idx
