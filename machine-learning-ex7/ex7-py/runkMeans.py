import numpy as np

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from plotProgresskMeans import plotProgresskMeans


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # Run K-Means
    for i in range(1, max_iters + 1):
        print('K-Means iteration %d/%d' % (i, max_iters))

        idx = findClosestCentroids(X, centroids)

        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        centroids = computeCentroids(X, idx, K)

    return centroids, idx
