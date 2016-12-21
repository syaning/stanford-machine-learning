import matplotlib.pyplot as plt

from plotDataPoints import plotDataPoints
from drawLine import drawLine


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx)

    plt.plot(previous[:, 0], previous[:, 1], 'rx', lw=3)
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx', lw=3)

    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :])

    plt.title('Iteration number %d' % i)
    plt.show()
