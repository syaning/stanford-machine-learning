import numpy as np
from scipy.io import loadmat

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids


# ================= Part 1: Find Closest Centroids ====================
# To help you implement K-Means, we have divided the learning algorithm
# into two functions -- findClosestCentroids and computeCentroids. In this
# part, you shoudl complete the code in the findClosestCentroids function.
print('Finding closest centroids.')

data = loadmat('ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = findClosestCentroids(X, initial_centroids)
print('Closest centroids for the first 3 examples:')
print(idx[:3].ravel())
print('(the closest centroids should be 1, 3, 2 respectively)\n')


# ===================== Part 2: Compute Means =========================
# After implementing the closest centroids function, you should now
# complete the computeCentroids function.
print('Computing centroids means.')

# Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]')
