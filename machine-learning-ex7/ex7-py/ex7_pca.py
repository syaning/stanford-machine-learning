import numpy as np
from scipy.io import loadmat
from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from featureNormalize import featureNormalize
from pca import pca
from drawLine import drawLine
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from plotDataPoints import plotDataPoints


# ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
print('Visualizing example dataset for PCA.\n')

data = loadmat('ex7data1.mat')
X = data['X']

plt.scatter(X[:, 0], X[:, 1], marker='o', s=12,
            edgecolors='blue', facecolors='none')
plt.xlim(0.5, 6.5)
plt.ylim(2, 8)
plt.show()


# =============== Part 2: Principal Component Analysis ===============
# You should now implement PCA, a dimension reduction technique. You
# should complete the code in pca.py
print('Running PCA on example dataset.')

# Before running PCA, it is important to first normalize X
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S, _ = pca(X_norm)

print('Top eigenvector:')
print(' U(:,0) = %f %f' % (U[0, 0], U[1, 0]))
print('(you should expect to see -0.707107 -0.707107)\n')

# Draw the eigenvectors centered at mean of data. These lines show the
# directions of maximum variations in the dataset.
plt.scatter(X[:, 0], X[:, 1], marker='o', s=12,
            edgecolors='blue', facecolors='none')
drawLine(mu, mu + 1.5 * S[0] * U[:, 0])
drawLine(mu, mu + 1.5 * S[1] * U[:, 1])
plt.show()


# =================== Part 3: Dimension Reduction ===================
# You should now implement the projection step to map the data onto the
# first k eigenvectors. The code will then plot the data in this reduced
# dimensional space.  This will show you what the data looks like when
# using only the corresponding eigenvectors to reconstruct it.
#
# You should complete the code in projectData.py
print('Dimension reduction on example dataset.')

# Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example: %f' % Z[0])
print('(this value should be about 1.481274)')

X_rec = recoverData(Z, U, K)
print('Approximation of the first example: %f %f' % (X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='o', s=12,
            edgecolors='blue', facecolors='none')
plt.scatter(X_rec[:, 0], X_rec[:, 1], marker='o', s=12,
            edgecolors='red', facecolors='none')
for i in range(X_norm.shape[0]):
    drawLine(X_norm[i], X_rec[i], c='r', linestyle='--', linewidth=0.5)
plt.xlim(-4, 3)
plt.ylim(-4, 3)
plt.show()


# =============== Part 4: Loading and Visualizing Face Data =============
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment
print('Loading face dataset.\n')

data = loadmat('ex7faces.mat')
X = data['X']

displayData(X[:100, :])
plt.show()


# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
# Run PCA and visualize the eigenvectors which are in this case eigenfaces
# We display the first 36 eigenfaces.
print('Running PCA on face dataset.')
print('(this mght take a minute or two ...)\n')

# Before running PCA, it is important to first normalize X by subtracting
# the mean value from each feature
X_norm, mu, sigma = featureNormalize(X)

# Run PCA
U, S, _ = pca(X_norm)

# Visualize the top 36 eigenvectors found
displayData(U[:, :36].T)
plt.show()


# ============= Part 6: Dimension Reduction for Faces =================
# Project images to the eigen space using the top k eigenvectors
# If you are applying a machine learning algorithm
print('Dimension reduction for face dataset.')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: %d %d\n' % Z.shape)


# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
# Project images to the eigen space using the top K eigen vectors and
# visualize only using those K dimensions
# Compare to the original input, which is also displayed
print('Visualizing the projected (reduced dimension) faces.\n')

K = 100
X_rec = recoverData(Z, U, K)

# Display normalized data
plt.subplot(1, 2, 1)
displayData(X_norm[:100, :])
plt.title('Original faces')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
displayData(X_rec[:100, :])
plt.title('Recovered faces')

plt.show()


# === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
# One useful application of PCA is to use it to visualize high-dimensional
# data. In the last K-Means exercise you ran K-Means on 3-dimensional
# pixel colors of an image. We first visualize this output in 3D, and then
# apply PCA to obtain a visualization in 2D.

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
A = imread('bird_small.png')

# If imread does not work for you, you can try instead
#   loadmat('bird_small.mat')

A = A / 255
img_size = A.shape
X = A.reshape((img_size[0] * img_size[1], img_size[2]))
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

# Sample 1000 random indexes (since working with all the data is
# too expensive. If you have a fast computer, you may increase this.
sel = np.floor(np.random.random(1000) * X.shape[0])

# Visualize the data and centroid memberships in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Xs = np.array([X[int(s)] for s in sel])
cmap = plt.get_cmap('jet')
idxn = sel / np.max(sel)
colors = cmap(idxn)
ax.scatter3D(Xs[:, 0], Xs[:, 1], zs=Xs[:, 2], edgecolors=colors,
             marker='o', facecolors='none', lw=0.4, s=10)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()


# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, mu, sigma = featureNormalize(X)

# PCA and project the data to 2D
U, S, _ = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
zs = np.array([Z[int(s)] for s in sel])
idxs = np.array([idx[int(s)] for s in sel])

plotDataPoints(zs, idxs)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()
