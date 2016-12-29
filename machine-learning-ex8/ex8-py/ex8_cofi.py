import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction


# =============== Part 1: Loading movie ratings dataset ================
# You will start by loading the movie ratings dataset to understand the
# structure of the data.
print('Loading movie ratings dataset.')

# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
# 943 users
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
# rating to movie i
data = loadmat('ex8_movies.mat')
Y, R = data['Y'], data['R']

# From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): %f / 5\n' %
      Y[0, R[0] == 1].mean())

# We can "visualize" the ratings matrix by plotting it with imagesc
plt.figure()
plt.imshow(Y, aspect='equal', origin='upper',
           extent=(0, Y.shape[1], 0, Y.shape[0] / 2))
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()


# ============ Part 2: Collaborative Filtering Cost Function ===========
# You will now implement the cost function for collaborative filtering.
# To help you debug your cost function, we have included set of weights
# that we trained on that. Specifically, you should complete the code in
# cofiCostFunc.m to return J.

# Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
data = loadmat('ex8_movieParams.mat')
X = data['X']
Theta = data['Theta']
num_users = data['num_users'][0][0]
num_movies = data['num_movies'][0][0]
num_features = data['num_features'][0][0]

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3

X = X[:num_movies, :num_features]
Theta = Theta[:num_users, :num_features]
Y = Y[:num_movies, :num_users]
R = R[:num_movies, :num_users]

# Evaluate cost function
J, _ = cofiCostFunc(X, Theta, Y, R, 0)
print('Cost at loaded parameters: %f' % J)
print('(this value should be about 22.22)\n')


# ============== Part 3: Collaborative Filtering Gradient ==============
# Once your cost function matches up with ours, you should now implement
# the collaborative filtering gradient function. Specifically, you should
# complete the code in cofiCostFunc.py to return the grad argument.
print('Checking Gradients (without regularization) ...')

checkCostFunction()