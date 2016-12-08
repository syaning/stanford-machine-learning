import numpy as np
from scipy.io import loadmat

from displayData import displayData
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

num_labels = 10


# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
print('Loading and Visualizing Data ...')
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]
displayData(sel)


# ============ Part 2: Vectorize Logistic Regression ============
# In this part of the exercise, you will reuse your logistic regression
# code from the last exercise. You task here is to make sure that your
# regularized logistic regression implementation is vectorized. After
# that, you will implement one-vs-all classification for the handwritten
# digit dataset.
print('Training One-vs-All Logistic Regression...')
# X = np.hstack((np.ones((X.shape[0], 1)), X))
lamda = 0.1
all_theta = oneVsAll(X, y, num_labels, lamda)


# ================ Part 3: Predict for One-Vs-All ================
p = predictOneVsAll(all_theta, X)
accuracy = np.mean((p == y).astype(int))
print('Training Set Accuracy: %.2f %%' % (accuracy * 100))
