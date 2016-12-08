import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from mapFeature import mapFeature
from plotData import plotData
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict


# Plot Data
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, [2]]
plotData(X, y, xlabel='Microchip Test 1', ylabel='Microchip Test 2',
         legends=['y = 1', 'y = 0'])


# =========== Part 1: Regularized Logistic Regression ============
# In this part, you are given a dataset with data points that are not
# linearly separable. However, you would still like to use logistic
# regression to classify the data points.
#
# To do so, you introduce more features to use -- in particular, you add
# polynomial features to our data matrix (similar to polynomial
# regression).

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:, [0]], X[:, [1]])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lamda = 1

cost, _ = costFunctionReg(initial_theta, X, y, lamda)
print('Cost at initial theta (zeros):', cost)


# ============= Part 2: Regularization and Accuracies =============
# Optional Exercise:
# In this part, you will get to try different values of lambda and
# see how regularization affects the decision coundart
#
# Try the following values of lambda (0, 1, 10, 100).
#
# How does the decision boundary change when you vary lambda? How does
# the training set accuracy vary?
cost_function = lambda p: costFunctionReg(p, X, y, lamda)[0]
grad_function = lambda p: costFunctionReg(p, X, y, lamda)[1]

theta = fmin_bfgs(cost_function, initial_theta, fprime=grad_function)

plotDecisionBoundary(theta, X[:, 1:], y, xlabel='Microchip Test 1', ylabel='Microchip Test 2',
                     legends=['y = 1', 'y = 0', 'Decision Boundary'])


# Compute accuracy on our training set
p = predict(theta, X)
p = np.mean(p == y) * 100
print('Train Accuracy: %.2f %%' % p)
