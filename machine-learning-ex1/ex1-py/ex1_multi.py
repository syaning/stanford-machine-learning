import numpy as np
import matplotlib.pyplot as plt

from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn


# ================ Part 1: Feature Normalization ================
print('Loading data ...')
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, [2]]
m = y.size

print('First 10 examples from the dataset:')
for i in range(10):
    print('x = [%d %d], y = %d' % (X[i, 0], X[i, 1], y[i]))

print('Normalizing Features ...\n')
X, mu, sigma = featureNormalize(X)
X = np.hstack((np.ones((m, 1)), X))


# ================ Part 2: Gradient Descent ================
print('Running gradient descent ...')
alpha = 0.01
iters = 400
theta = np.zeros((X.shape[1], 1))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, iters)

# Plot the convergence graph
plt.plot(range(1, len(J_history) + 1), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]], theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $%f\n' % price)


# ================ Part 3: Normal Equations ================
print('Solving with normal equations...')
X = np.hstack((np.ones((m, 1)), data[:, [0, 1]]))
y = data[:, [2]]
theta = normalEqn(X, y)

print('Theta computed from the normal equations:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1, 1650, 3], theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): $%f' % price)
