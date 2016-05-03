import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def map_feature(X1, X2):
    cols = [np.ones((X1.shape[0], 1))]
    for i in range(1, 7):
        for j in range(i + 1):
            cols.append(np.power(X1, i - j) * np.power(X2, j))
    return np.hstack(cols)


def cost_function_reg(theta, X, y, lmbda):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum() / m
    J = J - lmbda / 2 / m * np.dot(theta[1:, :].T, theta[1:, :]).sum()
    grad = np.dot(X.T, h - y) / m + lmbda / m * theta
    grad[0] = grad[0] - lmbda / m * theta[0]
    return J, grad

data = np.loadtxt('ex2data2.txt', delimiter=',')
positive = data[data[:, 2] == 1, :]
negative = data[data[:, 2] == 0, :]
plt.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', label='y = 1')
plt.scatter(negative[:, 0], negative[:, 1], c='y', marker='o', label='y = 0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

# Part 1: Regularized Logistic Regression
X = map_feature(data[:, [0]], data[:, [1]])
y = data[:, [2]]
initial_theta = np.zeros((X.shape[1], 1))
lmbda = 1
cost, grad = cost_function_reg(initial_theta, X, y, lmbda)

print('Cost at initial theta (zeros): %f' % cost)

# Part 2: Regularization and Accuracies
# TODO
