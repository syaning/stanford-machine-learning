import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, X, y):
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum() / m
    grad = np.dot(X.T, h - y) / m
    return J, grad


# Part 1: Plotting
data = np.loadtxt('ex2data1.txt', delimiter=',')

print('Plotting data with + indicating (y = 1) examples',
      'and o indicating (y = 0) examples.')
positive = data[data[:, 2] == 1, :]
negative = data[data[:, 2] == 0, :]
plt.scatter(positive[:, 0], positive[:, 1],
            c='b', marker='+', label='Admitted')
plt.scatter(negative[:, 0], negative[:, 1],
            c='y', marker='o', label='Not admmitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

# Part 2: Compute Cost and Gradient
X = data[:, [0, 1]]
y = data[:, [2]]
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
initial_theta = np.zeros((n + 1, 1))
cost, grad = cost_function(initial_theta, X, y)

print('Cost at initial theta (zeros): %f' % cost)
print('Gradient at initial theta (zeros):')
print(grad)

# Part 3: Optimizing using fminunc
# TODO
