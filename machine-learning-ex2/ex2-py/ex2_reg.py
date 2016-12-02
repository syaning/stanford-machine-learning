import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def map_feature(x1, x2):
    cols = [np.ones((x1.size, 1))]
    for i in range(1, 7):
        for j in range(i + 1):
            cols.append(np.power(x1, i - j) * np.power(x2, j))
    return np.hstack(cols)


def cost_function_eg(theta, x, y, lmbda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(x.dot(theta))
    cost = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    cost -= lmbda / 2 / m * np.dot(theta[1:, :].T, theta[1:, :]).sum()
    return cost


def gradient_reg(theta, x, y, lmbda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(x.dot(theta))
    grad = x.T.dot(h - y) / m + lmbda / m * theta
    grad[0] = grad[0] - lmbda / m * theta[0]
    return grad.flatten()


def predict(theta, x):
    theta = theta.reshape((theta.size, 1))
    p = sigmoid(x.dot(theta))
    return np.round(p)

# Plot Data
data = np.loadtxt('ex2data2.txt', delimiter=',')
positive = data[data[:, -1] == 1, :]
negative = data[data[:, -1] == 0, :]
plt.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', label='y = 1')
plt.scatter(negative[:, 0], negative[:, 1], c='y', marker='o', label='y = 0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

# =========== Part 1: Regularized Logistic Regression ============
x = map_feature(data[:, [0]], data[:, [1]])
y = data[:, [2]]
initial_theta = np.zeros(x.shape[1])
lmbda = 1
cost = cost_function_eg(initial_theta, x, y, lmbda)
print('Cost at initial theta (zeros): %f' % cost)

# ============= Part 2: Regularization and Accuracies =============
theta = fmin_bfgs(cost_function_eg, initial_theta,
                  fprime=gradient_reg, args=(x, y, lmbda))
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((u.size, v.size))
for i in range(u.size):
    for j in range(v.size):
        z[i, j] = map_feature(np.array(u[i]).reshape((1, 1)),
                              np.array(v[j]).reshape((1, 1))).dot(theta)

plt.contour(u, v, z.T, levels=[0.0])
plt.scatter(positive[:, 0], positive[:, 1], c='b', marker='+', label='y = 1')
plt.scatter(negative[:, 0], negative[:, 1], c='y', marker='o', label='y = 0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

# Compute accuracy on our training set
p = predict(theta, x)
p = np.mean(p == y) * 100
print('Train Accuracy: %f' % p)