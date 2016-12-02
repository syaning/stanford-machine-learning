import numpy as np
import matplotlib.pyplot as plt


def normalize_feature(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost_multi(X, y, theta):
    m = y.shape[0]
    J = np.power(np.dot(X, theta) - y, 2).sum() / 2 / m
    return J


def gradient_descent_multi(X, y, theta, alpha, iters):
    m = y.shape[0]
    J_history = list(range(iters))
    for i in range(iters):
        theta = theta - (alpha / m) * np.dot(X.T, np.dot(X, theta) - y)
        J_history[i] = compute_cost_multi(X, y, theta)
    return theta, J_history


def normal_equations(X, y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return theta

# ================ Part 1: Feature Normalization ================
print('Loading data ...')
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:-1]
y = data[:, -1:]
m = y.size

print('First 10 examples from the dataset:')
for i in range(10):
    print('x = [%d %d], y = %d' % (X[i, 0], X[i, 1], y[i]))

print('Normalizing Features ...')
X, mu, sigma = normalize_feature(X)
X = np.hstack((np.ones((X.shape[0], 1)), X))

# ================ Part 2: Gradient Descent ================
print('Running gradient descent ...')
alpha, iters = 0.01, 400
theta = np.zeros((X.shape[1], 1))
theta, J_history = gradient_descent_multi(X, y, theta, alpha, iters)

# Plot the convergence graph
plt.plot(range(1, len(J_history) + 1), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]], theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): %f' % price)

# ================ Part 3: Normal Equations ================
print('Solving with normal equations...')
X = np.hstack((np.ones((data.shape[0], 1)), data[:, 0:2]))
y = data[:, 2:]
theta = normal_equations(X, y)

print('Theta computed from the normal equations:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.dot([1, 1650, 3], theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): %f' % price)
