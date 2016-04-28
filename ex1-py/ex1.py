import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def compute_cost(X, y, theta):
    m = y.shape[0]
    J = np.power(np.dot(X, theta) - y, 2).sum() / 2 / m
    return J


def gradient_descent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    for i in range(iterations):
        theta = theta - (alpha / m) * np.dot(X.T, np.dot(X, theta) - y)
    return theta


# Part 1: Basic Function
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')
print(np.eye(5))

# Part 2: Plotting
print('Plotting Data ...')
data = np.loadtxt('ex1data1.txt', delimiter=',')
plt.scatter(data[:, 0], data[:, 1], s=30, marker='+')
plt.xlim(5)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# Part 3: Gradient descent
print('Running Gradient Descent ...')
X = np.hstack((np.ones((data.shape[0], 1)), data[:, :1]))
y = data[:, 1:]
theta = np.zeros((2, 1))
iterations, alpha = 1500, 0.01
theta = gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: %f %f' % (theta[0, 0], theta[1, 0]))

plt.plot(data[:, 0], data[:, 1], 'b+')
plt.plot(data[:, 0], np.dot(X, theta), 'r')
plt.xlim(5)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

predict1 = np.dot([1, 3.5], theta) * 10000
predict2 = np.dot([1, 7], theta) * 10000
print('For population = 35,000, we predict a profit of %f' % predict1)
print('For population = 70,000, we predict a profit of %f' % predict2)

# Part 4: Visualizing J(theta_0, theta_1)
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i, j] = compute_cost(X, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals,
                rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.xlabel('theta 0')
plt.ylabel('theta 1')
plt.show()

plt.contour(theta0_vals, theta1_vals, J_vals, 40)
plt.plot(theta[0], theta[1], 'b+')
plt.xlabel('theta 0')
plt.ylabel('theta 1')
plt.show()
