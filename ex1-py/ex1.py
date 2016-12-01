import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def compute_cost(x, y, theta):
    m = y.size
    J = np.power(x.dot(theta) - y, 2).sum() / 2 / m
    return J


def gradient_descent(x, y, theta, alpha, iterations):
    m = y.size
    for i in range(iterations):
        theta = theta - (alpha / m) * x.T.dot(x.dot(theta) - y)
    return theta


# ==================== Part 1: Basic Function ====================
print('Running warmup exercise ...')
print('5x5 Identity Matrix:')
print(np.eye(5))

# ======================= Part 2: Plotting =======================
print('Plotting Data ...')
data = np.loadtxt('ex1data1.txt', delimiter=',')
plt.scatter(data[:, 0], data[:, 1], marker='+')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
m, n = data.shape
x = np.hstack((np.ones((m, 1)), data[:, :1]))
y = data[:, 1:]
theta = np.zeros((n, 1))
alpha = 0.01
iterations = 1500
theta = gradient_descent(x, y, theta, alpha, iterations)
print('Theta found by gradient descent: %f %f' % (theta[0, 0], theta[1, 0]))

plt.plot(data[:, 0], data[:, 1], 'b+')
plt.plot(data[:, 0], x.dot(theta), 'r')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

predict1 = np.dot([1, 3.5], theta) * 10000
predict2 = np.dot([1, 7], theta) * 10000
print('For population = 35,000, we predict a profit of %f' % predict1)
print('For population = 70,000, we predict a profit of %f' % predict2)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i, j] = compute_cost(x, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals,
                rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
plt.xlabel('theta 0')
plt.ylabel('theta 1')
plt.show()

plt.contour(theta0_vals, theta1_vals, J_vals, 20)
plt.plot(theta[0], theta[1], 'b+')
plt.xlabel('theta 0')
plt.ylabel('theta 1')
plt.show()
