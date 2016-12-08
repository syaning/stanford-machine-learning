import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from warmUpExercise import warmUpExercise
from plotData import plotData, plotData2
from gradientDescent import gradientDescent
from computeCost import computeCost


# ==================== Part 1: Basic Function ====================
print('Running warmup exercise ...')
print('5x5 Identity Matrix:')
print(warmUpExercise(), '\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X, y = data[:, 0], data[:, 1]
m = data.shape[0]
plotData(X, y)


# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...')
m, n = data.shape
X = np.hstack((np.ones((m, 1)), data[:, [0]]))
y = data[:, [1]]
theta = np.zeros((2, 1))
iterations = 1500
alpha = 0.01

theta = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: %f %f' % (theta[0, 0], theta[1, 0]))

plotData2(theta, X, y)

predict1 = np.dot([1, 3.5], theta) * 10000
predict2 = np.dot([1, 7], theta) * 10000
print('For population = 35,000, we predict a profit of %f' % predict1)
print('For population = 70,000, we predict a profit of %f' % predict2)


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('\nVisualizing J(theta_0, theta_1) ...')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i, j] = computeCost(X, y, t)

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
plt.plot(theta[0, 0], theta[1, 0], 'b+')
plt.xlabel('theta 0')
plt.ylabel('theta 1')
plt.show()
