import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def cost_function(theta, x, y):
    theta = theta.reshape((theta.size, 1))
    h = sigmoid(x.dot(theta))
    cost = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return cost


def gradient(theta, x, y):
    theta = theta.reshape((theta.size, 1))
    h = sigmoid(x.dot(theta))
    grad = np.dot(x.T, h - y) / y.size
    return grad.flatten()


def predict(theta, x):
    theta = theta.reshape((theta.size, 1))
    p = sigmoid(x.dot(theta))
    return np.round(p)


# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples, ',
      'and o indicating (y = 0) examples.\n')
data = np.loadtxt('ex2data1.txt', delimiter=',')
positive = data[data[:, -1] == 1, :]
negative = data[data[:, -1] == 0, :]

plt.scatter(positive[:, 0], positive[:, 1],
            c='k', marker='+', label='Admitted')
plt.scatter(negative[:, 0], negative[:, 1],
            c='y', marker='o', label='Not admmitted', alpha=0.5)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

# ============ Part 2: Compute Cost and Gradient ============
x = data[:, :-1]
y = data[:, -1:]
m, n = x.shape
x = np.hstack((np.ones((m, 1)), x))
initial_theta = np.zeros(n + 1)

print('Cost at initial theta (zeros):', cost_function(initial_theta, x, y))
print('Gradient at initial theta (zeros):',
      gradient(initial_theta, x, y), '\n')

# =========== Part 3: Optimizing using fmin_bfgs  ===========
theta = fmin_bfgs(cost_function, initial_theta,
                  fprime=gradient, args=(x, y), disp=False)
print('theta: ', theta, '\n')

plot_x = np.array([min(x[:, 1]),  max(x[:, 1])])
plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(positive[:, 0], positive[:, 1],
            c='k', marker='+', label='Admitted')
plt.scatter(negative[:, 0], negative[:, 1],
            c='y', marker='o', label='Not admmitted', alpha=0.5)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

# ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission ',
      'probability of %f' % prob)

p = predict(theta, x)
p = np.mean(p == y) * 100
print('Train Accuracy: %f' % p)
