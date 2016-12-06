import numpy as np
from scipy.optimize import fmin_cg


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, lamda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(X.dot(theta))
    J = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    J += lamda / 2 / m * np.sum(np.square(theta[1:]))
    return J


def gradient(theta, X, y, lamda):
    theta = theta.reshape((theta.size, 1))
    m = y.size
    h = sigmoid(X.dot(theta))
    grad = X.T.dot(h - y) / m
    grad[1:] = grad[1:] + lamda * theta[1:] / m
    return grad.flatten()


def one_vs_all(X, y, num_labels, lamda):
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n))

    for c in range(1, num_labels + 1):
        initial_theta = np.zeros(n)
        all_theta[c % 10, :] = fmin_cg(cost_function, initial_theta, fprime=gradient,
                                       args=(X, (y == c).astype(int), lamda),
                                       maxiter=100, disp=0)
        print('Finish one_vs_all checking number: %d' % c)

    return all_theta


def predict_one_vs_all(all_theta, X):
    pred = sigmoid(X.dot(all_theta.T))
    prob = np.amax(pred, axis=1)
    p = np.argmax(pred, axis=1)
    p[p == 0] = 10
    return prob.reshape((prob.size, 1)), p.reshape((p.size, 1))
