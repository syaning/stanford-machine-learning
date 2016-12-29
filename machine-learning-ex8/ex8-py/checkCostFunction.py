import numpy as np

from computeNumericalGradient import computeNumericalGradient
from cofiCostFunc import cofiCostFunc


def checkCostFunction(lamda=0):
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = X_t.dot(Theta_t.T)
    Y[np.where(np.random.random_sample(Y.shape) > 0.5)] = 0
    R = np.zeros(Y.shape)
    R[np.where(Y != 0)] = 1

    # Run Gradient Checking
    X = np.random.random_sample(X_t.shape)
    Theta = np.random.random_sample(Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    # params = np.hstack((X.T.flatten(), Theta.T.flatten()))
    costFunc = lambda X, Theta: cofiCostFunc(X, Theta, Y, R, lamda)
    costFunc_w = lambda X, Theta: costFunc(X, Theta)[0]
    numgrad = computeNumericalGradient(costFunc_w, X, Theta)

    cost, grad = cofiCostFunc(X, Theta, Y, R, lamda)

    print(grad)
    print(numgrad)
