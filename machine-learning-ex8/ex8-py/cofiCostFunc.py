import numpy as np


# def cofiCostFunc(X, Theta, Y, R, num_users, num_movies, num_features, lamda):
def cofiCostFunc(X, Theta, Y, R, lamda):
    # X = np.array(params[:num_movies * num_features]
    #              ).reshape(num_features, num_movies).T
    # Theta = np.array(params[num_movies * num_features:]
    #                  ).reshape(num_features, num_users).T

    J = ((X.dot(Theta.T) - Y)**2 * R).sum() / 2 + \
        ((Theta**2).sum() + (X**2).sum()) / 2 * lamda
    X_grad = ((X.dot(Theta.T) - Y) * R).dot(Theta) + lamda * X
    Theta_grad = ((X.dot(Theta.T) - Y) * R).T.dot(X) + lamda * Theta

    grad = np.hstack((X_grad.ravel(), Theta_grad.ravel()))
    return J, grad
