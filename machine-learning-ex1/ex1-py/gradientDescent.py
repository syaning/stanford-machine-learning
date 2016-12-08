import numpy as np

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.size
    for i in range(iterations):
        theta = theta - (alpha / m) * X.T.dot(X.dot(theta) - y)
    return theta