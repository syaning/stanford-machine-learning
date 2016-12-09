import numpy as np


def polyFeatures(X, p):
    X_poly = []
    for i in range(p):
        X_poly.append(np.power(X, i + 1))
    return np.hstack(X_poly)
