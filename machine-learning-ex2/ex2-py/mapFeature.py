import numpy as np


def mapFeature(X1, X2):
    degree = 6
    out = [np.ones((X1.size, 1))]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append(np.power(X1, i - j) * np.power(X2, j))

    return np.hstack(out)
