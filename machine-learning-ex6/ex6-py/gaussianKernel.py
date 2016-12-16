import numpy as np


def gaussianKernel(x1, x2, sigma):
    x1 = x1.ravel()
    x2 = x2.ravel()
    sim = np.exp(-((x1 - x2)**2).sum() / 2 / (sigma**2))
    return sim
