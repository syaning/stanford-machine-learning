import numpy as np


def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, fan_in + 1))
    W = np.sin(range(1, W.size + 1)).reshape(W.T.shape).T / 10
    return W
