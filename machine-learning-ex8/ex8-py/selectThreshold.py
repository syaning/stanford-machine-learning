import numpy as np


def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval), stepsize):
        predictions = (pval < epsilon).astype(int)
        tp = np.logical_and(predictions == 1, yval == 1).sum()
        fp = np.logical_and(predictions == 1, yval == 0).sum()
        fn = np.logical_and(predictions == 0, yval == 1).sum()

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
