import numpy as np


def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)
    X = X - mu
    p = (2 * np.pi)**(-k / 2) * np.linalg.det(Sigma2) ** (-0.5) * \
        np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(Sigma2)) * X, axis=1))
    return p.reshape(-1, 1)
