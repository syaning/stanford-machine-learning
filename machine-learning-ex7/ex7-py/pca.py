from numpy.linalg import svd


def pca(X):
    sigma = X.T.dot(X) / X.shape[0]
    return svd(sigma)
