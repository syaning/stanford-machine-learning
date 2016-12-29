def estimateGaussian(X):
    mu = X.mean(axis=0)
    sigma2 = X.var(axis=0)
    return mu, sigma2
