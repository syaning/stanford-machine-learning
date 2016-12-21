def recoverData(Z, U, K):
    return Z.dot(U[:, :K].T)
