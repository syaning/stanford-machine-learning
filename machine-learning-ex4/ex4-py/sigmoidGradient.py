from sigmoid import sigmoid


def sigmoidGradient(z):
    g = sigmoid(z)
    return g * (1 - g)
