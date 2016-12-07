import numpy as np

from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnCostFunction
from computeNumericalGradient import computeNumericalGradient


def checkNNGradients(lamda=0):
    input_layer_size = 3
    hidden_layer_size = 3
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.mod(range(1, m + 1), num_labels).reshape((m, 1))

    # Unroll parameters
    nn_params = np.hstack((Theta1.T.ravel(), Theta2.T.ravel()))

    # Short hand for cost function
    costFunc = lambda p: nnCostFunction(
        p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    print(np.vstack((numgrad, grad)).T)
    print('The above two columns you get should be very similar.\n' +
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.py, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n' +
          'the relative difference will be small (less than 1e-9). \n' +
          'Relative Difference: %g\n' % diff)
