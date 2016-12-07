import numpy as np

from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lamda):
    # Reshape nn_params back into the parameters Theta1 and Theta2,
    # the weight matrices for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
        (num_labels, hidden_layer_size + 1), order='F')

    m = X.shape[0]
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    X = np.hstack((np.ones((m, 1)), X))
    yv = np.zeros((m, num_labels))
    for i in range(m):
        yv[i, y[i][0] - 1] = 1

    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.py
    a1 = X
    a2 = np.hstack((np.ones((m, 1)), sigmoid(a1.dot(Theta1.T))))
    a3 = sigmoid(a2.dot(Theta2.T))

    for i in range(m):
        J += (-yv[i, :] * np.log(a3[i, :]) -
              (1 - yv[i, :]) * np.log(1 - a3[i, :])).sum()
    J /= m
    J += ((Theta1[:, 1:] ** 2).sum() +
          (Theta2[:, 1:] ** 2).sum()) * lamda / 2 / m

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    for i in range(m):
        a1 = X[i:i + 1, :].T
        z2 = Theta1.dot(a1)
        a2 = np.vstack(([1], sigmoid(z2)))
        z3 = Theta2.dot(a2)
        a3 = sigmoid(z3)

        delta3 = a3 - yv[i:i + 1, :].T
        delta2 = Theta2.T.dot(delta3) * np.vstack(([1], sigmoidGradient(z2)))

        Theta1_grad += delta2[1:, :].dot(a1.T)
        Theta2_grad += delta3.dot(a2.T)

    Theta1_grad /= m
    Theta2_grad /= m

    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    Theta1_grad[:, 1:] += lamda / m * Theta1[:, 1:]
    Theta2_grad[:, 1:] += lamda / m * Theta2[:, 1:]

    # Unroll gradients
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))

    return J, grad
