import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_cg

from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predict import predict

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
print('Loading and Visualizing Data ...\n')
data = loadmat('ex4data1.mat')
X, y = data['X'], data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]
displayData(sel)


# ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.
print('Loading Saved Neural Network Parameters ...\n')
parameters = loadmat('ex4weights.mat')
Theta1, Theta2 = parameters['Theta1'], parameters['Theta2']

# Unroll parameters
nn_params = np.hstack((Theta1.T.ravel(), Theta2.T.ravel()))

# ================ Part 3: Compute Cost (Feedforward) ================
# To the neural network, you should first start by implementing the
# feedforward part of the neural network that returns the cost only. You
# should complete the code in nnCostFunction.py to return cost. After
# implementing the feedforward to compute the cost, you can verify that
# your implementation is correct by verifying that you get the same cost
# as us for the fixed debugging parameters.
#
# We suggest implementing the feedforward cost *without* regularization
# first so that it will be easier for you to debug. Later, in part 4, you
# will get to implement the regularized cost.
print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lamda = 0

J, _ = nnCostFunction(nn_params, input_layer_size,
                      hidden_layer_size, num_labels, X, y, lamda)
print('Cost at parameters (loaded from ex4weights): %f' % J)
print('(this value should be about 0.287629)\n')


# =============== Part 4: Implement Regularization ===============
# Once your cost function implementation is correct, you should now
# continue to implement the regularization with the cost.
print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
lamda = 1

J, _ = nnCostFunction(nn_params, input_layer_size,
                      hidden_layer_size, num_labels, X, y, lamda)
print('Cost at parameters (loaded from ex4weights): %f' % J)
print('(this value should be about 0.383770)\n')


# ================ Part 5: Sigmoid Gradient  ================
# Before you start implementing the neural network, you will first
# implement the gradient for the sigmoid function. You should complete the
# code in the sigmoidGradient.py file.
print('Evaluating sigmoid gradient...')
g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n', g, '\n')


# ================ Part 6: Initializing Pameters ================
# In this part of the exercise, you will be starting to implment a two
# layer neural network that classifies digits. You will start by
# implementing a function to initialize the weights of the neural network
# (randInitializeWeights.py)
print('Initializing Neural Network Parameters ...\n')
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack(
    (initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))


# =============== Part 7: Implement Backpropagation ===============
# Once your cost matches up with ours, you should proceed to implement the
# backpropagation algorithm for the neural network. You should add to the
# code you've written in nnCostFunction.py to return the partial
# derivatives of the parameters.
print('Checking Backpropagation...')
checkNNGradients()


# =============== Part 8: Implement Regularization ===============
# Once your backpropagation implementation is correct, you should now
# continue to implement the regularization with the cost and gradient.
print('Checking Backpropagation (w/ Regularization) ...')

lamda = 3
checkNNGradients(lamda)
debug_J, _ = nnCostFunction(nn_params, input_layer_size,
                            hidden_layer_size, num_labels, X, y, lamda)

print('Cost at (fixed) debugging parameters (w/ lambda = 10): %f' % debug_J)
print('(this value should be about 0.576051)\n')


# =================== Part 9: Training NN ===================
# You have now implemented all the code necessary to train a neural
# network. To train your neural network, we will now use "fmincg", which
# is a function which works similarly to "fminunc". Recall that these
# advanced optimizers are able to train our cost functions efficiently as
# long as we provide them with the gradient computations.
print('Training Neural Network...')

lamda = 1
costFunction = lambda p: nnCostFunction(
    p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[0]
gradFunction = lambda p: nnCostFunction(
    p, input_layer_size, hidden_layer_size, num_labels, X, y, lamda)[1]

nn_params = fmin_cg(costFunction, initial_nn_params,
                    fprime=gradFunction, maxiter=50)

Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(
    (hidden_layer_size, input_layer_size + 1), order='F')
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(
    (num_labels, hidden_layer_size + 1), order='F')


# ================= Part 10: Visualize Weights =================
# You can now "visualize" what the neural network is learning by
# displaying the hidden units to see what features they are capturing in
# the data.
print('Visualizing Neural Network...')
displayData(Theta1[:, 1:])


# ================= Part 11: Implement Predict =================
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.
pred = predict(Theta1, Theta2, X)
accuracy = np.mean((pred == y).astype(int))
print('Training Set Accuracy: %.2f %%' % (accuracy * 100))
