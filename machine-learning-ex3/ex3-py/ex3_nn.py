import numpy as np
from scipy.io import loadmat

from displayData import displayData
from sigmoid import sigmoid
from predict import predict


# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
print('Loading and Visualizing Data ...')
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]
displayData(sel)


# ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.
print('Loading Saved Neural Network Parameters ...')
params = loadmat('ex3weights.mat')
Theta1 = params['Theta1']
Theta2 = params['Theta2']


# ================= Part 3: Implement Predict =================
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.
p = predict(Theta1, Theta2, X)
accuracy = np.mean((p == y).astype(int))
print('Training Set Accuracy: %.2f %%' % (accuracy * 100))

# To give you an idea of the network's output, you can also run
# through the examples one at the a time to see what it is predicting.
for i in range(10):
    print('Displaying Example Image')
    pred = predict(Theta1, Theta2, sel[i:i + 1, :])
    print('Neural Network Prediction: %d (digit %d)' % (pred, pred % 10))
    displayData(sel[i:i + 1, :])
