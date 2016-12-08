import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from sigmoid import sigmoid
from plotData import plotData
from costFunction import costFunction
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict


data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, [0, 1]]
y = data[:, [2]]


# ==================== Part 1: Plotting ====================
# We start the exercise by first plotting the data to understand the
# the problem we are working with.
print('Plotting data with + indicating (y = 1) examples,',
      'and o indicating (y = 0) examples.\n')
plotData(X, y, xlabel='Exam 1 score', ylabel='Exam 2 score',
         legends=['Admitted', 'Not Admitted'])


# ============ Part 2: Compute Cost and Gradient ============
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. You neeed to complete the code in
# costFunction.py
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
initial_theta = np.zeros(n + 1)

cost, grad = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Gradient at initial theta (zeros):', grad, '\n')


# =========== Part 3: Optimizing using fmin_bfgs  ===========
# In this exercise, you will use a built-in function (fminunc) to find the
# optimal parameters theta.
cost_function = lambda p: costFunction(p, X, y)[0]
grad_function = lambda p: costFunction(p, X, y)[1]

theta = fmin_bfgs(cost_function, initial_theta, fprime=grad_function)
print('theta:', theta, '\n')

plotDecisionBoundary(theta, X[:, 1:], y, xlabel='Exam 1 score', ylabel='Exam 2 score',
                     legends=['Admitted', 'Not Admitted', 'Decision Boundary'])


# ============== Part 4: Predict and Accuracies ==============
# After learning the parameters, you'll like to use it to predict the outcomes
# on unseen data. In this part, you will use the logistic regression model
# to predict the probability that a student with score 45 on exam 1 and
# score 85 on exam 2 will be admitted.
#
# Furthermore, you will compute the training and test set accuracies of
# our model.
#
# Your task is to complete the code in predict.py
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission',
      'probability of %f' % prob)

p = predict(theta, X)
p = np.mean(p == y) * 100
print('Train Accuracy: %.2f %%' % p)
