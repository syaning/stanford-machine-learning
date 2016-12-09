import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from linearRegCostFunction import linearRegCostFunction
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve


# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and plot
# the data.
print('Loading and Visualizing Data ...\n')
data = loadmat('ex5data1.mat')
X, Xval, Xtest = data['X'], data['Xval'], data['Xtest']
y, yval, ytest = data['y'], data['yval'], data['ytest']
m = X.shape[0]
X_ones = np.hstack((np.ones((m, 1)), X))
Xval_ones = np.hstack((np.ones((Xval.shape[0], 1)), Xval))
Xtest_ones = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

plt.scatter(X, y, marker='x', c='r', s=60)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


# =========== Part 2: Regularized Linear Regression Cost =============
# You should now implement the cost function for regularized linear
# regression.
theta = np.array([1, 1])
J, _ = linearRegCostFunction(X_ones, y, theta, 1)
print('Cost at theta = [1 ; 1]:', J)
print('(this value should be about 303.993192)\n')


# =========== Part 3: Regularized Linear Regression Gradient =============
# You should now implement the gradient for regularized linear
# regression.
theta = np.array([1, 1])
_, grad = linearRegCostFunction(X_ones, y, theta, 1)
print('Gradient at theta = [1 ; 1]:', grad)
print('(this value should be about [-15.303016; 598.250744])\n')


# =========== Part 4: Train Linear Regression =============
# Once you have implemented the cost and gradient correctly, the
# trainLinearReg function will use your cost function to train
# regularized linear regression.
#
# Write Up Note: The data is non-linear, so this will not give a great
#                fit.
lamda = 0
theta = trainLinearReg(X_ones, y, lamda)

plt.scatter(X, y, marker='x', c='r', s=60)
plt.plot(X, X_ones.dot(theta))
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


# =========== Part 5: Learning Curve for Linear Regression =============
# Next, you should implement the learningCurve function.
#
# Write Up Note: Since the model is underfitting the data, we expect to
#                see a graph with "high bias" -- slide 8 in ML-advice.pdf
lamda = 0
error_train, error_val = learningCurve(X_ones, y, Xval_ones, yval, lamda)

plt.plot(range(1, m + 1), error_train, label='Train')
plt.plot(range(1, m + 1), error_val, label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.show()

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))


# =========== Part 6: Feature Mapping for Polynomial Regression =============
# One solution to this is to use polynomial regression. You should now
# complete polyFeatures to map each example into its powers
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.hstack((np.ones((m, 1)), X_poly))

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.hstack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.hstack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

print('\nNormalized Training Example 1:')
print(X_poly[1, 1:].reshape((p, 1)), '\n')


# =========== Part 7: Learning Curve for Polynomial Regression =============
# Now, you will get to experiment with polynomial regression with multiple
# values of lambda. The code below runs polynomial regression with
# lambda = 0. You should try running the code with different values of
# lambda to see how the fit and learning curve change.
lamda = 0
theta = trainLinearReg(X_poly, y, lamda)

plt.scatter(X, y, marker='x', c='r', s=60)
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %f)' % lamda)
plt.show()

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lamda)
plt.plot(range(1, m + 1), error_train, label='Train')
plt.plot(range(1, m + 1), error_val, label='Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend()
plt.show()

print('Polynomial Regression (lambda = %f)' % lamda)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))


# =========== Part 8: Validation for Selecting Lambda =============
# You will now implement validationCurve to test various values of
# lambda on a validation set. You will then use this to select the
# "best" lambda value.
lambda_vec, error_train, error_val = validationCurve(
    X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train, label='Train')
plt.plot(lambda_vec, error_val, label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('\nlambda\t\tTrain Error\tValidation Error')
for i in range(lambda_vec.size):
    print('%f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))
