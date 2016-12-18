function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error_val = 1.0;

for C_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for sigma_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 3]
    model = svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try));
    predictions = svmPredict(model, Xval);
    error_val_try = mean(double(predictions ~= yval));
    printf('C=%f, sigma=%f, error=%f', C_try, sigma_try, error_val_try);
    if error_val_try < error_val
      C = C_try;
      sigma = sigma_try;
      error_val = error_val_try;
    end
  end
end





% =========================================================================

end
