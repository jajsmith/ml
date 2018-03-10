function [J, grad] = logisticCostAndGradient(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Cost Function

hypothesis = sigmoid( X * theta );
S = sum(- y .* log(hypothesis) - (1 - y) .* log( 1 - hypothesis));
J = S ./ m;

% Gradient

S = sum( (hypothesis - y) .* X );
grad = S ./ m;

end
