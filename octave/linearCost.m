function J = linearCost(X, y, theta)

% Requires
% X - training examples matrix
% y - class labels
% theta - predicted features

m = size(X, 1);
predictions = X*theta;
squareErrors = (predictions - y) .^ 2;
J = 1 / (2*m) * sum(squareErrors);

end
