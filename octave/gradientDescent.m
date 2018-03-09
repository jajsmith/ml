function theta = gradientDescentUpdate(X, Y, theta, learningRate, numIterations)

% NOTE: Uses vectorization optimizations in octave rather than the iterative
% approach we're using in python right now

alpha = learningRate
m = length(Y)

for i = 1:numIterations

  gradient = (X * theta - y);
  delta = 1 / m .* sum( gradient .* X );
  theta = theta - alpha .* delta

end

end
