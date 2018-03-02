# gradient.py
# Functions related to getting gradients

from functools import reduce

def squared_error(a, b):
  return (b - a)*(b - a)

def hypothesis_2d(a, b, x):
  return a + b*x

def mse_cost_2d(theta, training_set):
  """Returns the cost associated with a 2D hypothesis and training set
  Uses mean squared error
  
  Args:
    theta ([int, int]): vector of two features defining the hypothesis function
    training_set (array[int, int]): set of points
    
  Returns:
    int: cost
  """
  
  m = len(training_set)

  errors = map(
    (lambda point: hypothesis_2d(theta[0], theta[1], point[0]) - point[1]),
    training_set
  )
  squared_errors = map( lambda x: x*x, errors )

  sum_squared_errors = 0
  for se in squared_errors:
    sum_squared_errors += se

  return 1 / (2*m) * sum_squared_errors 
