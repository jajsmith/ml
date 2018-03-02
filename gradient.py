# gradient.py
# Functions related to getting gradients

def squared_error(a, b):
  return (b - a)*(b - a)

def hypothesis_2d(a, b, x):
  return a + b*x

def mse_cost_2d(theta, training_set):
  """Returns the cost associated with a 2D hypothesis and training set
  Uses mean squared error
  
  Args:
    theta ([float, float]): vector of two features defining the hypothesis function
    training_set (array[float, float]): set of points
    
  Returns:
    float: cost
  """
  
  m = len(training_set)

  squared_errors = map(
    lambda point: squared_error( hypothesis_2d(theta[0], theta[1], point[0]), point[1] ),
    training_set
  )

  sum_squared_errors = 0
  for se in squared_errors:
    sum_squared_errors += se

  return 1 / (2*m) * sum_squared_errors 

def array_abs_diff(a, b):
  arr = [abs(a_i - b_i) for a_i, b_i in zip(a, b)]
  print('Diff: ', arr)
  return arr

def array_all_lt(a, b):
  for a_i, b_i in zip(a, b):
    if a_i >= b_i:
      return False
  return True



def gradient_descent_mse_2d(training_set, cost_function, learning_rate=0.75, convergence_diff=[0.01,0.01]):
  """Returns the feature vector theta that minimizes the cost function
  Partial derivative based on using mean squared error cost function
  
  Args:
    training_set (array[float, float]): set of points
    cost_function ( function([float, float], array[float, float]) ): uses feature hypothesis
      and training set to determine cost of hypothesis
    learning_rate (float): how fast we change our hypothesis
    convergance_diff (float): difference to check before assuming convergence
    
  Returns:
    [float, float]: ideal features to describe training set
  """

  # Randomly set initial hypothesis
  hypothesis_theta = [0.0, 0.0]

  # so that we don't converge immediately
  last_theta = [
    hypothesis_theta[0]+1.0,
    hypothesis_theta[1]+1.0,
  ]

  m = len(training_set)
  training_xs = [point[0] for point in training_set]
  training_ys = [point[1] for point in training_set]

  while not array_all_lt( array_abs_diff(hypothesis_theta, last_theta), convergence_diff ):
    print("Hypothesis:", hypothesis_theta)
    last_theta = hypothesis_theta

    # this is hard-coded for now based on using the mse_cost_2d function
    # partial deriv of mse_cost_2d w.r.t feature 0
    #  (t + t2*x - y)*(t + t2*x - y) = t^2 + t*t2*x - t*y + t*t2*x -t*y ... partial deriv:
    #  ( sum(2*t + 2*t_2*x - 2*y) ) / 2*m
    # partial deriv of mse_cost_2d w.r.t feature 1
    # ... = t*t_2*x + (t_2*t*x + (t_2*x)^2 - t_2*x*y) - t_2*x*y
    # = 2*t*x + 2*t_2*x^2 + 2*x*y / 2*m
    partial_deriv = [
      (hypothesis_theta[0]*m +
        hypothesis_theta[0]*hypothesis_theta[1]*sum(training_xs) +
        sum(training_ys)) / m,
      (hypothesis_theta[0]*sum(training_xs) +
        hypothesis_theta[0]*sum(map(lambda x: x*x, training_xs)) +
        sum(map(lambda x,y: x*y, training_xs, training_ys))) / m,
    ]

    new_hypothesis = [
      hypothesis_theta[0] - learning_rate * partial_deriv[0],
      hypothesis_theta[1] - learning_rate * partial_deriv[1],
    ]

    hypothesis_theta = new_hypothesis



