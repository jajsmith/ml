# gradient.py
# Functions related to getting gradients

import matplotlib.pyplot as plt
import numpy as np

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

  # Use 2*m to make derivative simpler later
  return 1 / (2*m) * sum_squared_errors 

def array_abs_diff(a, b):
  arr = [abs(a_i - b_i) for a_i, b_i in zip(a, b)]
  return arr

def array_all_lt(a, b):
  for a_i, b_i in zip(a, b):
    if a_i >= b_i:
      return False
  return True

def gradient_descent_mse_2d(training_set, learning_rate=0.05, convergence_diff=[0.01,0.01], plotting=False):
  """Returns the feature vector theta that minimizes the cost function
  Partial derivative from mean squared error cost function
  
  Args:
    training_set (array[float, float]): set of points
      and training set to determine cost of hypothesis
    learning_rate (float): how fast we change our hypothesis
    convergence_diff (float): difference to check before assuming convergence
    
  Returns:
    [float, float]: ideal features to describe training set
  """

  # Randomly set initial hypothesis
  hypothesis_theta = [0.0, 1.0]

  # so that we don't converge immediately
  last_theta = [
    hypothesis_theta[0]+1.0,
    hypothesis_theta[1]+1.0,
  ]

  m = len(training_set)
  training_xs = [point[0] for point in training_set]
  training_ys = [point[1] for point in training_set]

  costs = []

  while not array_all_lt( array_abs_diff(hypothesis_theta, last_theta), convergence_diff ):
    if (plotting):
      # plot the line created by theta features
      # Don't actually show plot here, we'll show it at the end
      x = np.linspace(-15, 15, 100)
      y = hypothesis_theta[0] + hypothesis_theta[1]*x
      plt.plot(x, y)
    cost = mse_cost_2d(hypothesis_theta, training_set)
    costs.append(cost)
    
    print("Hypothesis: ", hypothesis_theta)
    print("  Last Hyp: ", last_theta)
    print("  Difference:", array_abs_diff(hypothesis_theta, last_theta))
    print("  Cost: ", cost)
    last_theta = hypothesis_theta

    # this is hard-coded for now based on using the mse_cost_2d function
    ht = hypothesis_theta

    # Partial Derivative of mse_cost_2d function
    theta0_pd = 0
    for i in range(0, m):
      theta0_pd += hypothesis_2d(ht[0], ht[1], training_xs[i]) - training_ys[i]
    theta0_pd = theta0_pd / m
    
    theta1_pd = 0
    for i in range(0, m):
      theta1_pd += training_xs[i] * (hypothesis_2d(ht[0], ht[1], training_xs[i]) - training_ys[i]) 
    theta1_pd = theta1_pd / m

    partial_deriv = [theta0_pd, theta1_pd]

    # Generate new hypothesis
    new_hypothesis = [
      hypothesis_theta[0] - learning_rate * partial_deriv[0],
      hypothesis_theta[1] - learning_rate * partial_deriv[1],
    ]
    hypothesis_theta = new_hypothesis
  
  if (plotting):
    # This plots what our line has been doing through all iterations
    # Ideally there will be one final line going through the training set
    plt.plot(training_xs, training_ys, 'ro')
    plt.title('Iterations of theta vs. training set')
    plt.show()

    # This plots our cost function. Should converge to zero
    plt.plot(costs, 'ro')
    plt.title('Cost vs. Iterations of Gradient Descent')
    plt.ylabel('J(theta) cost')
    plt.xlabel('Iteration')
    plt.show()
  return hypothesis_theta



