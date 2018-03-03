# test_gradient.py
import unittest
import gradient

class TestGradientFunctions(unittest.TestCase):
  def setUp(self):
    pass

  def test_mse_cost_2d(self):
    theta = [0, 1]
    training_set = [
        [0, 0],
        [1, 2],
      ]
    self.assertEqual(gradient.mse_cost_2d(theta, training_set), 0.25)

  def test_gradient_descent_mse_2d(self):
    training_set = [
        [0, 0],
        [1, 2],
        [2, 4],
        [3, 6],
      ] 
    expected_theta = [0, 2]
    cd = 0.01
    estimated_theta = gradient.gradient_descent_mse_2d(training_set, learning_rate=0.1, convergence_diff=[cd, cd], plotting=True)
    self.assertTrue(abs(expected_theta[0] - estimated_theta[0]) < cd)
    self.assertTrue(abs(expected_theta[1] - estimated_theta[1]) < cd)

if __name__ == '__main__':
  unittest.main();
