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

if __name__ == '__main__':
  unittest.main();
