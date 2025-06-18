import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.losses import MeanSquaredError, CrossEntropyLoss

class TestLosses(unittest.TestCase):

    def test_mean_squared_error_forward(self):
        mse = MeanSquaredError()
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1.5, 2.5], [3.0, 3.5]])
        # MSE = mean([(1-1.5)^2, (2-2.5)^2, (3-3)^2, (4-3.5)^2])
        #     = mean([(-0.5)^2, (-0.5)^2, (0)^2, (0.5)^2])
        #     = mean([0.25, 0.25, 0, 0.25]) = 0.75 / 4 = 0.1875
        # However, np.mean averages over all elements.
        # If interpreted as per-sample loss, then average of (0.25+0.25)/2 and (0+0.25)/2
        # My implementation: np.mean(np.power(y_true - y_pred, 2))
        # ( (1-1.5)^2 + (2-2.5)^2 + (3-3)^2 + (4-3.5)^2 ) / 4
        # = (0.25 + 0.25 + 0 + 0.25) / 4 = 0.75 / 4 = 0.1875
        expected_loss = np.mean(np.power(y_true - y_pred, 2))
        self.assertAlmostEqual(mse.forward(y_pred, y_true), expected_loss)

    def test_mean_squared_error_backward(self):
        mse = MeanSquaredError()
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]]) # batch_size=2, num_outputs=2
        y_true = np.array([[1.5, 2.5], [3.0, 3.5]])
        # dL/dy_pred = 2 * (y_pred - y_true) / m (where m is batch_size)
        # m = 2
        expected_grad = 2 * (y_pred - y_true) / y_true.shape[0]
        np.testing.assert_array_almost_equal(mse.backward(y_pred, y_true), expected_grad)

    def test_cross_entropy_loss_forward(self):
        ce = CrossEntropyLoss()
        # y_pred should be probabilities (e.g., output of softmax)
        y_pred = np.array([[0.1, 0.9], [0.8, 0.2]]) # Batch of 2
        y_true = np.array([[0, 1], [1, 0]])   # One-hot encoded

        # Sample 1: - (0*log(0.1) + 1*log(0.9)) = -log(0.9)
        # Sample 2: - (1*log(0.8) + 0*log(0.2)) = -log(0.8)
        # Loss = (-log(0.9) - log(0.8)) / 2 (average over batch)
        epsilon = 1e-12 # from implementation
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
        log_likelihood_s1 = -np.sum(y_true[0] * np.log(y_pred_clipped[0]))
        log_likelihood_s2 = -np.sum(y_true[1] * np.log(y_pred_clipped[1]))
        expected_loss = (log_likelihood_s1 + log_likelihood_s2) / 2.0

        self.assertAlmostEqual(ce.forward(y_pred, y_true), expected_loss)

    def test_cross_entropy_loss_backward(self):
        ce = CrossEntropyLoss()
        y_pred = np.array([[0.1, 0.9], [0.8, 0.2]])
        y_true = np.array([[0, 1], [1, 0]])
        epsilon = 1e-12 # from implementation

        # dL/dy_pred = -y_true / y_pred_clipped
        # Averaged over batch size m
        m = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)
        expected_grad = (-y_true / y_pred_clipped) / m

        np.testing.assert_array_almost_equal(ce.backward(y_pred, y_true), expected_grad)

    def test_cross_entropy_loss_numerical_stability(self):
        ce = CrossEntropyLoss()
        # Test with y_pred values at the boundaries (0 or 1)
        y_pred_zeros = np.array([[0.0, 1.0]])
        y_true = np.array([[0, 1]])
        # Should not result in NaN or Inf due to clipping
        loss_zeros = ce.forward(y_pred_zeros, y_true)
        self.assertTrue(np.isfinite(loss_zeros))
        grad_zeros = ce.backward(y_pred_zeros, y_true)
        self.assertTrue(np.all(np.isfinite(grad_zeros)))

        y_pred_ones = np.array([[1.0, 0.0]])
        y_true_ones = np.array([[1,0]])
        loss_ones = ce.forward(y_pred_ones, y_true_ones)
        self.assertTrue(np.isfinite(loss_ones))
        grad_ones = ce.backward(y_pred_ones, y_true_ones)
        self.assertTrue(np.all(np.isfinite(grad_ones)))


if __name__ == '__main__':
    unittest.main()
