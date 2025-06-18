import unittest
import numpy as np
import sys
import os

# Adjust path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.activations import ReLU, Sigmoid, Tanh, Softmax, get_activation, Activation

class TestActivations(unittest.TestCase):

    def test_relu_forward(self):
        relu = ReLU()
        data = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_almost_equal(relu.forward(data), expected)

    def test_relu_backward(self):
        relu = ReLU()
        data = np.array([-2, -1, 0, 1, 2])
        relu.forward(data) # Call forward to store Z
        grad_output = np.array([1, 1, 1, 1, 1])
        expected_grad_input = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_almost_equal(relu.backward(grad_output), expected_grad_input)

    def test_sigmoid_forward(self):
        sigmoid = Sigmoid()
        data = np.array([-1, 0, 1])
        expected = 1 / (1 + np.exp(-data))
        np.testing.assert_array_almost_equal(sigmoid.forward(data), expected)

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()
        data = np.array([-1, 0, 1])
        s = sigmoid.forward(data) # Call forward to store A and Z
        grad_output = np.array([0.5, 0.5, 0.5])
        expected_grad_input = grad_output * s * (1 - s)
        np.testing.assert_array_almost_equal(sigmoid.backward(grad_output), expected_grad_input)

    def test_tanh_forward(self):
        tanh = Tanh()
        data = np.array([-1, 0, 1])
        expected = np.tanh(data)
        np.testing.assert_array_almost_equal(tanh.forward(data), expected)

    def test_tanh_backward(self):
        tanh = Tanh()
        data = np.array([-1, 0, 1])
        a = tanh.forward(data) # Call forward to store A
        grad_output = np.array([0.5, 0.5, 0.5])
        expected_grad_input = grad_output * (1 - a**2)
        np.testing.assert_array_almost_equal(tanh.backward(grad_output), expected_grad_input)

    def test_softmax_forward(self):
        softmax = Softmax()
        data = np.array([[1, 2, 3], [1, 1, 1]]) # Batch of 2 samples, 3 classes
        # Sample 1: exp(1-3), exp(2-3), exp(3-3) -> e^-2, e^-1, e^0 / sum(...)
        z1 = data[0] - np.max(data[0])
        exp_z1 = np.exp(z1)
        expected1 = exp_z1 / np.sum(exp_z1)
        # Sample 2: 1/3, 1/3, 1/3
        z2 = data[1] - np.max(data[1])
        exp_z2 = np.exp(z2)
        expected2 = exp_z2 / np.sum(exp_z2)

        output = softmax.forward(data)
        np.testing.assert_array_almost_equal(output[0], expected1)
        np.testing.assert_array_almost_equal(output[1], expected2)

    def test_softmax_backward(self):
        softmax = Softmax()
        # Example from a known source/calculation if possible, or ensure properties
        # For simplicity, checking if shapes match and it runs.
        # A more rigorous test would involve numerical gradient checking.
        data = np.array([[1, 2, 3]])
        softmax.forward(data) # Store A
        grad_output = np.array([[0.1, 0.2, -0.3]]) # dL/dA

        # dL/dZ_i = A_i * (dL/dA_i - sum_j(dL/dA_j * A_j))
        # This is one way to compute it:
        s_i = softmax.A[0,:]
        da_i = grad_output[0,:]
        jacobian_matrix = np.diagflat(s_i) - np.outer(s_i, s_i) # C x C
        expected_grad = np.dot(da_i.reshape(1,-1), jacobian_matrix)

        grad_input = softmax.backward(grad_output)
        self.assertEqual(grad_input.shape, data.shape)
        np.testing.assert_array_almost_equal(grad_input, expected_grad, decimal=6)


    def test_activation_get_config(self):
        relu = ReLU()
        self.assertEqual(relu.get_config(), {'name': 'relu'})
        sigmoid = Sigmoid()
        self.assertEqual(sigmoid.get_config(), {'name': 'sigmoid'})
        tanh = Tanh()
        self.assertEqual(tanh.get_config(), {'name': 'tanh'})
        softmax = Softmax()
        self.assertEqual(softmax.get_config(), {'name': 'softmax'})

    def test_get_activation_factory(self):
        self.assertIsInstance(get_activation('relu'), ReLU)
        self.assertIsInstance(get_activation('sigmoid'), Sigmoid)
        self.assertIsInstance(get_activation('tanh'), Tanh)
        self.assertIsInstance(get_activation('softmax'), Softmax)
        self.assertIsNone(get_activation(None))
        with self.assertRaises(ValueError):
            get_activation('unknown_activation')

if __name__ == '__main__':
    unittest.main()
