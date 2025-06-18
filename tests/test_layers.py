import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.layers import DenseLayer, get_layer
from src.activations import ReLU, Sigmoid, get_activation # For testing with activations

class TestDenseLayer(unittest.TestCase):

    def test_dense_layer_initialization(self):
        layer = DenseLayer(input_size=3, output_size=2)
        self.assertEqual(layer.weights.shape, (3, 2))
        self.assertEqual(layer.biases.shape, (1, 2))
        self.assertIsNone(layer.activation)

        relu_activation = ReLU()
        layer_with_activation = DenseLayer(input_size=3, output_size=2, activation=relu_activation)
        self.assertIsNotNone(layer_with_activation.activation)
        self.assertIsInstance(layer_with_activation.activation, ReLU)

    def test_dense_layer_forward_no_activation(self):
        layer = DenseLayer(input_size=2, output_size=2)
        # Manually set weights and biases for predictable output
        layer.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
        layer.biases = np.array([[0.05, -0.05]])

        input_data = np.array([[1, 2], [3, 4]]) # Batch of 2 samples

        # Expected output: input_data @ weights + biases
        # Sample 1: [1,2] @ [[0.1,0.2],[0.3,0.4]] + [0.05,-0.05]
        #         = [1*0.1+2*0.3, 1*0.2+2*0.4] + [0.05,-0.05]
        #         = [0.1+0.6, 0.2+0.8] + [0.05,-0.05]
        #         = [0.7, 1.0] + [0.05,-0.05] = [0.75, 0.95]
        # Sample 2: [3,4] @ [[0.1,0.2],[0.3,0.4]] + [0.05,-0.05]
        #         = [3*0.1+4*0.3, 3*0.2+4*0.4] + [0.05,-0.05]
        #         = [0.3+1.2, 0.6+1.6] + [0.05,-0.05]
        #         = [1.5, 2.2] + [0.05,-0.05] = [1.55, 2.15]
        expected_output = np.array([[0.75, 0.95], [1.55, 2.15]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)
        np.testing.assert_array_almost_equal(layer.input, input_data) # Check input caching

    def test_dense_layer_forward_with_activation(self):
        relu = ReLU()
        layer = DenseLayer(input_size=2, output_size=2, activation=relu)
        layer.weights = np.array([[0.1, -0.2], [-0.3, 0.4]]) # Include negative values for ReLU
        layer.biases = np.array([[0.05, -0.05]])
        input_data = np.array([[1, 2]])

        # Linear part: [1,2] @ [[0.1,-0.2],[-0.3,0.4]] + [0.05,-0.05]
        #            = [1*0.1+2*-0.3, 1*-0.2+2*0.4] + [0.05,-0.05]
        #            = [0.1-0.6, -0.2+0.8] + [0.05,-0.05]
        #            = [-0.5, 0.6] + [0.05,-0.05] = [-0.45, 0.55]
        # After ReLU: [0, 0.55]
        expected_output = np.array([[0, 0.55]])

        output = layer.forward(input_data)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=5)

    def test_dense_layer_backward(self):
        # Test with a simple case, focusing on shapes and weight updates
        # Test with a simple case, focusing on shapes and weight updates
        # Using no activation to simplify gradient checking for update part
        layer = DenseLayer(input_size=2, output_size=1, activation=None)
        input_data = np.array([[1.0, 2.0]]) # Batch of 1

        # Manually set weights and biases to ensure non-zero gradients if possible
        layer.weights = np.array([[0.5], [0.5]]) # w1, w2
        layer.biases = np.array([[0.1]])         # b
        # Linear output Z = 1.0*0.5 + 2.0*0.5 + 0.1 = 0.5 + 1.0 + 0.1 = 1.6

        initial_weights = layer.weights.copy()
        initial_biases = layer.biases.copy()

        # Perform a forward pass to set layer.input
        # For no activation, layer.output is Z
        layer.forward(input_data)

        # Assume output_gradient (dL/dZ for this layer as no activation) is some value
        output_gradient_from_next_layer = np.array([[0.5]]) # dL/dZ
        learning_rate = 0.1

        # Call backward
        # dL/dW = X.T @ (dL/dZ) = [[1],[2]] @ [[0.5]] = [[0.5], [1.0]]
        # dL/dB = sum(dL/dZ) = 0.5
        # new_W = old_W - lr * dL/dW = [[0.5],[0.5]] - 0.1 * [[0.5],[1.0]] = [[0.5-0.05], [0.5-0.1]] = [[0.45], [0.4]]
        # new_B = old_B - lr * dL/dB = 0.1 - 0.1 * 0.5 = 0.1 - 0.05 = 0.05
        input_gradient_to_prev_layer = layer.backward(output_gradient_from_next_layer, learning_rate)

        # Check shapes
        self.assertEqual(input_gradient_to_prev_layer.shape, input_data.shape)
        self.assertEqual(layer.weights.shape, initial_weights.shape)

        # Check if weights and biases were updated
        self.assertFalse(np.allclose(layer.weights, initial_weights), "Weights were not updated or update is too small.")
        self.assertFalse(np.allclose(layer.biases, initial_biases), "Biases were not updated or update is too small.")

        # Check specific values for this controlled scenario
        expected_new_weights = np.array([[0.45], [0.4]])
        expected_new_biases = np.array([[0.05]])
        np.testing.assert_array_almost_equal(layer.weights, expected_new_weights, decimal=5)
        np.testing.assert_array_almost_equal(layer.biases, expected_new_biases, decimal=5)


    def test_dense_layer_get_config(self):
        layer_no_activation = DenseLayer(input_size=3, output_size=2)
        config1 = layer_no_activation.get_config()
        self.assertEqual(config1, {
            'type': 'dense', 'input_size': 3, 'output_size': 2, 'activation': None
        })

        relu_activation = ReLU()
        layer_with_activation = DenseLayer(input_size=3, output_size=2, activation=relu_activation)
        config2 = layer_with_activation.get_config()
        self.assertEqual(config2, {
            'type': 'dense', 'input_size': 3, 'output_size': 2, 'activation': 'relu'
        })

    def test_get_layer_factory(self):
        config_dense_relu = {
            'type': 'dense', 'input_size': 5, 'output_size': 10, 'activation': 'relu'
        }
        layer = get_layer(config_dense_relu)
        self.assertIsInstance(layer, DenseLayer)
        self.assertEqual(layer.input_size, 5)
        self.assertEqual(layer.output_size, 10)
        self.assertIsInstance(layer.activation, ReLU)

        config_dense_none = {
            'type': 'dense', 'input_size': 3, 'output_size': 1, 'activation': None
        }
        layer_none = get_layer(config_dense_none)
        self.assertIsInstance(layer_none, DenseLayer)
        self.assertIsNone(layer_none.activation)

        with self.assertRaises(ValueError):
            get_layer({'type': 'unknown_layer'})

if __name__ == '__main__':
    unittest.main()
