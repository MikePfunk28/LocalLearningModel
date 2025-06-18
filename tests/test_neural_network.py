import unittest
import numpy as np
import sys
import os
import json # For checking saved arch file

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.neural_network import NeuralNetwork
from src.layers import DenseLayer
from src.activations import ReLU, Sigmoid # Using Sigmoid for simple binary-like output
from src.losses import MeanSquaredError # Simple loss for testing training step

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # Common setup for tests that need a simple network
        self.nn = NeuralNetwork()
        self.nn.add_layer(DenseLayer(input_size=2, output_size=3, activation=ReLU()))
        self.nn.add_layer(DenseLayer(input_size=3, output_size=1, activation=Sigmoid()))
        self.sample_input = np.array([[0.5, -0.5]])

    def test_add_layer(self):
        self.assertEqual(len(self.nn.layers), 2)
        self.assertIsInstance(self.nn.layers[0], DenseLayer)
        self.assertEqual(self.nn.layers[0].output_size, 3)
        self.assertIsInstance(self.nn.layers[0].activation, ReLU)
        self.assertIsInstance(self.nn.layers[1], DenseLayer)
        self.assertEqual(self.nn.layers[1].output_size, 1)
        self.assertIsInstance(self.nn.layers[1].activation, Sigmoid)

    def test_forward_pass(self):
        output = self.nn.forward(self.sample_input)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, 1)) # Batch of 1, 1 output neuron

    def test_predict_method(self):
        # Predict should be same as forward for now
        output_fwd = self.nn.forward(self.sample_input)
        output_pred = self.nn.predict(self.sample_input)
        np.testing.assert_array_almost_equal(output_pred, output_fwd)

    def test_compile_method(self):
        loss_fn = MeanSquaredError()
        lr = 0.01
        self.nn.compile(loss_function=loss_fn, learning_rate=lr)
        self.assertIs(self.nn.loss, loss_fn)
        self.assertEqual(self.nn.learning_rate, lr)

    def test_train_one_step(self):
        # Test if a single training step runs and loss changes
        X_train = np.array([[0.1, 0.2], [0.3, 0.4]])
        y_train = np.array([[0.0], [1.0]]) # Target for sigmoid output

        # Using a higher learning rate and a couple of epochs for the test
        # to increase likelihood of a measurable change in loss.
        self.nn.compile(loss_function=MeanSquaredError(), learning_rate=0.5)

        # Get initial predictions/loss
        initial_predictions = self.nn.predict(X_train)
        initial_loss = self.nn.loss.forward(initial_predictions, y_train)

        # Train for a few epochs
        self.nn.train(X_train, y_train, epochs=5, batch_size=X_train.shape[0], verbose=False)

        # Check if loss has changed
        final_predictions = self.nn.predict(X_train)
        final_loss = self.nn.loss.forward(final_predictions, y_train)

        self.assertNotEqual(initial_loss, final_loss, msg="Loss did not change after training steps.")


    def test_save_and_load_model(self):
        model_path_base = os.path.join(os.path.dirname(__file__), 'dummy_data', 'test_model_sl')

        # Compile and train for a few steps to have some trained weights
        self.nn.compile(loss_function=MeanSquaredError(), learning_rate=0.1) # Increased LR here too
        X_sample = np.array([[0.1, 0.2]])
        y_sample = np.array([[0.8]])
        self.nn.train(X_sample, y_sample, epochs=5, batch_size=1, verbose=False)

        original_predictions = self.nn.predict(self.sample_input)
        original_weights_l1 = self.nn.layers[0].weights.copy()
        original_biases_l1 = self.nn.layers[0].biases.copy()

        self.nn.save_model(model_path_base)

        # Check if files were created
        self.assertTrue(os.path.exists(f"{model_path_base}_arch.json"))
        self.assertTrue(os.path.exists(f"{model_path_base}_weights.npz"))

        loaded_nn = NeuralNetwork.load_model(model_path_base)

        # Verify architecture
        self.assertEqual(len(loaded_nn.layers), len(self.nn.layers))
        self.assertEqual(loaded_nn.layers[0].get_config(), self.nn.layers[0].get_config())
        self.assertEqual(loaded_nn.layers[1].get_config(), self.nn.layers[1].get_config())

        # Verify weights
        np.testing.assert_array_almost_equal(loaded_nn.layers[0].weights, original_weights_l1)
        np.testing.assert_array_almost_equal(loaded_nn.layers[0].biases, original_biases_l1)

        loaded_predictions = loaded_nn.predict(self.sample_input)
        np.testing.assert_array_almost_equal(loaded_predictions, original_predictions)

        # Clean up created files
        os.remove(f"{model_path_base}_arch.json")
        os.remove(f"{model_path_base}_weights.npz")

    def test_summary_method(self):
        # Just test if it runs without error
        try:
            self.nn.summary()
        except Exception as e:
            self.fail(f"nn.summary() raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
