import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        # This method should be overridden by concrete layer types
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        # This method should be overridden by concrete layer types
        raise NotImplementedError

class DenseLayer(Layer):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation # This should be an activation object

    def get_config(self):
        return {
            'type': 'dense',
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation.get_config()['name'] if self.activation else None
        }

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        if self.activation:
            self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        if self.activation:
            output_gradient = self.activation.backward(output_gradient)

        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        # Update parameters
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient

# Factory function for layers (currently only DenseLayer)
# Assumes get_activation is available from src.activations
from .activations import get_activation

def get_layer(config):
    layer_type = config.get('type', '').lower()

    if layer_type == 'dense':
        activation_name = config.get('activation')
        activation_obj = get_activation(activation_name) # Factory for activations
        return DenseLayer(
            input_size=config['input_size'],
            output_size=config['output_size'],
            activation=activation_obj
        )
    # Add other layer types here if they are created (e.g., Conv2D, LSTM)
    # elif layer_type == 'conv2d':
    #     return Conv2DLayer(...)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
