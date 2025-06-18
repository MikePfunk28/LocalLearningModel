import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.learning_rate = None # Will store the learning rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss_function, learning_rate):
        """
        Configures the model for training.
        loss_function: Loss function instance
        learning_rate: Learning rate for the optimizer.
        """
        self.loss = loss_function
        self.learning_rate = learning_rate # This will act as our simple SGD optimizer for now

    def forward(self, X):
        """
        Performs a forward pass through the network.
        X: input data
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def train(self, X_train, y_train, epochs, batch_size=32, verbose=True, print_every=10):
        """
        Trains the neural network using mini-batch gradient descent.
        X_train: Training data
        y_train: Training labels
        epochs: Number of epochs to train for
        batch_size: Size of mini-batches
        verbose: Whether to print training progress
        print_every: Interval of epochs to print loss (if verbose is True)
        """
        if self.loss is None or self.learning_rate is None:
            raise ValueError("Model must be compiled before training. Call model.compile(loss_function, learning_rate).")

        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle training data at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Calculate loss for the batch
                batch_loss = self.loss.forward(y_pred, y_batch) # Changed from calculate to forward
                epoch_loss += batch_loss * X_batch.shape[0] # Weighted by batch size for accuracy if last batch is smaller

                # Calculate gradient of loss w.r.t. y_pred for the batch
                loss_gradient = self.loss.backward(y_pred, y_batch)

                # Backward pass for the batch
                current_grad = loss_gradient
                for layer in reversed(self.layers):
                    current_grad = layer.backward(current_grad, self.learning_rate)

            avg_epoch_loss = epoch_loss / num_samples
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

    def predict(self, X):
        """
        Predicts output for given input X.
        """
        # For prediction, we don't need gradients, so this is just a forward pass.
        # In more complex scenarios, this might involve disabling dropout, etc.
        return self.forward(X)

    def save_model(self, filepath_base):
        """
        Saves the model architecture and weights.
        filepath_base: Base path for saving. Architecture to <filepath_base>.json, weights to <filepath_base>_weights.npz
        """
        import json

        architecture = {
            'layers': [layer.get_config() for layer in self.layers],
            # Potentially save optimizer config, loss function type here too if needed for full state restoration
        }

        # Define file paths
        arch_filepath = f"{filepath_base}_arch.json"
        weights_filepath = f"{filepath_base}_weights.npz"

        # Add reference to weights file in architecture
        architecture['weights_file'] = weights_filepath

        # Save architecture
        with open(arch_filepath, 'w') as f:
            json.dump(architecture, f, indent=4)

        # Save weights
        weights_to_save = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                weights_to_save[f'layer_{i}_weights'] = layer.weights
                weights_to_save[f'layer_{i}_biases'] = layer.biases

        if weights_to_save: # Only save if there are weights
            np.savez(weights_filepath, **weights_to_save)
        print(f"Model architecture saved to {arch_filepath}")
        if weights_to_save:
            print(f"Model weights saved to {weights_filepath}")

    @staticmethod
    def load_model(filepath_base):
        """
        Loads a model architecture and weights.
        filepath_base: Base path from which to load. Architecture from <filepath_base>.json, weights from <filepath_base>_weights.npz
        """
        import json
        from .layers import get_layer # Factory function for layers
        # Note: get_activation is used by get_layer

        arch_filepath = f"{filepath_base}_arch.json"

        # Load architecture
        try:
            with open(arch_filepath, 'r') as f:
                architecture = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Architecture file not found: {arch_filepath}")

        weights_filepath = architecture.get('weights_file')
        if not weights_filepath:
            # Backward compatibility or if weights are not stored separately by convention
            weights_filepath = f"{filepath_base}_weights.npz"
            print(f"Warning: 'weights_file' not found in arch. Trying default: {weights_filepath}")


        loaded_nn = NeuralNetwork() # Create a new instance

        # Reconstruct layers
        for layer_config in architecture['layers']:
            layer = get_layer(layer_config) # Use factory to create layer
            loaded_nn.add_layer(layer)

        # Load and set weights
        try:
            # Check if weights file actually exists before trying to load
            import os
            if os.path.exists(weights_filepath):
                with np.load(weights_filepath, allow_pickle=True) as loaded_weights:
                    for i, layer in enumerate(loaded_nn.layers):
                        if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                            weights_key = f'layer_{i}_weights'
                            biases_key = f'layer_{i}_biases'
                            if weights_key in loaded_weights and biases_key in loaded_weights:
                                layer.weights = loaded_weights[weights_key]
                                layer.biases = loaded_weights[biases_key]
                            else:
                                print(f"Warning: Weights/biases for layer {i} not found in {weights_filepath}")
            elif architecture.get('layers'): # If there are layers, weights are expected.
                 print(f"Warning: Weights file {weights_filepath} not found, but architecture has layers. Model loaded without pretrained weights.")

        except FileNotFoundError:
            # This case should ideally be caught by os.path.exists, but as a fallback:
            print(f"Warning: Weights file {weights_filepath} not found. Model loaded without pretrained weights.")
        except Exception as e:
            print(f"Error loading weights from {weights_filepath}: {e}. Model loaded without pretrained weights.")

        # TODO: Compile step should ideally be restored too (loss function, learning rate/optimizer state)
        # For now, the loaded model will need to be compiled manually after loading if training is to be continued.
        print(f"Model loaded from {arch_filepath}. Compile it before training or use for prediction.")
        return loaded_nn

    def summary(self):
        """Prints a summary of the model architecture."""
        print("Model Architecture:")
        print("--------------------------------------------------")
        if not self.layers:
            print("No layers in model.")
            print("--------------------------------------------------")
            return

        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_config = layer.get_config()
            layer_type = layer_config.get('type', 'Unknown')
            output_shape_info = f"Output Units: {layer_config.get('output_size', 'N/A')}"

            params = 0
            if hasattr(layer, 'weights') and layer.weights is not None:
                params += layer.weights.size
            if hasattr(layer, 'biases') and layer.biases is not None:
                params += layer.biases.size
            total_params += params

            activation_info = f"Activation: {layer_config.get('activation', 'None')}"

            print(f"Layer {i+1}: {layer_type.capitalize()} | {output_shape_info} | {activation_info} | Params: {params}")
        print("--------------------------------------------------")
        print(f"Total Trainable Parameters: {total_params}")
        print("--------------------------------------------------")
