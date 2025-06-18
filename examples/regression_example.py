import numpy as np
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary modules from the neural network library
from src.neural_network import NeuralNetwork
from src.layers import DenseLayer
from src.activations import ReLU # Sigmoid, Tanh, Softmax also available
from src.losses import MeanSquaredError # CrossEntropyLoss also available
from src.data_utils import load_csv, normalize # one_hot_encode also available

def main():
    print("--- Simple Regression Example ---")

    # --- 1. Data Loading and Preprocessing ---
    print("\nStep 1: Data Loading and Preprocessing")
    csv_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'dummy_regression.csv')
    target_column = 'target_value'

    print(f"Loading data from: {csv_filepath}")
    X, y = load_csv(filepath=csv_filepath, target_column_name=target_column)
    print(f"Successfully loaded data. Features shape: {X.shape}, Target shape: {y.shape}")

    # Normalize features
    X_normalized, X_mean, X_std = normalize(X)
    print(f"Features normalized. Original X mean (approx): {np.mean(X, axis=0)}")
    # For regression, sometimes target variable is also normalized. We'll skip it here for simplicity,
    # but if y has a very large scale or is offset, normalizing it (and then denormalizing predictions) can help.
    # y_normalized, y_mean, y_std = normalize(y)

    # --- 2. Model Definition ---
    print("\nStep 2: Model Definition")
    model = NeuralNetwork()
    # For regression, the final layer typically has 1 unit (for a single target value)
    # and no activation function (or a 'linear' activation).
    model.add_layer(DenseLayer(input_size=X_normalized.shape[1], output_size=16, activation=ReLU()))
    model.add_layer(DenseLayer(input_size=16, output_size=8, activation=ReLU()))
    model.add_layer(DenseLayer(input_size=8, output_size=1, activation=None)) # Linear output layer

    print("Model Summary:")
    model.summary()

    # --- 3. Model Compilation ---
    print("\nStep 3: Model Compilation")
    # Use MeanSquaredError for regression tasks.
    model.compile(loss_function=MeanSquaredError(), learning_rate=0.01) # Adjusted learning rate for regression
    print("Model compiled with MeanSquaredError and learning rate 0.01.")

    # --- 4. Model Training ---
    print("\nStep 4: Model Training")
    epochs = 300 # Fewer epochs for this simple task
    batch_size = 2
    model.train(X_normalized, y, epochs=epochs, batch_size=batch_size, print_every=epochs//10)
    print("Training finished.")

    # --- 5. Model Evaluation (on training data) ---
    print("\nStep 5: Model Evaluation (on training data)")
    predictions = model.predict(X_normalized)

    # Calculate Mean Squared Error on training data
    # The loss object can compute this directly
    mse = model.loss.forward(predictions, y)
    print(f"Mean Squared Error on training data: {mse:.4f}")

    print("\nDetailed predictions (Original Input -> Predicted vs True):")
    for i in range(len(X_normalized)):
        print(f"Input: {X[i]}, Normalized Input: [{X_normalized[i,0]:.2f}, {X_normalized[i,1]:.2f}], \
Predicted: {predictions[i,0]:.2f}, True: {y[i,0]:.2f}")

    # --- 6. Model Saving ---
    print("\nStep 6: Model Saving")
    model_path_base = os.path.join(os.path.dirname(__file__), '..', 'models', 'simple_regressor')
    print(f"Saving model to files starting with: {model_path_base}")
    model.save_model(model_path_base)
    print("Model saved successfully.")

    # --- 7. Model Loading ---
    print("\nStep 7: Model Loading")
    print(f"Loading model from files starting with: {model_path_base}")
    loaded_model = NeuralNetwork.load_model(model_path_base)
    print("Model loaded successfully.")
    print("Loaded Model Summary:")
    loaded_model.summary()

    # --- 8. Verification of Loaded Model ---
    print("\nStep 8: Verification of Loaded Model")
    loaded_predictions = loaded_model.predict(X_normalized)

    loaded_mse = MeanSquaredError().forward(loaded_predictions, y) # Use a new MSE instance for calculation
    print(f"MSE of loaded model on training data: {loaded_mse:.4f}")

    # Compare if original and loaded model predictions are very close
    if np.allclose(predictions, loaded_predictions, atol=1e-5): # Check for near equality for float values
        print("Predictions from original and loaded model are SUFFICIENTLY CLOSE.")
    else:
        print("Warning: Predictions from original and loaded model DIFFER significantly.")
        # For detailed comparison:
        # for i in range(len(predictions)):
        # print(f"Orig: {predictions[i,0]:.4f}, Loaded: {loaded_predictions[i,0]:.4f}, Diff: {predictions[i,0] - loaded_predictions[i,0]:.4f}")


if __name__ == "__main__":
    main()
