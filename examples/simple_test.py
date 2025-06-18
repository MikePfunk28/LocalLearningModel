import numpy as np
import sys
import os

# Add src directory to Python path to allow direct imports of library components
# This assumes simple_test.py is in the examples/ directory and src/ is a sibling of examples/
# For a real package, you'd install the library and import normally.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary modules from the neural network library
from src.neural_network import NeuralNetwork
from src.layers import DenseLayer
from src.activations import ReLU, Softmax # Import specific activations
from src.losses import CrossEntropyLoss # Import a specific loss
from src.data_utils import load_csv, normalize, one_hot_encode # Import data utilities

def main():
    print("--- Simple Classification Example ---")

    # --- 1. Data Loading and Preprocessing ---
    print("\nStep 1: Data Loading and Preprocessing")
    # Define file path relative to this script's location
    # Assumes 'data/dummy_classification.csv' exists in the parent directory's 'data' folder
    csv_filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'dummy_classification.csv')
    target_column = 'target' # Specify the name of your target variable column

    # Load data from CSV
    # feature_columns_names can be specified if you only want a subset of columns, e.g., ['feature1', 'feature3']
    # If feature_columns_names is None, all columns except target_column are used.
    print(f"Loading data from: {csv_filepath}")
    X, y = load_csv(filepath=csv_filepath, target_column_name=target_column)
    print(f"Successfully loaded data. Features shape: {X.shape}, Target shape: {y.shape}")

    # Normalize features (Z-score normalization)
    # This helps with training stability and performance.
    # The mean and std are returned to be used for normalizing test data later, if any.
    X_normalized, data_mean, data_std = normalize(X)
    print(f"Features normalized. Original data mean (approx): {np.mean(X, axis=0)}, Original data std (approx): {np.std(X, axis=0)}")
    print(f"Using calculated mean for normalization: {data_mean}, and std: {data_std}")
    # print("Sample of normalized X (first 3 rows):\n", X_normalized[:3])

    # One-hot encode target labels for classification
    # For a target variable with K classes, this converts it to a K-dimensional binary vector.
    # num_classes can be inferred if labels are 0, 1, ..., K-1.
    # It's good practice to specify if known, especially if some classes might be missing in a small sample.
    # Our dummy data has classes 0 and 1, so num_classes = 2.
    y_one_hot = one_hot_encode(y, num_classes=2)
    print(f"Target labels one-hot encoded. Shape: {y_one_hot.shape}")
    # print("Sample of one-hot y (first 3 rows):\n", y_one_hot[:3])

    # --- 2. Model Definition ---
    print("\nStep 2: Model Definition")
    # Create a NeuralNetwork instance
    model = NeuralNetwork()

    # Add layers to the network:
    # - DenseLayer(input_size, output_size, activation_object)
    # - input_size for the first layer must match the number of features in X_normalized.
    # - output_size for the last layer must match the number of classes (from y_one_hot).
    # - ReLU is a common activation for hidden layers.
    # - Softmax is used for the output layer in multi-class classification to get probabilities.
    model.add_layer(DenseLayer(input_size=X_normalized.shape[1], output_size=16, activation=ReLU()))
    model.add_layer(DenseLayer(input_size=16, output_size=y_one_hot.shape[1], activation=Softmax()))

    # Print model summary
    print("Model Summary:")
    model.summary()

    # --- 3. Model Compilation ---
    print("\nStep 3: Model Compilation")
    # Compile the model by specifying the loss function and learning rate.
    # - CrossEntropyLoss is suitable for classification tasks with a Softmax output layer.
    # - learning_rate controls how much the weights are adjusted during training.
    model.compile(loss_function=CrossEntropyLoss(), learning_rate=0.05)
    print("Model compiled with CrossEntropyLoss and learning rate 0.05.")

    # --- 4. Model Training ---
    print("\nStep 4: Model Training")
    # Train the model using the normalized features and one-hot encoded labels.
    # - epochs: Number of times to iterate over the entire dataset.
    # - batch_size: Number of samples processed before updating weights.
    # - print_every: How often to print the loss during training.
    # For this small dataset, a small batch_size is fine.
    epochs = 500
    batch_size = 2
    model.train(X_normalized, y_one_hot, epochs=epochs, batch_size=batch_size, print_every=epochs//10)
    print("Training finished.")

    # --- 5. Model Evaluation (on training data) ---
    print("\nStep 5: Model Evaluation (on training data)")
    # Make predictions on the training data.
    # predict() returns probabilities for each class (due to Softmax).
    predictions_proba = model.predict(X_normalized)

    # Convert probabilities to class labels by taking the argmax.
    predicted_labels = np.argmax(predictions_proba, axis=1)
    # True labels can be obtained by argmax on one-hot or flattening the original y if it was (n,1)
    true_labels = y.flatten() # Or np.argmax(y_one_hot, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    final_loss = model.loss.forward(predictions_proba, y_one_hot) # Calculate final loss
    print(f"Final Loss on training data: {final_loss:.4f}")
    print(f"Accuracy on training data: {accuracy*100:.2f}%")

    print("\nDetailed predictions (Original Input -> Probabilities -> Predicted vs True):")
    for i in range(len(X_normalized)):
        print(f"Input: {X[i]}, Probs: [{predictions_proba[i,0]:.2f} {predictions_proba[i,1]:.2f}], \
Predicted: {predicted_labels[i]}, True: {true_labels[i]}")

    # --- 6. Model Saving ---
    print("\nStep 6: Model Saving")
    # Define the base path for saving the model. Files will be <base_path>_arch.json and <base_path>_weights.npz
    model_path_base = os.path.join(os.path.dirname(__file__), '..', 'models', 'simple_classifier_v2')
    print(f"Saving model to files starting with: {model_path_base}")
    model.save_model(model_path_base)
    print("Model saved successfully.")

    # --- 7. Model Loading ---
    print("\nStep 7: Model Loading")
    # Load the model from the saved files.
    # The sys.path.append at the top ensures custom classes can be found during loading.
    print(f"Loading model from files starting with: {model_path_base}")
    loaded_model = NeuralNetwork.load_model(model_path_base)
    print("Model loaded successfully.")
    print("Loaded Model Summary:")
    loaded_model.summary()

    # --- 8. Verification of Loaded Model ---
    print("\nStep 8: Verification of Loaded Model")
    # Make predictions with the loaded model.
    # Note: The loaded model is not compiled (loss function, learning rate are not restored by current load_model).
    # For predictions, this is fine. If further training is needed, loaded_model.compile(...) would be required.
    loaded_predictions_proba = loaded_model.predict(X_normalized)
    loaded_predicted_labels = np.argmax(loaded_predictions_proba, axis=1)

    loaded_accuracy = np.mean(loaded_predicted_labels == true_labels)
    print(f"Accuracy of loaded model on training data: {loaded_accuracy*100:.2f}%")

    # Compare if original and loaded model predictions (labels) are the same
    if np.array_equal(predicted_labels, loaded_predicted_labels):
        print("Predictions from original and loaded model are IDENTICAL.")
    else:
        print("Warning: Predictions from original and loaded model DIFFER.")
        print("Original model predicted labels:", predicted_labels)
        print("Loaded model predicted labels:  ", loaded_predicted_labels)

if __name__ == "__main__":
    main()
