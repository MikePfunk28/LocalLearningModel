import numpy as np
import pandas as pd

def load_csv(filepath, target_column_name, feature_columns_names=None):
    """
    Loads data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        target_column_name (str): The name of the column to be used as the target variable.
        feature_columns_names (list, optional): List of column names to be used as features.
                                                If None, all columns except the target are used.

    Returns:
        tuple: (X, y) where X is a numpy array of features and y is a numpy array of the target.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {e}")

    if target_column_name not in df.columns:
        raise ValueError(f"Error: Target column '{target_column_name}' not found in CSV.")

    if feature_columns_names:
        for col_name in feature_columns_names:
            if col_name not in df.columns:
                raise ValueError(f"Error: Feature column '{col_name}' not found in CSV.")
        X = df[feature_columns_names].values
    else:
        X = df.drop(columns=[target_column_name]).values

    y = df[target_column_name].values.reshape(-1, 1) # Ensure y is a column vector

    return X, y

def normalize(X, mean=None, std=None):
    """
    Normalizes features using Z-score normalization: (X - mean) / std.

    Args:
        X (np.ndarray): Input features (numpy array).
        mean (np.ndarray, optional): Mean to use for normalization. If None, calculated from X.
        std (np.ndarray, optional): Standard deviation to use for normalization. If None, calculated from X.

    Returns:
        tuple: (X_normalized, mean, std)
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input X must be a numpy array.")

    if X.size == 0: # Handle empty array
        return X, np.array([]), np.array([])

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    # Avoid division by zero if std is 0 for some features
    std_safe = np.where(std == 0, 1, std)

    X_normalized = (X - mean) / std_safe
    return X_normalized, mean, std

def one_hot_encode(y, num_classes=None):
    """
    Converts a vector of integer labels y into a one-hot encoded matrix.

    Args:
        y (np.ndarray): Input vector of integer labels (numpy array, shape (n_samples,) or (n_samples, 1)).
        num_classes (int, optional): Number of classes. If None, inferred from unique values in y.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (n_samples, num_classes).
    """
    if not isinstance(y, np.ndarray):
        raise ValueError("Input y must be a numpy array.")

    y_flat = y.flatten() # Ensure y is 1D for np.unique and indexing

    if num_classes is None:
        unique_labels = np.unique(y_flat)
        num_classes = len(unique_labels)
        # Create a mapping if labels are not 0-indexed or contiguous
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        y_indices = np.array([label_to_index[label] for label in y_flat])
    else:
        # Assume labels are 0 to num_classes-1
        if np.any((y_flat < 0) | (y_flat >= num_classes)):
            raise ValueError(f"If num_classes is provided, labels must be in range [0, {num_classes-1}]")
        y_indices = y_flat.astype(int)

    if y_indices.size == 0: # Handle empty array
        return np.empty((0, num_classes))

    one_hot_matrix = np.zeros((y_indices.size, num_classes))
    one_hot_matrix[np.arange(y_indices.size), y_indices] = 1

    return one_hot_matrix
