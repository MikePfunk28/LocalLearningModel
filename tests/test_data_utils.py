import unittest
import numpy as np
import pandas as pd # For creating expected DataFrames in tests
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_utils import load_csv, normalize, one_hot_encode

class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'dummy_data')
        self.csv_load_path = os.path.join(self.test_data_dir, 'dummy_load.csv')
        self.csv_features_path = os.path.join(self.test_data_dir, 'dummy_features.csv')
        # No need to create files here as they are done by tool_code earlier

    def test_load_csv_basic(self):
        X, y = load_csv(self.csv_load_path, target_column_name='target_col')

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape, (4, 2)) # 4 rows, 2 feature columns (col1, col2)
        self.assertEqual(y.shape, (4, 1)) # 4 rows, 1 target column
        np.testing.assert_array_equal(y.flatten(), np.array([10, 20, 30, 40]))
        # Check X dtypes if they are mixed (load_csv converts to common numerical type if possible or object)
        # In dummy_load.csv, col1 is int, col2 is object. Pandas will make X an object array.
        self.assertEqual(X.dtype, object)

    def test_load_csv_with_feature_selection(self):
        X, y = load_csv(self.csv_features_path,
                        target_column_name='target',
                        feature_columns_names=['f1', 'f3'])
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(y.shape, (3, 1))
        np.testing.assert_array_equal(X[:, 0], np.array([1.0, 0.1, 5.0])) # f1
        np.testing.assert_array_equal(X[:, 1], np.array([3.0, 0.3, 7.0])) # f3
        np.testing.assert_array_equal(y.flatten(), np.array(['A', 'B', 'A']))

    def test_load_csv_target_not_found(self):
        with self.assertRaisesRegex(ValueError, "Target column 'non_existent_target' not found"):
            load_csv(self.csv_load_path, target_column_name='non_existent_target')

    def test_load_csv_feature_not_found(self):
        with self.assertRaisesRegex(ValueError, "Feature column 'non_existent_feature' not found"):
            load_csv(self.csv_features_path,
                     target_column_name='target',
                     feature_columns_names=['f1', 'non_existent_feature'])

    def test_load_csv_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_csv("non_existent_file.csv", target_column_name='target')


    def test_normalize(self):
        X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        X_normalized, mean, std = normalize(X)

        expected_mean = np.array([2.0, 3.0, 4.0])
        expected_std = np.sqrt(np.array([2/3, 2/3, 2/3])) # Variance is ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3

        np.testing.assert_array_almost_equal(mean, expected_mean)
        np.testing.assert_array_almost_equal(std, expected_std)

        expected_normalized = (X - expected_mean) / expected_std
        np.testing.assert_array_almost_equal(X_normalized, expected_normalized)

        # Test with pre-computed mean and std
        X_new = np.array([[4.0, 5.0, 6.0]])
        X_new_normalized, _, _ = normalize(X_new, mean=mean, std=std)
        expected_new_normalized = (X_new - mean) / std
        np.testing.assert_array_almost_equal(X_new_normalized, expected_new_normalized)

    def test_normalize_zero_std(self):
        X = np.array([[1.0, 1.0], [1.0, 1.0]])
        X_normalized, mean, std = normalize(X)
        self.assertTrue(np.all(X_normalized == 0)) # Should be (X-mean)/1 if std is 0
        np.testing.assert_array_almost_equal(std, np.array([0.0, 0.0]))


    def test_one_hot_encode_infer_classes(self):
        y = np.array([0, 1, 2, 0, 1])
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        one_hot = one_hot_encode(y)
        np.testing.assert_array_equal(one_hot, expected)
        self.assertEqual(one_hot.shape[1], 3) # num_classes inferred

    def test_one_hot_encode_specify_classes(self):
        y = np.array([[1], [0], [2]]) # Column vector
        num_classes = 3
        expected = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        one_hot = one_hot_encode(y, num_classes=num_classes)
        np.testing.assert_array_equal(one_hot, expected)

    def test_one_hot_encode_non_contiguous_labels(self):
        y = np.array([10, 20, 10])
        # Expected: 10 -> 0, 20 -> 1 (if num_classes inferred)
        expected = np.array([
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        one_hot = one_hot_encode(y) # num_classes inferred
        np.testing.assert_array_equal(one_hot, expected)
        self.assertEqual(one_hot.shape[1], 2)

    def test_one_hot_encode_specify_classes_error(self):
        y = np.array([0, 1, 3]) # Label 3 is out of range for num_classes=3
        with self.assertRaisesRegex(ValueError, "labels must be in range"):
            one_hot_encode(y, num_classes=3)

    def test_one_hot_encode_empty_input(self):
        y = np.array([])
        one_hot_inferred = one_hot_encode(y)
        self.assertEqual(one_hot_inferred.shape, (0,0)) # No classes, no samples

        one_hot_specified = one_hot_encode(y, num_classes=3)
        self.assertEqual(one_hot_specified.shape, (0,3))


if __name__ == '__main__':
    unittest.main()
