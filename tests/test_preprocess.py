import unittest
import numpy as np
import pandas as pd
from src.preprocess import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.preprocessor = DataPreprocessor()
        # Create sample data
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })

    def test_preprocess_data(self):
        """Test the preprocess_data method."""
        X_scaled, y = self.preprocessor.preprocess_data(self.data, 'target')
        
        # Check output types
        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        
        # Check shapes
        self.assertEqual(X_scaled.shape, (5, 2))
        self.assertEqual(y.shape, (5,))
        
        # Check scaling (mean should be close to 0, std close to 1)
        self.assertAlmostEqual(X_scaled.mean(), 0, places=1)
        self.assertAlmostEqual(X_scaled.std(), 1, places=1)

    def test_preprocess_new_data(self):
        """Test the preprocess_new_data method."""
        # First fit the scaler with initial data
        self.preprocessor.preprocess_data(self.data, 'target')
        
        # Create new data
        new_data = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [12, 14]
        })
        
        # Preprocess new data
        X_new_scaled = self.preprocessor.preprocess_new_data(new_data)
        
        # Check output
        self.assertIsInstance(X_new_scaled, np.ndarray)
        self.assertEqual(X_new_scaled.shape, (2, 2))

    def test_invalid_target_column(self):
        """Test handling of invalid target column."""
        with self.assertRaises(KeyError):
            self.preprocessor.preprocess_data(self.data, 'nonexistent_target')

if __name__ == '__main__':
    unittest.main() 