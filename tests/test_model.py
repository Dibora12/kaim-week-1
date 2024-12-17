import unittest
import numpy as np
import os
from src.model import MLModel

class TestMLModel(unittest.TestCase):
    def setUp(self):
        """Set up test data and model."""
        self.model = MLModel()
        # Create sample data
        np.random.seed(42)
        self.X = np.random.rand(100, 4)  # 100 samples, 4 features
        self.y = np.random.randint(0, 2, 100)  # Binary classification

    def test_model_initialization(self):
        """Test model initialization with default and custom parameters."""
        # Test default parameters
        model_default = MLModel()
        self.assertEqual(model_default.model_params['n_estimators'], 100)
        self.assertEqual(model_default.model_params['random_state'], 42)

        # Test custom parameters
        custom_params = {'n_estimators': 200, 'max_depth': 5, 'random_state': 42}
        model_custom = MLModel(model_params=custom_params)
        self.assertEqual(model_custom.model_params['n_estimators'], 200)
        self.assertEqual(model_custom.model_params['max_depth'], 5)

    def test_train_and_predict(self):
        """Test model training and prediction."""
        # Train model
        self.model.train(self.X, self.y)
        
        # Make predictions
        predictions = self.model.predict(self.X)
        
        # Check predictions
        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))  # Binary predictions

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train model
        self.model.train(self.X, self.y)
        
        # Save model
        test_model_path = 'test_model.joblib'
        self.model.save_model(test_model_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(test_model_path))
        
        # Load model in new instance
        new_model = MLModel()
        new_model.load_model(test_model_path)
        
        # Compare predictions
        original_predictions = self.model.predict(self.X)
        loaded_predictions = new_model.predict(self.X)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        
        # Clean up
        os.remove(test_model_path)

    def test_invalid_input(self):
        """Test model behavior with invalid input."""
        # Test training with invalid shapes
        X_invalid = np.random.rand(10, 3)
        y_invalid = np.random.randint(0, 2, 5)
        
        with self.assertRaises(ValueError):
            self.model.train(X_invalid, y_invalid)

if __name__ == '__main__':
    unittest.main() 