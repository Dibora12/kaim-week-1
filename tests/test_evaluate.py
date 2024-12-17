import unittest
import numpy as np
from src.evaluate import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.evaluator = ModelEvaluator()
        
        # Create perfect predictions case
        self.y_true_perfect = np.array([0, 1, 0, 1, 0])
        self.y_pred_perfect = np.array([0, 1, 0, 1, 0])
        
        # Create imperfect predictions case
        self.y_true_imperfect = np.array([0, 1, 0, 1, 0])
        self.y_pred_imperfect = np.array([0, 1, 1, 1, 0])

    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        results = self.evaluator.evaluate_model(
            self.y_true_perfect, 
            self.y_pred_perfect
        )
        
        # Check accuracy
        self.assertEqual(results['accuracy'], 1.0)
        
        # Check if classification report exists and contains expected values
        self.assertIn('classification_report', results)
        self.assertIsInstance(results['classification_report'], str)
        self.assertIn('accuracy', results['classification_report'])
        self.assertIn('1.00', results['classification_report'])

    def test_imperfect_predictions(self):
        """Test evaluation with imperfect predictions."""
        results = self.evaluator.evaluate_model(
            self.y_true_imperfect, 
            self.y_pred_imperfect
        )
        
        # Check accuracy (4/5 = 0.8)
        self.assertEqual(results['accuracy'], 0.8)
        
        # Check classification report
        self.assertIn('classification_report', results)
        self.assertIsInstance(results['classification_report'], str)
        self.assertIn('0.80', results['classification_report'])

    def test_invalid_input(self):
        """Test evaluation with invalid input."""
        # Test with different length arrays
        y_true_invalid = np.array([0, 1, 0])
        y_pred_invalid = np.array([0, 1, 0, 1])
        
        with self.assertRaises(ValueError):
            self.evaluator.evaluate_model(y_true_invalid, y_pred_invalid)
        
        # Test with non-binary classifications
        y_true_invalid = np.array([0, 1, 2])
        y_pred_invalid = np.array([0, 1, 2])
        
        results = self.evaluator.evaluate_model(y_true_invalid, y_pred_invalid)
        self.assertEqual(results['accuracy'], 1.0)  # Should still work for multi-class

if __name__ == '__main__':
    unittest.main() 