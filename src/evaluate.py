from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any
import numpy as np

class ModelEvaluator:
    """Class to evaluate model performance."""
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, 
                      y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
