from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Any, Dict, Optional
import numpy as np

class MLModel:
    """Base ML model class that wraps scikit-learn estimators."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML model.
        
        Args:
            model_params: Dictionary of parameters for the model
        """
        self.model_params = model_params or {
            'n_estimators': 100,
            'random_state': 42
        }
        self.model = RandomForestClassifier(**self.model_params)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on given data.
        
        Args:
            X: Training features
            y: Training labels
        """
        self.model.fit(X, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Model predictions
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
