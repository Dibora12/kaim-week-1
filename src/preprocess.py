from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd
import numpy as np

class DataPreprocessor:
    """Class to handle all data preprocessing steps."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        
    def preprocess_data(self, 
                       data: pd.DataFrame,
                       target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of processed features and labels
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def preprocess_new_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess new data using fitted scaler.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Scaled features
        """
        return self.scaler.transform(data)

