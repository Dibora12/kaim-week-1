"""
Helper functions for common operations in the machine learning pipeline.
Includes utilities for data validation, file operations, and data transformations.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Union, List, Optional, Tuple
from pathlib import Path


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class FileOperationError(Exception):
    """Custom exception for file operation errors."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Validate a pandas DataFrame against specified requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        numeric_columns: List of columns that must contain numeric data
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        if df is None or df.empty:
            error_msg = "DataFrame is empty or None"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
        
        if numeric_columns:
            non_numeric = [col for col in numeric_columns if not np.issubdtype(df[col].dtype, np.number)]
            if non_numeric:
                error_msg = f"Non-numeric data in columns: {non_numeric}"
                logger.error(error_msg)
                raise DataValidationError(error_msg)
        
        logger.info("DataFrame validation successful")
        return True, "Validation successful"
        
    except DataValidationError as e:
        return False, str(e)
    except Exception as e:
        logger.exception("Unexpected error during DataFrame validation")
        return False, f"Validation failed: {str(e)}"


def safe_file_path(file_path: Union[str, Path], create_dir: bool = True) -> Path:
    """
    Ensure a file path is valid and optionally create the directory structure.
    
    Args:
        file_path: Path to validate
        create_dir: Whether to create the directory if it doesn't exist
    
    Returns:
        Path object representing the validated path
        
    Raises:
        FileOperationError: If path creation fails
    """
    try:
        path = Path(file_path)
        if create_dir:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created/verified: {path.parent}")
            except PermissionError as e:
                error_msg = f"Permission denied creating directory: {path.parent}"
                logger.error(error_msg)
                raise FileOperationError(error_msg) from e
            except Exception as e:
                error_msg = f"Failed to create directory: {path.parent}"
                logger.error(error_msg)
                raise FileOperationError(error_msg) from e
        return path
    except Exception as e:
        logger.exception("Unexpected error in safe_file_path")
        raise FileOperationError(f"Path operation failed: {str(e)}") from e


def load_data_safe(
    file_path: Union[str, Path],
    file_type: str = "csv",
    **kwargs
) -> Optional[pd.DataFrame]:
    """
    Safely load data from a file with error handling.
    
    Args:
        file_path: Path to the data file
        file_type: Type of file ('csv', 'excel', 'parquet')
        **kwargs: Additional arguments passed to the pandas read function
    
    Returns:
        DataFrame if successful, None if failed
        
    Raises:
        FileOperationError: If file loading fails
    """
    try:
        path = safe_file_path(file_path, create_dir=False)
        if not path.exists():
            error_msg = f"File not found: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading {file_type} file: {path}")
        if file_type.lower() == "csv":
            return pd.read_csv(path, **kwargs)
        elif file_type.lower() == "excel":
            return pd.read_excel(path, **kwargs)
        elif file_type.lower() == "parquet":
            return pd.read_parquet(path, **kwargs)
        else:
            error_msg = f"Unsupported file type: {file_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    except (FileNotFoundError, ValueError) as e:
        raise FileOperationError(str(e))
    except Exception as e:
        logger.exception(f"Error loading data from {file_path}")
        raise FileOperationError(f"Failed to load data: {str(e)}")


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    n_std: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the z-score method.
    
    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers
        n_std: Number of standard deviations to use as threshold
    
    Returns:
        DataFrame with outliers removed
    """
    try:
        df_clean = df.copy()
        total_rows = len(df)
        
        for column in columns:
            if not np.issubdtype(df[column].dtype, np.number):
                logger.warning(f"Skipping non-numeric column: {column}")
                continue
                
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df_clean = df_clean[z_scores < n_std]
            
            removed_rows = total_rows - len(df_clean)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} outliers from column {column}")
        
        return df_clean
        
    except Exception as e:
        logger.exception("Error removing outliers")
        raise ValueError(f"Failed to remove outliers: {str(e)}")


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "label"
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables using specified method.
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode
        method: Encoding method ('label' or 'onehot')
    
    Returns:
        Tuple of (encoded DataFrame, encoding mappings)
    """
    try:
        if method.lower() not in ["label", "onehot"]:
            error_msg = f"Unsupported encoding method: {method}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        df_encoded = df.copy()
        mappings = {}
        
        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column not found, skipping: {column}")
                continue
                
            logger.info(f"Encoding column {column} using {method} encoding")
            if method.lower() == "label":
                mapping = {val: idx for idx, val in enumerate(df[column].unique())}
                df_encoded[column] = df[column].map(mapping)
                mappings[column] = mapping
            else:  # onehot
                dummies = pd.get_dummies(df[column], prefix=column)
                df_encoded = pd.concat([df_encoded.drop(column, axis=1), dummies], axis=1)
                mappings[column] = dummies.columns.tolist()
        
        return df_encoded, mappings
        
    except Exception as e:
        logger.exception("Error encoding categorical variables")
        raise ValueError(f"Failed to encode categorical variables: {str(e)}") 