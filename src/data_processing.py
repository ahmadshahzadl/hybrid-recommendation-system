"""
Data Processing Module

This module handles data loading, cleaning, and preprocessing for the
hybrid recommendation system.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from the CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset from {file_path}")
        logger.info(f"Dataset shape: {df.shape}")

        if df.empty:
            raise ValueError("Dataset is empty")

        return df
    except pd.errors.EmptyDataError:
        raise ValueError(f"File {file_path} is empty or invalid")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by handling missing values, encoding categorical variables,
    and scaling numerical features. Also adds User_ID and Product_ID columns if missing.

    Args:
        df (pd.DataFrame): Raw dataset

    Returns:
        pd.DataFrame: Preprocessed dataset with encoded features
    """
    logger.info("Starting data preprocessing...")
    df = df.copy()

    # Add User_ID and Product_ID if they don't exist
    # We'll create them based on unique combinations of features
    if 'User_ID' not in df.columns:
        logger.info("User_ID column not found. Creating based on user features...")
        # Create User_ID based on unique combinations of user-related features
        user_features = ['Gender', 'Median purchasing price (in rupees)', 
                        'Number of clicks on similar products', 
                        'Number of similar products purchased so far']
        # Use hash of user features to create User_ID
        df['User_ID'] = pd.util.hash_pandas_object(
            df[user_features].fillna(''), index=False
        ).abs() % 10000  # Limit to reasonable range

    if 'Product_ID' not in df.columns:
        logger.info("Product_ID column not found. Creating based on product features...")
        # Create Product_ID based on unique combinations of product-related features
        product_features = ['Brand of the product', 'Price of the product', 
                          'Rating of the product']
        # Use hash of product features to create Product_ID
        df['Product_ID'] = pd.util.hash_pandas_object(
            df[product_features].fillna(''), index=False
        ).abs() % 10000  # Limit to reasonable range

    # Fill missing values for numerical columns with the mean
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            logger.info(f"Filled missing values in {col} with mean: {mean_val:.2f}")

    # Fill missing values for categorical columns with the mode
    categorical_cols = ['Gender', 'Holiday', 'Season', 'Geographical locations']
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"Filled missing values in {col} with mode: {mode_val}")

    # Encode categorical variables using one-hot encoding
    logger.info("Encoding categorical variables...")
    categorical_cols_to_encode = [col for col in ['Gender', 'Holiday', 'Season', 'Geographical locations'] 
                                  if col in df.columns]
    
    if categorical_cols_to_encode:
        df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    else:
        df_encoded = df.copy()

    # Normalize numerical columns (excluding User_ID and Product_ID)
    numerical_cols_to_scale = [
        'Number of clicks on similar products',
        'Number of similar products purchased so far',
        'Median purchasing price (in rupees)',
        'Rating of the product',
        'Price of the product'
    ]
    
    numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in df_encoded.columns]
    
    if numerical_cols_to_scale:
        logger.info(f"Scaling numerical columns: {numerical_cols_to_scale}")
        scaler = StandardScaler()
        df_encoded[numerical_cols_to_scale] = scaler.fit_transform(
            df_encoded[numerical_cols_to_scale]
        )

    logger.info(f"Preprocessing complete. Final shape: {df_encoded.shape}")
    return df_encoded


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataset to a CSV file.

    Args:
        df (pd.DataFrame): Processed dataset
        output_path (str): Path to save the processed data
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise


if __name__ == "__main__":
    """Main execution for data preprocessing"""
    import sys

    # Default paths
    input_path = 'data/dataset.csv'
    output_path = 'data/processed_data.csv'

    # Allow command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    try:
        # Load and preprocess data
        logger.info(f"Loading data from {input_path}...")
        df = load_data(input_path)
        
        logger.info("Preprocessing data...")
        df_processed = preprocess_data(df)
        
        # Save processed data
        save_processed_data(df_processed, output_path)
        
        logger.info("Data preprocessing completed successfully!")
        print(f"\nProcessed data saved to: {output_path}")
        print(f"Shape: {df_processed.shape}")
        print(f"Columns: {list(df_processed.columns)}")
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        sys.exit(1)

