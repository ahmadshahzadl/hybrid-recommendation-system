"""
Collaborative Filtering Module

This module implements Collaborative Filtering using Singular Value Decomposition (SVD)
for recommendation systems.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os

# Try to import surprise, fallback to scikit-learn if not available
try:
    from surprise import SVD, Reader, Dataset
    from surprise.model_selection import train_test_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    logger = logging.getLogger(__name__)
    logger.warning("surprise library not available. Using scikit-learn TruncatedSVD as fallback.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SklearnSVDWrapper:
    """
    Wrapper class for scikit-learn TruncatedSVD to mimic Surprise API.
    This class can be pickled for model persistence.
    """
    def __init__(self, user_factors, product_factors, user_ids, product_ids, mean_rating):
        self.user_factors = user_factors
        self.product_factors = product_factors
        self.user_ids = user_ids
        self.product_ids = product_ids
        self.mean_rating = mean_rating
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    def predict(self, user_id, product_id):
        class Prediction:
            def __init__(self, est):
                self.est = est
        
        if user_id in self.user_id_to_idx and product_id in self.product_id_to_idx:
            user_idx = self.user_id_to_idx[user_id]
            product_idx = self.product_id_to_idx[product_id]
            pred = np.dot(self.user_factors[user_idx], self.product_factors[product_idx])
            pred = np.clip(pred + self.mean_rating, 1.0, 5.0)
        else:
            pred = self.mean_rating
        
        return Prediction(pred)


def train_collaborative_filtering(df: pd.DataFrame, model_path: str = None, 
                                  test_size: float = 0.2, random_state: int = 42):
    """
    Train the Collaborative Filtering model using Singular Value Decomposition (SVD).

    Args:
        df (pd.DataFrame): DataFrame containing User_ID, Product_ID, and Rating columns
        model_path (str, optional): Path to save the trained model
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random state for reproducibility (default: 42)

    Returns:
        Trained SVD model (surprise.SVD or sklearn-based wrapper)

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    logger.info("Training Collaborative Filtering model...")

    # Validate required columns
    required_cols = ['User_ID', 'Product_ID', 'Rating of the product']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare data
    try:
        # Create a copy with only required columns
        rating_data = df[required_cols].copy()
        
        # Rename columns
        rating_data.columns = ['User_ID', 'Product_ID', 'Rating']
        
        # Remove any rows with invalid ratings
        rating_data = rating_data.dropna(subset=['Rating'])
        
        # Ensure ratings are within valid range (1-5)
        rating_data['Rating'] = rating_data['Rating'].clip(lower=1, upper=5)
        
        # Check if we have enough data
        if len(rating_data) < 10:
            raise ValueError("Insufficient data for training. Need at least 10 ratings.")

        logger.info(f"Prepared {len(rating_data)} ratings for training")
        logger.info(f"Rating range: {rating_data['Rating'].min():.2f} - {rating_data['Rating'].max():.2f}")

        if SURPRISE_AVAILABLE:
            # Use Surprise library
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(rating_data, reader)
            trainset = data.build_full_trainset()
            logger.info(f"Training set size: {trainset.n_users} users, {trainset.n_items} items, {trainset.n_ratings} ratings")
            
            model_cf = SVD(random_state=random_state, n_epochs=20, lr_all=0.005, reg_all=0.02)
            model_cf.fit(trainset)
            logger.info("Collaborative Filtering model trained successfully (using Surprise)")
        else:
            # Use scikit-learn TruncatedSVD as fallback
            logger.info("Using scikit-learn TruncatedSVD as fallback")
            
            # Create user-item matrix
            user_item_matrix = rating_data.pivot_table(
                index='User_ID', 
                columns='Product_ID', 
                values='Rating', 
                fill_value=0
            )
            
            # Store mapping for later predictions
            user_ids = user_item_matrix.index.values
            product_ids = user_item_matrix.columns.values
            
            # Apply SVD
            n_components = min(50, min(len(user_ids), len(product_ids)) - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=random_state)
            user_factors = svd.fit_transform(user_item_matrix.values)
            product_factors = svd.components_.T
            
            # Use the wrapper class defined at module level
            mean_rating = rating_data['Rating'].mean()
            model_cf = SklearnSVDWrapper(user_factors, product_factors, user_ids, product_ids, mean_rating)
            
            logger.info(f"Training set size: {len(user_ids)} users, {len(product_ids)} items")
            logger.info("Collaborative Filtering model trained successfully (using scikit-learn)")

        # Save model if path provided
        if model_path:
            save_model(model_cf, model_path)

        return model_cf

    except Exception as e:
        logger.error(f"Error training Collaborative Filtering model: {str(e)}")
        raise


def predict_rating(model_cf, user_id: int, product_id: int) -> float:
    """
    Predict the rating for a given user and product using the trained Collaborative Filtering model.

    Args:
        model_cf (SVD): Trained SVD model
        user_id (int): User ID
        product_id (int): Product ID

    Returns:
        float: Predicted rating (clipped to 1-5 range)

    Raises:
        ValueError: If user_id or product_id is invalid
    """
    try:
        # Predict rating
        prediction = model_cf.predict(user_id, product_id)
        
        # Clip prediction to valid rating range
        predicted_rating = np.clip(prediction.est, 1.0, 5.0)
        
        return float(predicted_rating)
    
    except Exception as e:
        logger.warning(f"Error predicting rating for user {user_id}, product {product_id}: {str(e)}")
        # Return default rating if prediction fails
        return 3.0  # Neutral rating


def load_model(model_path: str):
    """
    Load a trained SVD model from disk.

    Args:
        model_path (str): Path to the saved model file

    Returns:
        SVD: Loaded SVD model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def save_model(model_cf, model_path: str) -> None:
    """
    Save a trained SVD model to disk.

    Args:
        model_cf (SVD): Trained SVD model
        model_path (str): Path to save the model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_cf, f)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise


def get_top_recommendations(model_cf, user_id: int, product_ids: list, top_n: int = 10) -> list:
    """
    Get top N product recommendations for a user.

    Args:
        model_cf (SVD): Trained SVD model
        user_id (int): User ID
        product_ids (list): List of product IDs to consider
        top_n (int): Number of top recommendations to return

    Returns:
        list: List of tuples (product_id, predicted_rating) sorted by rating
    """
    predictions = []
    for product_id in product_ids:
        try:
            rating = predict_rating(model_cf, user_id, product_id)
            predictions.append((product_id, rating))
        except Exception as e:
            logger.warning(f"Skipping product {product_id}: {str(e)}")
            continue

    # Sort by rating (descending) and return top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

