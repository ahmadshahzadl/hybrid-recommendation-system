"""
Hybrid Model Module

This module combines Collaborative Filtering and Content-Based Filtering
to create a hybrid recommendation system.
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from functools import lru_cache
from typing import Optional, Tuple

from .collaborative_filtering import train_collaborative_filtering, predict_rating, load_model, save_model
from .content_based_filtering import (
    calculate_content_similarity, 
    get_content_prediction,
    load_similarity_matrix,
    save_similarity_matrix
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridRecommendationSystem:
    """
    Hybrid Recommendation System combining Collaborative Filtering and Content-Based Filtering.
    """
    
    def __init__(self, data_path: str = 'data/processed_data.csv',
                 cf_model_path: str = 'models/cf_model.pkl',
                 cb_similarity_path: str = 'models/cb_similarity.csv',
                 alpha: float = 0.5):
        """
        Initialize the Hybrid Recommendation System.

        Args:
            data_path (str): Path to processed data CSV
            cf_model_path (str): Path to save/load Collaborative Filtering model
            cb_similarity_path (str): Path to save/load Content-Based similarity matrix
            alpha (float): Weight for Collaborative Filtering (0-1). 
                          Content-Based weight = 1 - alpha
        """
        self.data_path = data_path
        self.cf_model_path = cf_model_path
        self.cb_similarity_path = cb_similarity_path
        self.alpha = alpha
        
        self.df = None
        self.cf_model = None
        self.cb_similarity_df = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load the processed dataset."""
        if self.df is None:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Processed data file not found: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path}, shape: {self.df.shape}")
        return self.df
    
    def load_or_train_cf_model(self, retrain: bool = False) -> None:
        """Load or train the Collaborative Filtering model."""
        if self.cf_model is not None and not retrain:
            return
        
        # Try to load existing model
        if os.path.exists(self.cf_model_path) and not retrain:
            try:
                self.cf_model = load_model(self.cf_model_path)
                logger.info("Loaded existing CF model")
                return
            except Exception as e:
                logger.warning(f"Could not load CF model: {e}. Training new model...")
        
        # Train new model
        df = self.load_data()
        self.cf_model = train_collaborative_filtering(df, model_path=self.cf_model_path)
        logger.info("CF model trained and saved")
    
    def load_or_calculate_cb_similarity(self, retrain: bool = False) -> None:
        """Load or calculate the Content-Based similarity matrix."""
        if self.cb_similarity_df is not None and not retrain:
            return
        
        # Try to load existing similarity matrix
        if os.path.exists(self.cb_similarity_path) and not retrain:
            try:
                self.cb_similarity_df = load_similarity_matrix(self.cb_similarity_path)
                logger.info("Loaded existing CB similarity matrix")
                return
            except Exception as e:
                logger.warning(f"Could not load CB similarity matrix: {e}. Calculating new matrix...")
        
        # Calculate new similarity matrix
        df = self.load_data()
        self.cb_similarity_df = calculate_content_similarity(
            df, 
            similarity_matrix_path=self.cb_similarity_path
        )
        logger.info("CB similarity matrix calculated and saved")
    
    def initialize(self, retrain: bool = False) -> None:
        """Initialize all components of the hybrid system."""
        logger.info("Initializing Hybrid Recommendation System...")
        self.load_data()
        self.load_or_train_cf_model(retrain=retrain)
        self.load_or_calculate_cb_similarity(retrain=retrain)
        logger.info("Hybrid Recommendation System initialized successfully")
    
    def predict(self, user_id: int, product_id: int, 
                user_product_history: Optional[list] = None) -> float:
        """
        Predict hybrid recommendation score for a user-product pair.

        Args:
            user_id (int): User ID
            product_id (int): Product ID
            user_product_history (list, optional): List of product IDs user has interacted with

        Returns:
            float: Hybrid recommendation score (normalized to 0-1 range)
        """
        if self.cf_model is None or self.cb_similarity_df is None:
            raise ValueError("System not initialized. Call initialize() first.")
        
        try:
            # Collaborative Filtering prediction (normalized to 0-1 range)
            cf_prediction = predict_rating(self.cf_model, user_id, product_id)
            cf_score = (cf_prediction - 1.0) / 4.0  # Normalize from 1-5 to 0-1
            
            # Content-Based prediction
            cb_score = get_content_prediction(
                self.cb_similarity_df, 
                product_id, 
                user_product_history
            )
            
            # Hybrid score
            hybrid_score = self.alpha * cf_score + (1 - self.alpha) * cb_score
            
            # Ensure score is in valid range
            hybrid_score = np.clip(hybrid_score, 0.0, 1.0)
            
            return float(hybrid_score)
            
        except Exception as e:
            logger.error(f"Error predicting for user {user_id}, product {product_id}: {e}")
            return 0.5  # Default neutral score


# Global instance for backward compatibility
_hybrid_system = None


def hybrid_recommendation(user_id: int, product_id: int, alpha: float = 0.5,
                         data_path: str = 'data/processed_data.csv',
                         retrain: bool = False) -> float:
    """
    Combine Collaborative Filtering and Content-Based Filtering predictions 
    to generate a hybrid recommendation score.

    Args:
        user_id (int): User ID
        product_id (int): Product ID
        alpha (float): Weight for Collaborative Filtering (0-1). Default: 0.5
        data_path (str): Path to processed data CSV
        retrain (bool): Whether to retrain models even if cached versions exist

    Returns:
        float: Hybrid recommendation score (0-1 range)
    """
    global _hybrid_system
    
    # Initialize system if needed
    if _hybrid_system is None or _hybrid_system.alpha != alpha or _hybrid_system.data_path != data_path:
        _hybrid_system = HybridRecommendationSystem(
            data_path=data_path,
            alpha=alpha
        )
        _hybrid_system.initialize(retrain=retrain)
    elif retrain:
        _hybrid_system.initialize(retrain=True)
    
    # Get user's product history if available
    df = _hybrid_system.load_data()
    user_history = df[df['User_ID'] == user_id]['Product_ID'].tolist()
    
    return _hybrid_system.predict(user_id, product_id, user_product_history=user_history)


def get_top_recommendations(user_id: int, top_n: int = 10, alpha: float = 0.5,
                           data_path: str = 'data/processed_data.csv') -> list:
    """
    Get top N product recommendations for a user.

    Args:
        user_id (int): User ID
        top_n (int): Number of recommendations to return
        alpha (float): Weight for Collaborative Filtering (0-1)
        data_path (str): Path to processed data CSV

    Returns:
        list: List of tuples (product_id, hybrid_score) sorted by score
    """
    global _hybrid_system
    
    # Initialize system if needed
    if _hybrid_system is None or _hybrid_system.alpha != alpha or _hybrid_system.data_path != data_path:
        _hybrid_system = HybridRecommendationSystem(data_path=data_path, alpha=alpha)
        _hybrid_system.initialize()
    
    # Get all unique products
    df = _hybrid_system.load_data()
    all_products = df['Product_ID'].unique().tolist()
    
    # Get user's product history
    user_history = df[df['User_ID'] == user_id]['Product_ID'].tolist()
    
    # Calculate scores for all products
    recommendations = []
    for product_id in all_products:
        score = _hybrid_system.predict(user_id, product_id, user_product_history=user_history)
        recommendations.append((product_id, score))
    
    # Sort by score and return top N
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

