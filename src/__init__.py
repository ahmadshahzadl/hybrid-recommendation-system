"""
Hybrid Recommendation System Package

This package implements a hybrid recommendation system combining
Collaborative Filtering and Content-Based Filtering approaches.
"""

__version__ = "1.0.0"
__author__ = "AI Lab"

from .data_processing import load_data, preprocess_data
from .collaborative_filtering import train_collaborative_filtering, predict_rating
from .content_based_filtering import calculate_content_similarity, get_similar_products
from .hybrid_model import hybrid_recommendation

__all__ = [
    'load_data',
    'preprocess_data',
    'train_collaborative_filtering',
    'predict_rating',
    'calculate_content_similarity',
    'get_similar_products',
    'hybrid_recommendation',
]

