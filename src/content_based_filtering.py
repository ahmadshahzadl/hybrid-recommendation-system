"""
Content-Based Filtering Module

This module implements Content-Based Filtering using Cosine Similarity
to recommend products based on item features.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_content_similarity(df: pd.DataFrame, 
                                 feature_columns: list = None,
                                 similarity_matrix_path: str = None) -> pd.DataFrame:
    """
    Calculate the Cosine Similarity between products based on item features.

    Args:
        df (pd.DataFrame): DataFrame containing product features
        feature_columns (list, optional): List of column names to use for similarity.
                                         If None, uses default features.
        similarity_matrix_path (str, optional): Path to save the similarity matrix

    Returns:
        pd.DataFrame: DataFrame with cosine similarity scores between products,
                     indexed and columns by Product_ID

    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Calculating content-based similarity matrix...")

    # Validate Product_ID column exists
    if 'Product_ID' not in df.columns:
        raise ValueError("Product_ID column is required for content-based filtering")

    # Default feature columns if not provided
    if feature_columns is None:
        feature_columns = [
            'Price of the product',
            'Rating of the product',
            'Customer review sentiment score (overall)'
        ]

    # Check which feature columns are available
    available_features = [col for col in feature_columns if col in df.columns]
    if not available_features:
        raise ValueError(f"None of the specified feature columns are available: {feature_columns}")

    logger.info(f"Using features: {available_features}")

    # Prepare feature matrix
    try:
        # Group by Product_ID and take mean of features (in case of duplicates)
        product_features = df.groupby('Product_ID')[available_features].mean()
        
        # Handle any remaining NaN values
        product_features = product_features.fillna(product_features.mean())
        
        logger.info(f"Feature matrix shape: {product_features.shape}")
        logger.info(f"Number of unique products: {len(product_features)}")

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(product_features.values)
        
        # Create DataFrame with Product_ID as index and columns
        cosine_sim_df = pd.DataFrame(
            similarity_matrix,
            index=product_features.index,
            columns=product_features.index
        )

        logger.info("Content similarity matrix calculated successfully")

        # Save similarity matrix if path provided
        if similarity_matrix_path:
            save_similarity_matrix(cosine_sim_df, similarity_matrix_path)

        return cosine_sim_df

    except Exception as e:
        logger.error(f"Error calculating content similarity: {str(e)}")
        raise


def get_similar_products(cosine_sim_df: pd.DataFrame, product_id: int, top_n: int = 5) -> pd.Series:
    """
    Get the top-N most similar products to the given product_id based on cosine similarity.

    Args:
        cosine_sim_df (pd.DataFrame): Cosine similarity matrix between products
        product_id (int): Product ID to find similar products for
        top_n (int): Number of similar products to return (default: 5)

    Returns:
        pd.Series: Series of similar product IDs with similarity scores,
                  sorted by similarity (descending)

    Raises:
        ValueError: If product_id is not in the similarity matrix
    """
    try:
        if product_id not in cosine_sim_df.index:
            raise ValueError(f"Product ID {product_id} not found in similarity matrix")

        # Get similarity scores for the product
        sim_scores = cosine_sim_df[product_id].sort_values(ascending=False)
        
        # Exclude the product itself (similarity = 1.0)
        sim_scores = sim_scores[sim_scores.index != product_id]
        
        # Return top N
        return sim_scores.head(top_n)

    except Exception as e:
        logger.warning(f"Error getting similar products for {product_id}: {str(e)}")
        return pd.Series(dtype=float)


def get_content_prediction(cosine_sim_df: pd.DataFrame, product_id: int, 
                          user_product_history: list = None) -> float:
    """
    Get content-based prediction score for a product.
    If user_product_history is provided, calculates weighted average based on user's history.
    Otherwise, returns average similarity score.

    Args:
        cosine_sim_df (pd.DataFrame): Cosine similarity matrix
        product_id (int): Product ID to predict for
        user_product_history (list, optional): List of product IDs the user has interacted with

    Returns:
        float: Content-based prediction score (0-1 range, normalized)
    """
    try:
        # Convert product_id to match index type if needed
        product_id_matched = product_id
        if product_id not in cosine_sim_df.index:
            # Try converting index to numeric and matching
            try:
                numeric_index = pd.to_numeric(cosine_sim_df.index)
                if product_id in numeric_index.values:
                    product_id_matched = numeric_index[numeric_index == product_id].index[0]
                else:
                    logger.warning(f"Product ID {product_id} not found in similarity matrix, returning default score")
                    return 0.5  # Default neutral score
            except (ValueError, TypeError, IndexError):
                logger.warning(f"Product ID {product_id} not found, returning default score")
                return 0.5  # Default neutral score

        if user_product_history and len(user_product_history) > 0:
            # Calculate weighted average based on user's product history
            similarities = []
            for hist_product_id in user_product_history:
                try:
                    hist_id_matched = hist_product_id
                    if hist_product_id not in cosine_sim_df.index:
                        try:
                            numeric_index = pd.to_numeric(cosine_sim_df.index)
                            if hist_product_id in numeric_index.values:
                                hist_id_matched = numeric_index[numeric_index == hist_product_id].index[0]
                            else:
                                continue
                        except (ValueError, TypeError, IndexError):
                            continue
                    sim = cosine_sim_df.loc[product_id_matched, hist_id_matched]
                    similarities.append(sim)
                except (KeyError, IndexError):
                    continue
            
            if similarities:
                # Average similarity to user's history
                prediction = np.mean(similarities)
            else:
                # Fallback to average similarity
                prediction = cosine_sim_df.loc[product_id_matched].mean()
        else:
            # Average similarity to all other products
            prediction = cosine_sim_df.loc[product_id_matched].mean()

        # Normalize to 0-1 range (cosine similarity is already 0-1, but ensure it)
        prediction = np.clip(prediction, 0.0, 1.0)
        
        return float(prediction)

    except Exception as e:
        logger.warning(f"Error calculating content prediction: {str(e)}")
        return 0.5  # Default neutral score


def save_similarity_matrix(cosine_sim_df: pd.DataFrame, file_path: str) -> None:
    """
    Save the similarity matrix to disk.

    Args:
        cosine_sim_df (pd.DataFrame): Similarity matrix to save
        file_path (str): Path to save the matrix
    """
    try:
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        cosine_sim_df.to_csv(file_path)
        logger.info(f"Similarity matrix saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving similarity matrix: {str(e)}")
        raise


def load_similarity_matrix(file_path: str) -> pd.DataFrame:
    """
    Load a similarity matrix from disk.

    Args:
        file_path (str): Path to the saved similarity matrix

    Returns:
        pd.DataFrame: Loaded similarity matrix

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Similarity matrix file not found: {file_path}")

    try:
        cosine_sim_df = pd.read_csv(file_path, index_col=0)
        # Convert index to numeric to ensure proper type matching
        try:
            cosine_sim_df.index = pd.to_numeric(cosine_sim_df.index)
            cosine_sim_df.columns = pd.to_numeric(cosine_sim_df.columns)
        except (ValueError, TypeError):
            # If conversion fails, keep as is
            pass
        logger.info(f"Similarity matrix loaded from {file_path}")
        return cosine_sim_df
    except Exception as e:
        logger.error(f"Error loading similarity matrix: {str(e)}")
        raise

