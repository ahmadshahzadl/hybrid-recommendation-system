"""
Evaluation Module

This module provides evaluation metrics for the recommendation system models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, Tuple

# Try to import surprise, use fallback if not available
try:
    from surprise import accuracy, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("surprise library not available. Using scikit-learn metrics for evaluation.")

from .collaborative_filtering import train_collaborative_filtering, predict_rating
from .content_based_filtering import calculate_content_similarity, get_content_prediction
from .hybrid_model import hybrid_recommendation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_cf_model(model_cf, df: pd.DataFrame, test_size: float = 0.2, 
                     random_state: int = 42) -> Dict[str, float]:
    """
    Evaluate the Collaborative Filtering model using RMSE and MAE.

    Args:
        model_cf: Trained Collaborative Filtering model
        df (pd.DataFrame): DataFrame with User_ID, Product_ID, and Rating columns
        test_size (float): Proportion of data to use for testing
        random_state (int): Random state for reproducibility

    Returns:
        dict: Dictionary containing evaluation metrics (RMSE, MAE)
    """
    logger.info("Evaluating Collaborative Filtering model...")

    try:
        # Prepare data
        required_cols = ['User_ID', 'Product_ID', 'Rating of the product']
        rating_data = df[required_cols].copy()
        rating_data.columns = ['User_ID', 'Product_ID', 'Rating']
        rating_data = rating_data.dropna(subset=['Rating'])
        rating_data['Rating'] = rating_data['Rating'].clip(lower=1, upper=5)

        # Make predictions and calculate metrics
        if SURPRISE_AVAILABLE:
            # Use Surprise evaluation
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(rating_data, reader)
            trainset = data.build_full_trainset()
            testset = trainset.build_testset()
            predictions = model_cf.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
        else:
            # Use scikit-learn metrics
            actuals = []
            preds = []
            for _, row in rating_data.iterrows():
                try:
                    pred = predict_rating(model_cf, row['User_ID'], row['Product_ID'])
                    actuals.append(row['Rating'])
                    preds.append(pred)
                except:
                    continue
            
            if len(actuals) > 0:
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                mae = mean_absolute_error(actuals, preds)
            else:
                raise ValueError("Could not generate any predictions for evaluation")

        logger.info(f"CF Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return {
            'rmse': rmse,
            'mae': mae
        }

    except Exception as e:
        logger.error(f"Error evaluating CF model: {str(e)}")
        raise


def evaluate_cb_model(df: pd.DataFrame, sample_size: int = 100) -> Dict[str, float]:
    """
    Evaluate the Content-Based Filtering model using similarity metrics.

    Args:
        df (pd.DataFrame): DataFrame with product features
        sample_size (int): Number of products to sample for evaluation

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating Content-Based Filtering model...")

    try:
        # Calculate similarity matrix
        cosine_sim_df = calculate_content_similarity(df)

        # Sample products for evaluation
        product_ids = cosine_sim_df.index.tolist()
        if len(product_ids) > sample_size:
            np.random.seed(42)
            sample_products = np.random.choice(product_ids, sample_size, replace=False)
        else:
            sample_products = product_ids

        # Calculate average similarity scores
        similarities = []
        for product_id in sample_products:
            # Get similarity to other products (excluding itself)
            sim_scores = cosine_sim_df[product_id]
            sim_scores = sim_scores[sim_scores.index != product_id]
            avg_sim = sim_scores.mean()
            similarities.append(avg_sim)

        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)

        logger.info(f"CB Model - Average Similarity: {avg_similarity:.4f}, Std: {std_similarity:.4f}")

        return {
            'average_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }

    except Exception as e:
        logger.error(f"Error evaluating CB model: {str(e)}")
        raise


def evaluate_hybrid_model(df: pd.DataFrame, alpha: float = 0.5, 
                         sample_size: int = 50) -> Dict[str, float]:
    """
    Evaluate the Hybrid model by comparing predictions with actual ratings.

    Args:
        df (pd.DataFrame): DataFrame with User_ID, Product_ID, and Rating columns
        alpha (float): Weight for Collaborative Filtering
        sample_size (int): Number of user-product pairs to sample for evaluation

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    logger.info("Evaluating Hybrid model...")

    try:
        # Sample user-product pairs
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        predictions = []
        actuals = []

        for _, row in sample_df.iterrows():
            user_id = row['User_ID']
            product_id = row['Product_ID']
            actual_rating = row['Rating of the product']

            try:
                # Get hybrid prediction (0-1 range)
                hybrid_score = hybrid_recommendation(user_id, product_id, alpha=alpha)
                
                # Convert to 1-5 rating scale for comparison
                predicted_rating = 1 + (hybrid_score * 4)  # Scale from 0-1 to 1-5
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except Exception as e:
                logger.warning(f"Skipping evaluation for user {user_id}, product {product_id}: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("No valid predictions generated for evaluation")

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)

        logger.info(f"Hybrid Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'n_samples': len(predictions)
        }

    except Exception as e:
        logger.error(f"Error evaluating Hybrid model: {str(e)}")
        raise


def comprehensive_evaluation(df: pd.DataFrame, alpha: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Perform comprehensive evaluation of all models.

    Args:
        df (pd.DataFrame): Preprocessed dataset
        alpha (float): Weight for Collaborative Filtering in hybrid model

    Returns:
        dict: Dictionary containing evaluation results for all models
    """
    logger.info("Starting comprehensive evaluation...")

    results = {}

    try:
        # Evaluate Collaborative Filtering
        logger.info("=" * 50)
        logger.info("Evaluating Collaborative Filtering Model")
        logger.info("=" * 50)
        model_cf = train_collaborative_filtering(df)
        results['collaborative_filtering'] = evaluate_cf_model(model_cf, df)

    except Exception as e:
        logger.error(f"CF evaluation failed: {e}")
        results['collaborative_filtering'] = {'error': str(e)}

    try:
        # Evaluate Content-Based Filtering
        logger.info("=" * 50)
        logger.info("Evaluating Content-Based Filtering Model")
        logger.info("=" * 50)
        results['content_based'] = evaluate_cb_model(df)

    except Exception as e:
        logger.error(f"CB evaluation failed: {e}")
        results['content_based'] = {'error': str(e)}

    try:
        # Evaluate Hybrid Model
        logger.info("=" * 50)
        logger.info("Evaluating Hybrid Model")
        logger.info("=" * 50)
        results['hybrid'] = evaluate_hybrid_model(df, alpha=alpha)

    except Exception as e:
        logger.error(f"Hybrid evaluation failed: {e}")
        results['hybrid'] = {'error': str(e)}

    logger.info("=" * 50)
    logger.info("Evaluation Complete")
    logger.info("=" * 50)

    return results


def print_evaluation_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print evaluation results in a formatted way.

    Args:
        results (dict): Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric_name.upper()}: {value:.4f}")
                else:
                    print(f"  {metric_name.upper()}: {value}")

    print("\n" + "=" * 60)

