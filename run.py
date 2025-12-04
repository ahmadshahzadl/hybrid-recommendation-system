"""
Main script to run the Hybrid Recommendation System pipeline.

This script provides a command-line interface to:
- Preprocess data
- Train models
- Generate recommendations
- Evaluate models
"""

import argparse
import sys
import logging
from pathlib import Path

from src.data_processing import load_data, preprocess_data, save_processed_data
from src.hybrid_model import hybrid_recommendation, get_top_recommendations, HybridRecommendationSystem
from src.evaluation import comprehensive_evaluation, print_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('recommendation_system.log')
    ]
)
logger = logging.getLogger(__name__)


def preprocess_pipeline(input_path: str, output_path: str) -> None:
    """Run the data preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline...")
    try:
        df = load_data(input_path)
        df_processed = preprocess_data(df)
        save_processed_data(df_processed, output_path)
        logger.info("Data preprocessing completed successfully!")
        print(f"\n✓ Processed data saved to: {output_path}")
        print(f"  Shape: {df_processed.shape}")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        sys.exit(1)


def train_models(data_path: str, alpha: float = 0.5, retrain: bool = False) -> None:
    """Train all models."""
    logger.info("Starting model training...")
    try:
        system = HybridRecommendationSystem(data_path=data_path, alpha=alpha)
        system.initialize(retrain=retrain)
        logger.info("All models trained successfully!")
        print("\n✓ Models trained and saved successfully!")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)


def generate_recommendation(user_id: int, product_id: int, alpha: float = 0.5,
                           data_path: str = 'data/processed_data.csv') -> None:
    """Generate a single recommendation."""
    logger.info(f"Generating recommendation for user {user_id}, product {product_id}...")
    try:
        score = hybrid_recommendation(user_id, product_id, alpha=alpha, data_path=data_path)
        print(f"\n✓ Hybrid Recommendation Score: {score:.4f}")
        print(f"  User ID: {user_id}")
        print(f"  Product ID: {product_id}")
        print(f"  Alpha (CF weight): {alpha}")
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        sys.exit(1)


def get_recommendations(user_id: int, top_n: int = 10, alpha: float = 0.5,
                       data_path: str = 'data/processed_data.csv') -> None:
    """Get top N recommendations for a user."""
    logger.info(f"Getting top {top_n} recommendations for user {user_id}...")
    try:
        recommendations = get_top_recommendations(user_id, top_n=top_n, alpha=alpha, data_path=data_path)
        print(f"\n✓ Top {top_n} Recommendations for User {user_id}:")
        print("-" * 50)
        for i, (product_id, score) in enumerate(recommendations, 1):
            print(f"  {i}. Product ID: {product_id}, Score: {score:.4f}")
    except Exception as e:
        logger.error(f"Getting recommendations failed: {e}")
        sys.exit(1)


def evaluate_models(data_path: str, alpha: float = 0.5) -> None:
    """Evaluate all models."""
    logger.info("Starting model evaluation...")
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        results = comprehensive_evaluation(df, alpha=alpha)
        print_evaluation_results(results)
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        sys.exit(1)


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Hybrid Recommendation System - Main Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data
  python run.py preprocess --input data/dataset.csv --output data/processed_data.csv

  # Train models
  python run.py train --data data/processed_data.csv

  # Generate single recommendation
  python run.py recommend --user_id 123 --product_id 456

  # Get top recommendations
  python run.py top --user_id 123 --top_n 10

  # Evaluate models
  python run.py evaluate --data data/processed_data.csv

  # Run full pipeline
  python run.py pipeline
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess the dataset')
    preprocess_parser.add_argument('--input', type=str, default='data/dataset.csv',
                                  help='Input dataset path (default: data/dataset.csv)')
    preprocess_parser.add_argument('--output', type=str, default='data/processed_data.csv',
                                  help='Output processed data path (default: data/processed_data.csv)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train all models')
    train_parser.add_argument('--data', type=str, default='data/processed_data.csv',
                             help='Processed data path (default: data/processed_data.csv)')
    train_parser.add_argument('--alpha', type=float, default=0.5,
                             help='Weight for Collaborative Filtering (0-1, default: 0.5)')
    train_parser.add_argument('--retrain', action='store_true',
                             help='Retrain models even if cached versions exist')

    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Generate a single recommendation')
    recommend_parser.add_argument('--user_id', type=int, required=True,
                                 help='User ID')
    recommend_parser.add_argument('--product_id', type=int, required=True,
                                 help='Product ID')
    recommend_parser.add_argument('--alpha', type=float, default=0.5,
                                 help='Weight for Collaborative Filtering (0-1, default: 0.5)')
    recommend_parser.add_argument('--data', type=str, default='data/processed_data.csv',
                                 help='Processed data path (default: data/processed_data.csv)')

    # Top recommendations command
    top_parser = subparsers.add_parser('top', help='Get top N recommendations for a user')
    top_parser.add_argument('--user_id', type=int, required=True,
                           help='User ID')
    top_parser.add_argument('--top_n', type=int, default=10,
                           help='Number of recommendations (default: 10)')
    top_parser.add_argument('--alpha', type=float, default=0.5,
                           help='Weight for Collaborative Filtering (0-1, default: 0.5)')
    top_parser.add_argument('--data', type=str, default='data/processed_data.csv',
                           help='Processed data path (default: data/processed_data.csv)')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate all models')
    evaluate_parser.add_argument('--data', type=str, default='data/processed_data.csv',
                                help='Processed data path (default: data/processed_data.csv)')
    evaluate_parser.add_argument('--alpha', type=float, default=0.5,
                                help='Weight for Collaborative Filtering (0-1, default: 0.5)')

    # Pipeline command (run everything)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the complete pipeline')
    pipeline_parser.add_argument('--input', type=str, default='data/dataset.csv',
                                help='Input dataset path (default: data/dataset.csv)')
    pipeline_parser.add_argument('--output', type=str, default='data/processed_data.csv',
                                help='Output processed data path (default: data/processed_data.csv)')
    pipeline_parser.add_argument('--alpha', type=float, default=0.5,
                                help='Weight for Collaborative Filtering (0-1, default: 0.5)')
    pipeline_parser.add_argument('--user_id', type=int, default=None,
                                help='User ID for example recommendation (optional)')
    pipeline_parser.add_argument('--product_id', type=int, default=None,
                                help='Product ID for example recommendation (optional)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'preprocess':
            preprocess_pipeline(args.input, args.output)

        elif args.command == 'train':
            train_models(args.data, alpha=args.alpha, retrain=args.retrain)

        elif args.command == 'recommend':
            generate_recommendation(args.user_id, args.product_id, alpha=args.alpha, data_path=args.data)

        elif args.command == 'top':
            get_recommendations(args.user_id, top_n=args.top_n, alpha=args.alpha, data_path=args.data)

        elif args.command == 'evaluate':
            evaluate_models(args.data, alpha=args.alpha)

        elif args.command == 'pipeline':
            print("=" * 60)
            print("Running Complete Pipeline")
            print("=" * 60)

            # Step 1: Preprocess
            print("\n[1/4] Preprocessing data...")
            preprocess_pipeline(args.input, args.output)

            # Step 2: Train models
            print("\n[2/4] Training models...")
            train_models(args.output, alpha=args.alpha)

            # Step 3: Example recommendation (if provided)
            if args.user_id and args.product_id:
                print("\n[3/4] Generating example recommendation...")
                generate_recommendation(args.user_id, args.product_id, alpha=args.alpha, data_path=args.output)
            else:
                print("\n[3/4] Skipping example recommendation (use --user_id and --product_id to enable)")

            # Step 4: Evaluate
            print("\n[4/4] Evaluating models...")
            evaluate_models(args.output, alpha=args.alpha)

            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

