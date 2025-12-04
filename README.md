# Hybrid Recommendation System

This project implements a **Hybrid Recommendation System** using **Collaborative Filtering (CF)** and **Content-Based Filtering (CB)**. It combines user-item interactions (e.g., clicks, purchases, ratings) with product features (e.g., price, rating, sentiment) to provide personalized recommendations.

## Features

- **Collaborative Filtering**: Uses **Singular Value Decomposition (SVD)** to recommend products based on user-item interactions. Falls back to scikit-learn's TruncatedSVD if the `surprise` library is not available.
- **Content-Based Filtering**: Uses **Cosine Similarity** to recommend products based on item features such as price, rating, and sentiment score.
- **Hybrid Model**: Combines both CF and CB models using a weighted average to improve recommendation accuracy.
- **Model Persistence**: Trained models and similarity matrices are saved for faster subsequent runs.
- **Comprehensive Evaluation**: Includes RMSE, MAE, and similarity metrics for model evaluation.
- **Command-Line Interface**: Easy-to-use CLI for all operations.
- **Jupyter Notebooks**: Interactive notebooks for experimentation and analysis.

## File Structure

```
hybrid-recommendation-system/
│
├── data/                          # Data folder containing the CSV dataset
│   ├── dataset.csv                # The raw dataset in CSV format
│   └── processed_data.csv         # Processed dataset after cleaning and encoding
│
├── models/                        # Trained models and similarity matrices (auto-generated)
│   ├── cf_model.pkl              # Trained Collaborative Filtering model
│   └── cb_similarity.csv          # Content-Based similarity matrix
│
├── notebooks/                     # Jupyter notebooks for experiments, analysis, and model testing
│   ├── data_preprocessing.ipynb   # Data cleaning and preprocessing
│   ├── collaborative_filtering.ipynb  # Implementing collaborative filtering (SVD)
│   ├── content_based_filtering.ipynb  # Implementing content-based filtering (Cosine Similarity)
│   └── hybrid_model.ipynb         # Combining CF and CB models into a hybrid model
│
├── src/                           # Source code for model and utilities
│   ├── __init__.py                # Initialize the package
│   ├── data_processing.py         # Functions for data cleaning and preprocessing
│   ├── collaborative_filtering.py # Collaborative filtering model (SVD)
│   ├── content_based_filtering.py # Content-based filtering model (Cosine Similarity)
│   ├── hybrid_model.py            # Hybrid recommendation system combining CF and CB
│   └── evaluation.py              # Evaluation metrics and testing
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview and instructions
└── run.py                        # Main script to run the entire pipeline
```

## Dependencies

- pandas - Data manipulation and analysis
- scikit-learn - Machine learning algorithms and utilities
- scipy - Scientific computing
- numpy - Numerical computing
- surprise - Recommendation system algorithms (optional, falls back to scikit-learn if unavailable)
- jupyter - Interactive notebook environment

**Note**: On Windows with Python 3.12+, the `surprise` library may not be available. The system automatically falls back to scikit-learn's TruncatedSVD, which provides similar functionality.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your dataset CSV file in the `data/` directory as `dataset.csv`. The dataset should contain columns for user interactions and product features.

### 3. Preprocess Data

Run the data preprocessing script to clean and encode the dataset:

```bash
python src/data_processing.py
```

This will:
- Handle missing values
- Encode categorical variables (Gender, Holiday, Season, Geographical locations)
- Normalize numerical features
- Create User_ID and Product_ID columns if missing
- Save processed data to `data/processed_data.csv`

### 4. Run the Pipeline

You can either run individual commands or use the full pipeline:

```bash
# Run complete pipeline (preprocess, train, evaluate)
python run.py pipeline
```

## Usage

### Command Line Interface

The `run.py` script provides a comprehensive CLI with multiple commands:

#### Preprocess Data

```bash
python run.py preprocess --input data/dataset.csv --output data/processed_data.csv
```

#### Train Models

```bash
# Train all models with default alpha (0.5)
python run.py train --data data/processed_data.csv

# Train with custom alpha (0.7 = 70% CF, 30% CB)
python run.py train --data data/processed_data.csv --alpha 0.7

# Force retrain even if models exist
python run.py train --data data/processed_data.csv --retrain
```

#### Generate Single Recommendation

```bash
python run.py recommend --user_id 5186 --product_id 9346 --alpha 0.7
```

#### Get Top N Recommendations

```bash
# Get top 10 recommendations for a user
python run.py top --user_id 5186 --top_n 10 --alpha 0.7
```

#### Evaluate Models

```bash
python run.py evaluate --data data/processed_data.csv --alpha 0.7
```

#### Run Complete Pipeline

```bash
# Run everything: preprocess, train, evaluate
python run.py pipeline

# With example recommendation
python run.py pipeline --user_id 5186 --product_id 9346 --alpha 0.7
```

### Python API

#### Basic Usage

```python
from src.hybrid_model import hybrid_recommendation, get_top_recommendations

# Get a single recommendation score
user_id = 5186
product_id = 9346
score = hybrid_recommendation(user_id, product_id, alpha=0.7)
print(f"Hybrid Recommendation Score: {score:.4f}")

# Get top 10 recommendations for a user
recommendations = get_top_recommendations(user_id, top_n=10, alpha=0.7)
for product_id, score in recommendations:
    print(f"Product {product_id}: {score:.4f}")
```

#### Advanced Usage with HybridRecommendationSystem Class

```python
from src.hybrid_model import HybridRecommendationSystem

# Initialize the system
system = HybridRecommendationSystem(
    data_path='data/processed_data.csv',
    alpha=0.7  # 70% weight for Collaborative Filtering
)

# Initialize all components (loads or trains models)
system.initialize()

# Make predictions
score = system.predict(user_id=5186, product_id=9346)
print(f"Prediction score: {score:.4f}")
```

#### Using Individual Components

```python
from src.collaborative_filtering import train_collaborative_filtering, predict_rating
from src.content_based_filtering import calculate_content_similarity, get_similar_products
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Train Collaborative Filtering model
cf_model = train_collaborative_filtering(df)

# Train Content-Based model
cb_similarity = calculate_content_similarity(df)

# Get predictions
cf_prediction = predict_rating(cf_model, user_id=5186, product_id=9346)
similar_products = get_similar_products(cb_similarity, product_id=9346, top_n=5)
```

### Jupyter Notebooks

Interactive notebooks are available in the `notebooks/` directory:

- `data_preprocessing.ipynb` - Explore data preprocessing steps
- `collaborative_filtering.ipynb` - Experiment with CF model
- `content_based_filtering.ipynb` - Experiment with CB model
- `hybrid_model.ipynb` - Combine models and evaluate

To use the notebooks:

```bash
jupyter notebook notebooks/
```

## Model Details

### Collaborative Filtering (SVD)

- **Algorithm**: Singular Value Decomposition (SVD) or TruncatedSVD (fallback)
- **Input**: User-ID, Product-ID, Rating matrix
- **Process**: 
  - Factorizes the user-item interaction matrix
  - Learns latent factors from user-item ratings
  - Predicts ratings for unseen user-item pairs
- **Output**: Predicted rating (1-5 scale, normalized to 0-1 for hybrid model)
- **Model Persistence**: Saved as `models/cf_model.pkl`

### Content-Based Filtering

- **Algorithm**: Cosine Similarity
- **Features Used**: 
  - Price of the product
  - Rating of the product
  - Customer review sentiment score (overall)
- **Process**:
  - Calculates cosine similarity between products based on feature vectors
  - Groups products by Product_ID and averages features
  - Recommends similar products based on feature similarity
- **Output**: Similarity score (0-1 range)
- **Model Persistence**: Saved as `models/cb_similarity.csv`

### Hybrid Model

- **Combination Method**: Weighted average of CF and CB predictions
- **Alpha Parameter**: 
  - Controls the weight of Collaborative Filtering (0-1 range)
  - `alpha = 0.0`: Pure Content-Based Filtering
  - `alpha = 0.5`: Equal weighting (default)
  - `alpha = 1.0`: Pure Collaborative Filtering
- **Normalization**: Both CF and CB scores are normalized to 0-1 range before combination
- **Output**: Hybrid recommendation score (0-1 range)

## Evaluation

The system includes comprehensive evaluation metrics:

### Collaborative Filtering Metrics
- **RMSE** (Root Mean Squared Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual ratings

### Content-Based Filtering Metrics
- **Average Similarity**: Mean cosine similarity between products
- **Std Similarity**: Standard deviation of similarity scores
- **Min/Max Similarity**: Range of similarity values

### Hybrid Model Metrics
- **RMSE**: Prediction error for hybrid scores
- **MAE**: Mean absolute error for hybrid predictions
- **MSE**: Mean squared error

### Running Evaluation

```bash
python run.py evaluate --data data/processed_data.csv --alpha 0.7
```

This will output evaluation results for all three models (CF, CB, and Hybrid).

## Data Format

The input dataset (`data/dataset.csv`) should contain the following columns:

- **User Features**:
  - `Gender` - User gender (categorical)
  - `Median purchasing price (in rupees)` - User's typical purchase price
  - `Number of clicks on similar products` - User engagement metric
  - `Number of similar products purchased so far` - Purchase history

- **Product Features**:
  - `Brand of the product` - Product brand (categorical)
  - `Price of the product` - Product price
  - `Rating of the product` - Product rating (1-5 scale)
  - `Customer review sentiment score (overall)` - Sentiment score (-1 to 1)

- **Context Features**:
  - `Holiday` - Whether it's a holiday (Yes/No)
  - `Season` - Season (spring/summer/winter/monsoon)
  - `Geographical locations` - Location (plains/mountains/coastal)

- **Target**:
  - `Probability for the product to be recommended to the person` - Target variable (optional)

**Note**: If `User_ID` and `Product_ID` columns are missing, they will be automatically generated based on feature combinations.

## Troubleshooting

### Issue: "surprise library not available"

**Solution**: This is expected on Windows with Python 3.12+. The system automatically uses scikit-learn's TruncatedSVD as a fallback, which provides similar functionality.

### Issue: "Product ID not found"

**Solution**: Ensure the Product_ID exists in your processed dataset. The system will return a default score (0.5) for unknown products.

### Issue: "Insufficient data for training"

**Solution**: Ensure your dataset has at least 10 user-item interactions with ratings.

### Issue: Models not loading

**Solution**: Delete the `models/` directory and retrain:
```bash
rm -rf models/
python run.py train --data data/processed_data.csv --retrain
```

## Project Structure Details

- **`src/data_processing.py`**: Handles data loading, cleaning, encoding, and normalization
- **`src/collaborative_filtering.py`**: Implements SVD-based collaborative filtering with model persistence
- **`src/content_based_filtering.py`**: Implements cosine similarity-based content filtering
- **`src/hybrid_model.py`**: Combines CF and CB models with caching and optimization
- **`src/evaluation.py`**: Provides comprehensive evaluation metrics for all models
- **`run.py`**: Command-line interface for all operations

## Best Practices

1. **Alpha Tuning**: Experiment with different alpha values (0.0 to 1.0) to find the optimal balance between CF and CB for your dataset.

2. **Model Caching**: Trained models are automatically cached. Use `--retrain` flag only when you need to retrain with new data.

3. **Data Quality**: Ensure your dataset has sufficient user-item interactions (at least 10-20 per user/product) for best results.

4. **Evaluation**: Always evaluate models on a held-out test set in production scenarios.

## Performance Notes

- **First Run**: May take longer as models need to be trained
- **Subsequent Runs**: Much faster as models are loaded from cache
- **Memory Usage**: Similarity matrix can be large for datasets with many products
- **Scalability**: For very large datasets, consider using batch processing or incremental learning

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Quick Links
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Complete beginner's guide with step-by-step instructions
- **[Documentation Index](docs/INDEX.md)** - Navigate all documentation files
- **[CLI Reference](docs/CLI_REFERENCE.md)** - Complete command-line interface guide
- **[API Reference](docs/API_REFERENCE.md)** - Full Python API documentation
- **[Testing Guide](docs/TESTING_GUIDE.md)** - How to test each component
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Module Documentation
- [Data Format](docs/DATA_FORMAT.md) - Required data format and specifications
- [Data Processing](docs/DATA_PROCESSING.md) - Data preprocessing module
- [Collaborative Filtering](docs/COLLABORATIVE_FILTERING.md) - CF model details
- [Content-Based Filtering](docs/CONTENT_BASED_FILTERING.md) - CB model details
- [Hybrid Model](docs/HYBRID_MODEL.md) - Hybrid system documentation
- [Evaluation](docs/EVALUATION.md) - Evaluation metrics and methods

**For beginners**: Start with the [Getting Started Guide](docs/GETTING_STARTED.md)

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- Error handling is comprehensive

## Acknowledgments

- Uses scikit-learn for machine learning algorithms
- Uses pandas for data manipulation
- Uses surprise library (when available) for recommendation algorithms

