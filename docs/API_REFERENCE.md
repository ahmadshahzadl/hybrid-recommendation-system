# API Reference

Complete reference for all functions and classes in the Hybrid Recommendation System.

## Table of Contents

- [Data Processing](#data-processing)
- [Collaborative Filtering](#collaborative-filtering)
- [Content-Based Filtering](#content-based-filtering)
- [Hybrid Model](#hybrid-model)
- [Evaluation](#evaluation)

---

## Data Processing

### `load_data(file_path: str) -> pd.DataFrame`

Loads a CSV dataset file.

**Parameters**:
- `file_path` (str): Path to CSV file

**Returns**: `pd.DataFrame` - Loaded dataset

**Raises**:
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file is empty or invalid

**Example**:
```python
from src.data_processing import load_data
df = load_data('data/dataset.csv')
```

---

### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

Preprocesses raw data: handles missing values, encodes categories, normalizes features.

**Parameters**:
- `df` (pd.DataFrame): Raw dataset

**Returns**: `pd.DataFrame` - Preprocessed dataset

**Example**:
```python
from src.data_processing import preprocess_data
df_processed = preprocess_data(df)
```

---

### `save_processed_data(df: pd.DataFrame, output_path: str) -> None`

Saves processed dataset to CSV.

**Parameters**:
- `df` (pd.DataFrame): Processed dataset
- `output_path` (str): Output file path

**Example**:
```python
from src.data_processing import save_processed_data
save_processed_data(df, 'data/processed_data.csv')
```

---

## Collaborative Filtering

### `train_collaborative_filtering(df: pd.DataFrame, model_path: str = None, test_size: float = 0.2, random_state: int = 42)`

Trains a Collaborative Filtering model using SVD.

**Parameters**:
- `df` (pd.DataFrame): Data with User_ID, Product_ID, Rating columns
- `model_path` (str, optional): Path to save model
- `test_size` (float): Test set proportion (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Returns**: Trained SVD model

**Example**:
```python
from src.collaborative_filtering import train_collaborative_filtering
model = train_collaborative_filtering(df, model_path='models/cf_model.pkl')
```

---

### `predict_rating(model_cf, user_id: int, product_id: int) -> float`

Predicts rating for a user-product pair.

**Parameters**:
- `model_cf`: Trained CF model
- `user_id` (int): User identifier
- `product_id` (int): Product identifier

**Returns**: `float` - Predicted rating (1.0-5.0)

**Example**:
```python
from src.collaborative_filtering import predict_rating
rating = predict_rating(model, user_id=5186, product_id=9346)
```

---

### `get_top_recommendations(model_cf, user_id: int, product_ids: list, top_n: int = 10) -> list`

Gets top N product recommendations for a user.

**Parameters**:
- `model_cf`: Trained CF model
- `user_id` (int): User identifier
- `product_ids` (list): List of product IDs to consider
- `top_n` (int): Number of recommendations (default: 10)

**Returns**: `list` - List of (product_id, rating) tuples

**Example**:
```python
from src.collaborative_filtering import get_top_recommendations
recommendations = get_top_recommendations(model, user_id=5186, 
                                         product_ids=[1,2,3,4,5], top_n=5)
```

---

### `load_model(model_path: str)`

Loads a saved CF model.

**Parameters**:
- `model_path` (str): Path to model file

**Returns**: Trained model

**Example**:
```python
from src.collaborative_filtering import load_model
model = load_model('models/cf_model.pkl')
```

---

### `save_model(model_cf, model_path: str) -> None`

Saves a trained CF model.

**Parameters**:
- `model_cf`: Trained model
- `model_path` (str): Save path

**Example**:
```python
from src.collaborative_filtering import save_model
save_model(model, 'models/cf_model.pkl')
```

---

## Content-Based Filtering

### `calculate_content_similarity(df: pd.DataFrame, feature_columns: list = None, similarity_matrix_path: str = None) -> pd.DataFrame`

Calculates cosine similarity matrix between products.

**Parameters**:
- `df` (pd.DataFrame): Data with product features
- `feature_columns` (list, optional): Features to use (default: Price, Rating, Sentiment)
- `similarity_matrix_path` (str, optional): Path to save matrix

**Returns**: `pd.DataFrame` - Similarity matrix

**Example**:
```python
from src.content_based_filtering import calculate_content_similarity
matrix = calculate_content_similarity(df, 
                                     similarity_matrix_path='models/cb_similarity.csv')
```

---

### `get_similar_products(cosine_sim_df: pd.DataFrame, product_id: int, top_n: int = 5) -> pd.Series`

Gets top N similar products.

**Parameters**:
- `cosine_sim_df` (pd.DataFrame): Similarity matrix
- `product_id` (int): Product identifier
- `top_n` (int): Number of similar products (default: 5)

**Returns**: `pd.Series` - Similar products with scores

**Example**:
```python
from src.content_based_filtering import get_similar_products
similar = get_similar_products(matrix, product_id=9346, top_n=5)
```

---

### `get_content_prediction(cosine_sim_df: pd.DataFrame, product_id: int, user_product_history: list = None) -> float`

Gets content-based prediction score.

**Parameters**:
- `cosine_sim_df` (pd.DataFrame): Similarity matrix
- `product_id` (int): Product identifier
- `user_product_history` (list, optional): User's product history

**Returns**: `float` - Prediction score (0.0-1.0)

**Example**:
```python
from src.content_based_filtering import get_content_prediction
score = get_content_prediction(matrix, product_id=9346, 
                              user_product_history=[1234, 5678])
```

---

### `load_similarity_matrix(file_path: str) -> pd.DataFrame`

Loads a saved similarity matrix.

**Parameters**:
- `file_path` (str): Path to matrix file

**Returns**: `pd.DataFrame` - Similarity matrix

**Example**:
```python
from src.content_based_filtering import load_similarity_matrix
matrix = load_similarity_matrix('models/cb_similarity.csv')
```

---

### `save_similarity_matrix(cosine_sim_df: pd.DataFrame, file_path: str) -> None`

Saves similarity matrix to disk.

**Parameters**:
- `cosine_sim_df` (pd.DataFrame): Similarity matrix
- `file_path` (str): Save path

**Example**:
```python
from src.content_based_filtering import save_similarity_matrix
save_similarity_matrix(matrix, 'models/cb_similarity.csv')
```

---

## Hybrid Model

### `hybrid_recommendation(user_id: int, product_id: int, alpha: float = 0.5, data_path: str = 'data/processed_data.csv', retrain: bool = False) -> float`

Gets hybrid recommendation score.

**Parameters**:
- `user_id` (int): User identifier
- `product_id` (int): Product identifier
- `alpha` (float): CF weight, 0-1 (default: 0.5)
- `data_path` (str): Path to processed data (default: 'data/processed_data.csv')
- `retrain` (bool): Force retrain (default: False)

**Returns**: `float` - Hybrid score (0.0-1.0)

**Example**:
```python
from src.hybrid_model import hybrid_recommendation
score = hybrid_recommendation(user_id=5186, product_id=9346, alpha=0.7)
```

---

### `get_top_recommendations(user_id: int, top_n: int = 10, alpha: float = 0.5, data_path: str = 'data/processed_data.csv') -> list`

Gets top N hybrid recommendations.

**Parameters**:
- `user_id` (int): User identifier
- `top_n` (int): Number of recommendations (default: 10)
- `alpha` (float): CF weight, 0-1 (default: 0.5)
- `data_path` (str): Path to processed data (default: 'data/processed_data.csv')

**Returns**: `list` - List of (product_id, score) tuples

**Example**:
```python
from src.hybrid_model import get_top_recommendations
recommendations = get_top_recommendations(user_id=5186, top_n=10, alpha=0.7)
```

---

### `HybridRecommendationSystem` Class

Class-based interface for hybrid recommendations.

#### `__init__(data_path: str = 'data/processed_data.csv', cf_model_path: str = 'models/cf_model.pkl', cb_similarity_path: str = 'models/cb_similarity.csv', alpha: float = 0.5)`

Initializes the hybrid system.

**Parameters**:
- `data_path` (str): Path to processed data
- `cf_model_path` (str): Path to CF model
- `cb_similarity_path` (str): Path to CB similarity matrix
- `alpha` (float): CF weight, 0-1 (default: 0.5)

**Example**:
```python
from src.hybrid_model import HybridRecommendationSystem
system = HybridRecommendationSystem(alpha=0.7)
```

#### `initialize(retrain: bool = False) -> None`

Initializes all components (loads or trains models).

**Parameters**:
- `retrain` (bool): Force retrain (default: False)

**Example**:
```python
system.initialize()
```

#### `load_data() -> pd.DataFrame`

Loads the processed dataset.

**Returns**: `pd.DataFrame` - Dataset

**Example**:
```python
df = system.load_data()
```

#### `predict(user_id: int, product_id: int, user_product_history: list = None) -> float`

Makes a prediction.

**Parameters**:
- `user_id` (int): User identifier
- `product_id` (int): Product identifier
- `user_product_history` (list, optional): User's product history

**Returns**: `float` - Prediction score (0.0-1.0)

**Example**:
```python
score = system.predict(user_id=5186, product_id=9346)
```

---

## Evaluation

### `evaluate_cf_model(model_cf, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict`

Evaluates Collaborative Filtering model.

**Parameters**:
- `model_cf`: Trained CF model
- `df` (pd.DataFrame): Dataset
- `test_size` (float): Test proportion (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Returns**: `dict` - Dictionary with 'rmse' and 'mae' keys

**Example**:
```python
from src.evaluation import evaluate_cf_model
results = evaluate_cf_model(model, df)
print(f"RMSE: {results['rmse']}")
```

---

### `evaluate_cb_model(df: pd.DataFrame, sample_size: int = 100) -> dict`

Evaluates Content-Based model.

**Parameters**:
- `df` (pd.DataFrame): Dataset
- `sample_size` (int): Number of samples (default: 100)

**Returns**: `dict` - Dictionary with similarity metrics

**Example**:
```python
from src.evaluation import evaluate_cb_model
results = evaluate_cb_model(df)
print(f"Average Similarity: {results['average_similarity']}")
```

---

### `evaluate_hybrid_model(df: pd.DataFrame, alpha: float = 0.5, sample_size: int = 50) -> dict`

Evaluates Hybrid model.

**Parameters**:
- `df` (pd.DataFrame): Dataset
- `alpha` (float): CF weight (default: 0.5)
- `sample_size` (int): Number of samples (default: 50)

**Returns**: `dict` - Dictionary with 'rmse', 'mae', 'mse', 'n_samples'

**Example**:
```python
from src.evaluation import evaluate_hybrid_model
results = evaluate_hybrid_model(df, alpha=0.7)
print(f"RMSE: {results['rmse']}")
```

---

### `comprehensive_evaluation(df: pd.DataFrame, alpha: float = 0.5) -> dict`

Evaluates all models.

**Parameters**:
- `df` (pd.DataFrame): Dataset
- `alpha` (float): CF weight for hybrid (default: 0.5)

**Returns**: `dict` - Dictionary with results for all models

**Example**:
```python
from src.evaluation import comprehensive_evaluation
results = comprehensive_evaluation(df, alpha=0.7)
```

---

### `print_evaluation_results(results: dict) -> None`

Prints evaluation results in formatted way.

**Parameters**:
- `results` (dict): Results dictionary

**Example**:
```python
from src.evaluation import comprehensive_evaluation, print_evaluation_results
results = comprehensive_evaluation(df)
print_evaluation_results(results)
```

---

## Type Hints

All functions use type hints for better IDE support and documentation. Key types:

- `pd.DataFrame`: pandas DataFrame
- `int`: Integer
- `float`: Floating point number
- `str`: String
- `list`: List
- `dict`: Dictionary
- `bool`: Boolean

---

## Error Handling

All functions include error handling and will raise appropriate exceptions:

- `FileNotFoundError`: File doesn't exist
- `ValueError`: Invalid input data or parameters
- `KeyError`: Missing required columns
- `AttributeError`: Invalid model or object

---

## Next Steps

- See individual module documentation for detailed examples
- Check `docs/GETTING_STARTED.md` for usage examples
- See `docs/TESTING_GUIDE.md` for testing examples

