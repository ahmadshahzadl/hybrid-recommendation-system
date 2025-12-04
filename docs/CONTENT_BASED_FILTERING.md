# Content-Based Filtering Documentation

## Overview

Content-Based Filtering recommends items based on their features and attributes. It works on the principle: "If you liked this item, you'll like similar items."

**Simple analogy**: If you like action movies with Tom Cruise, the system will recommend other action movies with Tom Cruise or similar actors.

## What is Content-Based Filtering?

Content-Based Filtering uses **Cosine Similarity** to find items with similar features. It compares:
- Product prices
- Product ratings
- Sentiment scores
- Other product attributes

## File Location

`src/content_based_filtering.py`

## Key Concepts

### Feature Vector

Each product is represented as a vector of features:

```
Product A: [Price: 500, Rating: 4.5, Sentiment: 0.8]
Product B: [Price: 600, Rating: 4.3, Sentiment: 0.7]
```

### Cosine Similarity

Measures how similar two products are based on their feature vectors:
- **1.0**: Identical products
- **0.0**: Completely different products
- **-1.0**: Opposite products (rare)

## Key Functions

### 1. `calculate_content_similarity(df, feature_columns=None, similarity_matrix_path=None)`

**What it does**: Calculates how similar each product is to every other product.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with product features
- `feature_columns` (list, optional): Which columns to use for similarity (default: Price, Rating, Sentiment)
- `similarity_matrix_path` (str, optional): Where to save the similarity matrix

**Returns**: 
- `pd.DataFrame`: Similarity matrix (rows and columns are Product_IDs, values are similarity scores)

**Example**:
```python
from src.content_based_filtering import calculate_content_similarity
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Calculate similarity
similarity_matrix = calculate_content_similarity(
    df, 
    similarity_matrix_path='models/cb_similarity.csv'
)

print(f"Similarity matrix shape: {similarity_matrix.shape}")
```

**Default features used**:
- `Price of the product`
- `Rating of the product`
- `Customer review sentiment score (overall)`

### 2. `get_similar_products(cosine_sim_df, product_id, top_n=5)`

**What it does**: Finds the most similar products to a given product.

**Parameters**:
- `cosine_sim_df` (pd.DataFrame): Similarity matrix from `calculate_content_similarity()`
- `product_id` (int): Product to find similar items for
- `top_n` (int): Number of similar products to return (default: 5)

**Returns**:
- `pd.Series`: Similar products with similarity scores, sorted by similarity

**Example**:
```python
from src.content_based_filtering import calculate_content_similarity, get_similar_products
import pandas as pd

# Load and calculate similarity
df = pd.read_csv('data/processed_data.csv')
similarity_matrix = calculate_content_similarity(df)

# Get similar products
product_id = df['Product_ID'].iloc[0]
similar = get_similar_products(similarity_matrix, product_id, top_n=5)

print(f"Products similar to {product_id}:")
print(similar)
```

### 3. `get_content_prediction(cosine_sim_df, product_id, user_product_history=None)`

**What it does**: Calculates a content-based recommendation score for a product.

**Parameters**:
- `cosine_sim_df` (pd.DataFrame): Similarity matrix
- `product_id` (int): Product to score
- `user_product_history` (list, optional): Products the user has interacted with

**Returns**:
- `float`: Prediction score (0.0 to 1.0)

**Example**:
```python
from src.content_based_filtering import (
    calculate_content_similarity, 
    get_content_prediction
)
import pandas as pd

# Load and calculate similarity
df = pd.read_csv('data/processed_data.csv')
similarity_matrix = calculate_content_similarity(df)

# Get prediction for a product
product_id = 9346
score = get_content_prediction(similarity_matrix, product_id)
print(f"Content-based score: {score:.4f}")

# With user history
user_history = [1234, 5678, 9012]
score = get_content_prediction(similarity_matrix, product_id, user_history)
print(f"Score with user history: {score:.4f}")
```

### 4. `load_similarity_matrix(file_path)`

**What it does**: Loads a previously saved similarity matrix.

**Parameters**:
- `file_path` (str): Path to saved similarity matrix CSV

**Returns**: 
- `pd.DataFrame`: Similarity matrix

**Example**:
```python
from src.content_based_filtering import load_similarity_matrix

matrix = load_similarity_matrix('models/cb_similarity.csv')
print(f"Loaded matrix with {len(matrix)} products")
```

### 5. `save_similarity_matrix(cosine_sim_df, file_path)`

**What it does**: Saves a similarity matrix to disk.

**Parameters**:
- `cosine_sim_df` (pd.DataFrame): Similarity matrix to save
- `file_path` (str): Where to save

**Example**:
```python
from src.content_based_filtering import calculate_content_similarity, save_similarity_matrix
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
matrix = calculate_content_similarity(df)

save_similarity_matrix(matrix, 'models/my_similarity.csv')
```

## How to Run

### Method 1: Using the Main CLI (Recommended)

```bash
python run.py train --data data/processed_data.csv
```

This automatically calculates and saves the similarity matrix.

### Method 2: Python Script

```python
from src.content_based_filtering import (
    calculate_content_similarity,
    get_similar_products,
    get_content_prediction
)
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Calculate similarity matrix
print("Calculating similarity matrix...")
similarity_matrix = calculate_content_similarity(
    df,
    similarity_matrix_path='models/cb_similarity.csv'
)

# Find similar products
product_id = df['Product_ID'].iloc[0]
similar = get_similar_products(similarity_matrix, product_id, top_n=5)
print(f"\nProducts similar to {product_id}:")
print(similar)

# Get prediction score
score = get_content_prediction(similarity_matrix, product_id)
print(f"\nContent-based prediction score: {score:.4f}")
```

### Method 3: Using Jupyter Notebook

Open `notebooks/content_based_filtering.ipynb` and run the cells.

## Testing Content-Based Filtering

### Test 1: Basic Similarity Calculation

```python
from src.content_based_filtering import calculate_content_similarity
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed_data.csv')
similarity_matrix = calculate_content_similarity(df)

# Check shape
assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix should be square"
print("✓ Similarity matrix is square")

# Check diagonal (products should be identical to themselves)
diagonal = np.diag(similarity_matrix.values)
assert np.allclose(diagonal, 1.0), "Diagonal should be 1.0 (products identical to themselves)"
print("✓ Diagonal values are correct")

# Check symmetry (A similar to B = B similar to A)
assert np.allclose(similarity_matrix.values, similarity_matrix.values.T), "Matrix should be symmetric"
print("✓ Matrix is symmetric")
```

### Test 2: Get Similar Products

```python
from src.content_based_filtering import calculate_content_similarity, get_similar_products
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
similarity_matrix = calculate_content_similarity(df)

product_id = df['Product_ID'].iloc[0]
similar = get_similar_products(similarity_matrix, product_id, top_n=5)

assert len(similar) == 5, "Should return 5 products"
assert all(0 <= score <= 1 for score in similar.values), "Scores should be between 0-1"
assert similar.iloc[0] >= similar.iloc[-1], "Should be sorted descending"

print("✓ Get similar products works correctly")
print(f"Top 5 similar to product {product_id}:")
for prod_id, score in similar.items():
    print(f"  Product {prod_id}: {score:.4f}")
```

### Test 3: Content Prediction

```python
from src.content_based_filtering import (
    calculate_content_similarity,
    get_content_prediction
)
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
similarity_matrix = calculate_content_similarity(df)

product_id = df['Product_ID'].iloc[0]

# Without user history
score1 = get_content_prediction(similarity_matrix, product_id)
assert 0.0 <= score1 <= 1.0, "Score should be between 0-1"
print(f"✓ Prediction without history: {score1:.4f}")

# With user history
user_history = df[df['User_ID'] == df['User_ID'].iloc[0]]['Product_ID'].tolist()[:5]
score2 = get_content_prediction(similarity_matrix, product_id, user_history)
assert 0.0 <= score2 <= 1.0, "Score should be between 0-1"
print(f"✓ Prediction with history: {score2:.4f}")
```

### Test 4: Model Persistence

```python
from src.content_based_filtering import (
    calculate_content_similarity,
    save_similarity_matrix,
    load_similarity_matrix
)
import pandas as pd
import os

df = pd.read_csv('data/processed_data.csv')

# Calculate and save
matrix1 = calculate_content_similarity(df)
save_similarity_matrix(matrix1, 'models/test_similarity.csv')

# Load and compare
matrix2 = load_similarity_matrix('models/test_similarity.csv')

assert matrix1.shape == matrix2.shape, "Shapes should match"
assert matrix1.equals(matrix2), "Matrices should be identical"

print("✓ Model persistence works")

# Cleanup
os.remove('models/test_similarity.csv')
```

### Test 5: Handle Unknown Products

```python
from src.content_based_filtering import (
    calculate_content_similarity,
    get_content_prediction
)
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
similarity_matrix = calculate_content_similarity(df)

# Test with unknown product (should return default score)
score = get_content_prediction(similarity_matrix, product_id=99999)
assert score == 0.5, "Should return default score (0.5) for unknown products"
print("✓ Handles unknown products correctly")
```

## Expected Output

When calculating similarity, you should see:

```
2025-12-04 20:46:38,484 - INFO - Calculating content-based similarity matrix...
2025-12-04 20:46:38,485 - INFO - Using features: ['Price of the product', 'Rating of the product', 'Customer review sentiment score (overall)']
2025-12-04 20:46:38,491 - INFO - Feature matrix shape: (793, 3)
2025-12-04 20:46:38,492 - INFO - Number of unique products: 793
2025-12-04 20:46:38,497 - INFO - Content similarity matrix calculated successfully
```

## Understanding the Output

- **Feature matrix shape**: (number of products, number of features)
- **Number of unique products**: How many different products are in your data
- **Similarity scores**: Range from -1.0 to 1.0 (typically 0.0 to 1.0 for product features)

## How It Works (Simplified)

1. **Extract Features**: Get product features (price, rating, sentiment)
2. **Create Vectors**: Each product becomes a feature vector
3. **Calculate Similarity**: Compare all products using cosine similarity
4. **Build Matrix**: Create a matrix showing similarity between all product pairs
5. **Recommend**: Find products similar to ones the user likes

## Advantages

- ✅ Works for new users (doesn't need user history)
- ✅ Works for new items (if they have features)
- ✅ Explains recommendations (based on features)
- ✅ No cold start problem for items

## Limitations

- ❌ Limited by available features
- ❌ Can't discover unexpected connections
- ❌ May recommend only similar items (lack of diversity)

## Common Issues

### Issue: "Product_ID column is required"
**Cause**: Your data doesn't have a Product_ID column
**Solution**: Make sure your processed data has Product_ID (it's created during preprocessing)

### Issue: "None of the specified feature columns are available"
**Cause**: Required feature columns don't exist in your data
**Solution**: Check your data has Price, Rating, and Sentiment columns, or specify custom features

### Issue: "Product ID not found in similarity matrix"
**Cause**: The product doesn't exist in the similarity matrix
**Solution**: The function returns a default score (0.5) for unknown products

## Customizing Features

You can use different features for similarity:

```python
from src.content_based_filtering import calculate_content_similarity
import pandas as pd

df = pd.read_csv('data/processed_data.csv')

# Use custom features
custom_features = [
    'Price of the product',
    'Rating of the product',
    'Number of clicks on similar products'
]

similarity_matrix = calculate_content_similarity(
    df,
    feature_columns=custom_features
)
```

## Next Steps

1. Experiment with different feature combinations
2. Compare Content-Based with Collaborative Filtering
3. Combine both approaches (see `docs/HYBRID_MODEL.md`)
4. Evaluate the model (see `docs/EVALUATION.md`)

## Additional Resources

- See `docs/HYBRID_MODEL.md` for combining CB with Collaborative Filtering
- See `docs/EVALUATION.md` for evaluation metrics
- Check the Jupyter notebook: `notebooks/content_based_filtering.ipynb`

