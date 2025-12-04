# Collaborative Filtering Documentation

## Overview

Collaborative Filtering (CF) recommends items based on user behavior patterns. It works on the principle: "Users who liked similar items in the past will like similar items in the future."

**Simple analogy**: If you and your friend both like action movies, and your friend likes a new action movie you haven't seen, the system will recommend it to you.

## What is Collaborative Filtering?

Collaborative Filtering uses **Singular Value Decomposition (SVD)** to find hidden patterns in user-item interactions. It learns:
- What users have in common
- What items are similar
- How to predict ratings for new user-item pairs

## File Location

`src/collaborative_filtering.py`

## Key Concepts

### User-Item Matrix

A table showing how users rate items:

```
        Product1  Product2  Product3
User1      5         4         3
User2      4         5         2
User3      3         2         5
```

### SVD (Singular Value Decomposition)

SVD breaks down the user-item matrix into smaller matrices that capture:
- User preferences (latent factors)
- Item characteristics (latent factors)
- How these relate to each other

## Key Functions

### 1. `train_collaborative_filtering(df, model_path=None, test_size=0.2, random_state=42)`

**What it does**: Trains a Collaborative Filtering model using SVD.

**Parameters**:
- `df` (pd.DataFrame): DataFrame with columns: `User_ID`, `Product_ID`, `Rating of the product`
- `model_path` (str, optional): Where to save the trained model (e.g., `'models/cf_model.pkl'`)
- `test_size` (float): Proportion of data for testing (default: 0.2)
- `random_state` (int): For reproducibility (default: 42)

**Returns**: 
- Trained SVD model (can be used for predictions)

**Example**:
```python
from src.collaborative_filtering import train_collaborative_filtering
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed_data.csv')

# Train model
model = train_collaborative_filtering(df, model_path='models/cf_model.pkl')
print("Model trained!")
```

**What happens inside**:
1. Extracts User_ID, Product_ID, and Rating columns
2. Validates the data (removes invalid ratings)
3. Creates a user-item matrix
4. Applies SVD to find latent factors
5. Trains the model
6. Saves the model if path provided

### 2. `predict_rating(model_cf, user_id, product_id)`

**What it does**: Predicts how much a user would rate a product.

**Parameters**:
- `model_cf`: Trained CF model
- `user_id` (int): User identifier
- `product_id` (int): Product identifier

**Returns**:
- `float`: Predicted rating (1.0 to 5.0)

**Example**:
```python
from src.collaborative_filtering import train_collaborative_filtering, predict_rating
import pandas as pd

# Load and train
df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

# Predict
user_id = 5186
product_id = 9346
rating = predict_rating(model, user_id, product_id)
print(f"Predicted rating: {rating:.2f}")
```

### 3. `get_top_recommendations(model_cf, user_id, product_ids, top_n=10)`

**What it does**: Gets the top N product recommendations for a user.

**Parameters**:
- `model_cf`: Trained CF model
- `user_id` (int): User identifier
- `product_ids` (list): List of product IDs to consider
- `top_n` (int): Number of recommendations (default: 10)

**Returns**:
- `list`: List of tuples `(product_id, predicted_rating)` sorted by rating

**Example**:
```python
from src.collaborative_filtering import train_collaborative_filtering, get_top_recommendations
import pandas as pd

# Load and train
df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

# Get recommendations
user_id = 5186
all_products = df['Product_ID'].unique().tolist()
recommendations = get_top_recommendations(model, user_id, all_products, top_n=5)

print(f"Top 5 recommendations for user {user_id}:")
for product_id, rating in recommendations:
    print(f"  Product {product_id}: {rating:.2f}")
```

### 4. `load_model(model_path)`

**What it does**: Loads a previously saved model.

**Parameters**:
- `model_path` (str): Path to saved model file

**Returns**: 
- Trained model

**Example**:
```python
from src.collaborative_filtering import load_model, predict_rating

# Load saved model
model = load_model('models/cf_model.pkl')

# Use it
rating = predict_rating(model, user_id=5186, product_id=9346)
```

### 5. `save_model(model_cf, model_path)`

**What it does**: Saves a trained model to disk.

**Parameters**:
- `model_cf`: Trained model
- `model_path` (str): Where to save

**Example**:
```python
from src.collaborative_filtering import train_collaborative_filtering, save_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

# Save manually
save_model(model, 'models/my_cf_model.pkl')
```

## How to Run

### Method 1: Using the Main CLI (Recommended)

```bash
python run.py train --data data/processed_data.csv
```

This will:
- Train the CF model
- Save it to `models/cf_model.pkl`
- Also train the Content-Based model

### Method 2: Python Script

```python
from src.collaborative_filtering import train_collaborative_filtering, predict_rating
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Train model
print("Training Collaborative Filtering model...")
model = train_collaborative_filtering(df, model_path='models/cf_model.pkl')

# Test prediction
user_id = df['User_ID'].iloc[0]
product_id = df['Product_ID'].iloc[0]
rating = predict_rating(model, user_id, product_id)

print(f"Predicted rating for user {user_id}, product {product_id}: {rating:.2f}")
```

### Method 3: Using Jupyter Notebook

Open `notebooks/collaborative_filtering.ipynb` and run the cells.

## Testing Collaborative Filtering

### Test 1: Basic Training and Prediction

```python
from src.collaborative_filtering import train_collaborative_filtering, predict_rating
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Train
model = train_collaborative_filtering(df)
print("✓ Model trained successfully")

# Test prediction
user_id = df['User_ID'].iloc[0]
product_id = df['Product_ID'].iloc[0]
rating = predict_rating(model, user_id, product_id)

assert 1.0 <= rating <= 5.0, f"Rating should be between 1-5, got {rating}"
print(f"✓ Prediction works: {rating:.2f}")
```

### Test 2: Model Persistence

```python
from src.collaborative_filtering import (
    train_collaborative_filtering, 
    save_model, 
    load_model, 
    predict_rating
)
import pandas as pd
import os

df = pd.read_csv('data/processed_data.csv')

# Train and save
model1 = train_collaborative_filtering(df, model_path='models/test_cf_model.pkl')
rating1 = predict_rating(model1, 5186, 9346)

# Load and test
model2 = load_model('models/test_cf_model.pkl')
rating2 = predict_rating(model2, 5186, 9346)

assert abs(rating1 - rating2) < 0.01, "Loaded model should give same predictions"
print("✓ Model persistence works")

# Cleanup
os.remove('models/test_cf_model.pkl')
```

### Test 3: Top Recommendations

```python
from src.collaborative_filtering import train_collaborative_filtering, get_top_recommendations
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

user_id = df['User_ID'].iloc[0]
all_products = df['Product_ID'].unique().tolist()[:20]  # Test with first 20

recommendations = get_top_recommendations(model, user_id, all_products, top_n=5)

assert len(recommendations) == 5, "Should return 5 recommendations"
assert all(rating >= 1.0 and rating <= 5.0 for _, rating in recommendations), "Ratings should be valid"
assert recommendations[0][1] >= recommendations[-1][1], "Should be sorted descending"

print("✓ Top recommendations work correctly")
for i, (prod_id, rating) in enumerate(recommendations, 1):
    print(f"  {i}. Product {prod_id}: {rating:.2f}")
```

### Test 4: Handle Unknown Users/Products

```python
from src.collaborative_filtering import train_collaborative_filtering, predict_rating
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

# Test with unknown user/product (should return default rating)
rating = predict_rating(model, user_id=99999, product_id=99999)
assert 1.0 <= rating <= 5.0, "Should return valid rating even for unknown user/product"
print(f"✓ Handles unknown users/products: {rating:.2f}")
```

## Expected Output

When training, you should see:

```
2025-12-04 20:46:29,317 - INFO - Training Collaborative Filtering model...
2025-12-04 20:46:29,322 - INFO - Prepared 1474 ratings for training
2025-12-04 20:46:29,323 - INFO - Rating range: 1.00 - 1.13
2025-12-04 20:46:29,324 - INFO - Using scikit-learn TruncatedSVD as fallback
2025-12-04 20:46:29,446 - INFO - Training set size: 916 users, 793 items
2025-12-04 20:46:29,446 - INFO - Collaborative Filtering model trained successfully
```

## Understanding the Output

- **Prepared ratings**: Number of valid user-item-rating triplets
- **Rating range**: Min and max ratings in the data
- **Training set size**: Number of unique users and items
- **Model type**: Shows if using Surprise library or scikit-learn fallback

## How It Works (Simplified)

1. **Create Matrix**: Build a user-item rating matrix
2. **Decompose**: Use SVD to find hidden patterns
3. **Learn Factors**: Discover what makes users and items similar
4. **Predict**: Use learned factors to predict new ratings

## Advantages

- ✅ Works well when you have lots of user interactions
- ✅ Can discover unexpected connections
- ✅ Doesn't need item features (only ratings)

## Limitations

- ❌ Needs many users and ratings to work well
- ❌ Can't recommend to new users (cold start problem)
- ❌ Can't recommend new items (cold start problem)

## Common Issues

### Issue: "Missing required columns"
**Cause**: Your data doesn't have User_ID, Product_ID, or Rating columns
**Solution**: Make sure your processed data has these columns

### Issue: "Insufficient data for training"
**Cause**: Not enough user-item interactions (need at least 10)
**Solution**: Use a larger dataset or combine multiple data sources

### Issue: "surprise library not available"
**Cause**: The surprise library isn't installed (common on Windows with Python 3.12+)
**Solution**: This is fine! The system automatically uses scikit-learn's TruncatedSVD instead

## Next Steps

1. Test the model with different users and products
2. Compare CF predictions with actual ratings
3. Combine with Content-Based Filtering (see `docs/HYBRID_MODEL.md`)
4. Evaluate the model (see `docs/EVALUATION.md`)

## Additional Resources

- See `docs/HYBRID_MODEL.md` for combining CF with Content-Based
- See `docs/EVALUATION.md` for evaluation metrics
- Check the Jupyter notebook: `notebooks/collaborative_filtering.ipynb`

