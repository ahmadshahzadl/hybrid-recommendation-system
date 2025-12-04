# Hybrid Model Documentation

## Overview

The Hybrid Model combines **Collaborative Filtering (CF)** and **Content-Based Filtering (CB)** to provide better recommendations than either method alone. It uses a weighted average of both approaches.

**Simple analogy**: Instead of asking only your friends (CF) or only looking at product features (CB), you ask both and combine their opinions!

## Why Hybrid?

- **Collaborative Filtering** is great when you have lots of user interactions
- **Content-Based Filtering** works well for new users and items
- **Hybrid** gets the best of both worlds!

## File Location

`src/hybrid_model.py`

## Key Concepts

### Alpha Parameter

Controls how much weight to give each method:
- **alpha = 0.0**: Pure Content-Based (100% CB, 0% CF)
- **alpha = 0.5**: Equal weighting (50% CB, 50% CF) - **Default**
- **alpha = 1.0**: Pure Collaborative Filtering (0% CB, 100% CF)

### Hybrid Score Calculation

```
hybrid_score = alpha × CF_score + (1 - alpha) × CB_score
```

Both scores are normalized to 0-1 range before combination.

## Key Functions

### 1. `hybrid_recommendation(user_id, product_id, alpha=0.5, data_path='data/processed_data.csv', retrain=False)`

**What it does**: Gets a hybrid recommendation score for a user-product pair.

**Parameters**:
- `user_id` (int): User identifier
- `product_id` (int): Product identifier
- `alpha` (float): Weight for CF (0-1, default: 0.5)
- `data_path` (str): Path to processed data
- `retrain` (bool): Whether to retrain models (default: False)

**Returns**:
- `float`: Hybrid recommendation score (0.0 to 1.0)

**Example**:
```python
from src.hybrid_model import hybrid_recommendation

# Get recommendation with default alpha (0.5)
score = hybrid_recommendation(user_id=5186, product_id=9346)
print(f"Hybrid score: {score:.4f}")

# Get recommendation with custom alpha (70% CF, 30% CB)
score = hybrid_recommendation(user_id=5186, product_id=9346, alpha=0.7)
print(f"Hybrid score (alpha=0.7): {score:.4f}")
```

### 2. `get_top_recommendations(user_id, top_n=10, alpha=0.5, data_path='data/processed_data.csv')`

**What it does**: Gets the top N product recommendations for a user.

**Parameters**:
- `user_id` (int): User identifier
- `top_n` (int): Number of recommendations (default: 10)
- `alpha` (float): Weight for CF (0-1, default: 0.5)
- `data_path` (str): Path to processed data

**Returns**:
- `list`: List of tuples `(product_id, hybrid_score)` sorted by score

**Example**:
```python
from src.hybrid_model import get_top_recommendations

# Get top 10 recommendations
recommendations = get_top_recommendations(user_id=5186, top_n=10, alpha=0.7)

print(f"Top 10 recommendations for user 5186:")
for i, (product_id, score) in enumerate(recommendations, 1):
    print(f"{i}. Product {product_id}: {score:.4f}")
```

### 3. `HybridRecommendationSystem` Class

**What it does**: A class-based interface for the hybrid system with more control.

**Example**:
```python
from src.hybrid_model import HybridRecommendationSystem

# Initialize system
system = HybridRecommendationSystem(
    data_path='data/processed_data.csv',
    alpha=0.7  # 70% CF, 30% CB
)

# Initialize all components (loads or trains models)
system.initialize()

# Make predictions
score = system.predict(user_id=5186, product_id=9346)
print(f"Prediction: {score:.4f}")

# Get user's product history for better predictions
df = system.load_data()
user_history = df[df['User_ID'] == 5186]['Product_ID'].tolist()
score = system.predict(user_id=5186, product_id=9346, user_product_history=user_history)
```

## How to Run

### Method 1: Using the Main CLI (Recommended)

#### Get Single Recommendation

```bash
python run.py recommend --user_id 5186 --product_id 9346 --alpha 0.7
```

#### Get Top Recommendations

```bash
python run.py top --user_id 5186 --top_n 10 --alpha 0.7
```

#### Train Models First (if needed)

```bash
python run.py train --data data/processed_data.csv --alpha 0.7
```

### Method 2: Python Script

```python
from src.hybrid_model import hybrid_recommendation, get_top_recommendations

# Single recommendation
score = hybrid_recommendation(user_id=5186, product_id=9346, alpha=0.7)
print(f"Recommendation score: {score:.4f}")

# Top recommendations
recommendations = get_top_recommendations(user_id=5186, top_n=10, alpha=0.7)
for product_id, score in recommendations:
    print(f"Product {product_id}: {score:.4f}")
```

### Method 3: Using Jupyter Notebook

Open `notebooks/hybrid_model.ipynb` and run the cells.

## Testing the Hybrid Model

### Test 1: Basic Hybrid Recommendation

```python
from src.hybrid_model import hybrid_recommendation
import pandas as pd

# Load data to get valid IDs
df = pd.read_csv('data/processed_data.csv')
user_id = df['User_ID'].iloc[0]
product_id = df['Product_ID'].iloc[0]

# Test with different alpha values
for alpha in [0.0, 0.5, 1.0]:
    score = hybrid_recommendation(user_id, product_id, alpha=alpha)
    assert 0.0 <= score <= 1.0, f"Score should be 0-1, got {score}"
    print(f"Alpha={alpha:.1f}: Score={score:.4f}")

print("✓ Hybrid recommendation works with different alpha values")
```

### Test 2: Top Recommendations

```python
from src.hybrid_model import get_top_recommendations
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
user_id = df['User_ID'].iloc[0]

recommendations = get_top_recommendations(user_id, top_n=5, alpha=0.7)

assert len(recommendations) == 5, "Should return 5 recommendations"
assert all(0.0 <= score <= 1.0 for _, score in recommendations), "Scores should be 0-1"
assert recommendations[0][1] >= recommendations[-1][1], "Should be sorted descending"

print("✓ Top recommendations work correctly")
for i, (prod_id, score) in enumerate(recommendations, 1):
    print(f"  {i}. Product {prod_id}: {score:.4f}")
```

### Test 3: Compare Alpha Values

```python
from src.hybrid_model import hybrid_recommendation
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
user_id = df['User_ID'].iloc[0]
product_id = df['Product_ID'].iloc[0]

print("Comparing different alpha values:")
print("-" * 50)
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    score = hybrid_recommendation(user_id, product_id, alpha=alpha)
    cf_weight = alpha * 100
    cb_weight = (1 - alpha) * 100
    print(f"Alpha={alpha:.1f} (CF:{cf_weight:.0f}%, CB:{cb_weight:.0f}%): {score:.4f}")
```

### Test 4: HybridRecommendationSystem Class

```python
from src.hybrid_model import HybridRecommendationSystem
import pandas as pd

# Initialize
system = HybridRecommendationSystem(alpha=0.7)
system.initialize()

# Test prediction
df = system.load_data()
user_id = df['User_ID'].iloc[0]
product_id = df['Product_ID'].iloc[0]

score = system.predict(user_id, product_id)
assert 0.0 <= score <= 1.0, "Score should be 0-1"
print(f"✓ HybridRecommendationSystem works: {score:.4f}")

# Test with user history
user_history = df[df['User_ID'] == user_id]['Product_ID'].tolist()[:5]
score_with_history = system.predict(user_id, product_id, user_product_history=user_history)
print(f"✓ With user history: {score_with_history:.4f}")
```

### Test 5: Model Caching

```python
from src.hybrid_model import hybrid_recommendation
import time

user_id = 5186
product_id = 9346

# First call (trains/loads models)
start = time.time()
score1 = hybrid_recommendation(user_id, product_id, alpha=0.7)
time1 = time.time() - start

# Second call (uses cached models)
start = time.time()
score2 = hybrid_recommendation(user_id, product_id, alpha=0.7)
time2 = time.time() - start

assert abs(score1 - score2) < 0.01, "Scores should be identical"
assert time2 < time1, "Second call should be faster (cached)"
print(f"✓ Model caching works (first: {time1:.2f}s, second: {time2:.2f}s)")
```

## Expected Output

When using the hybrid model:

```
2025-12-04 20:47:29,079 - INFO - Initializing Hybrid Recommendation System...
2025-12-04 20:47:29,091 - INFO - Loaded data from data/processed_data.csv, shape: (1474, 18)
2025-12-04 20:47:29,093 - INFO - Model loaded from models/cf_model.pkl
2025-12-04 20:47:29,093 - INFO - Loaded existing CF model
2025-12-04 20:47:29,324 - INFO - Similarity matrix loaded from models/cb_similarity.csv
2025-12-04 20:47:29,324 - INFO - Loaded existing CB similarity matrix
2025-12-04 20:47:29,324 - INFO - Hybrid Recommendation System initialized successfully

✓ Hybrid Recommendation Score: 0.3749
  User ID: 5186
  Product ID: 9346
  Alpha (CF weight): 0.5
```

## Understanding the Output

- **Hybrid Recommendation Score**: Combined score from both CF and CB (0.0 to 1.0)
- **Alpha**: Weight given to Collaborative Filtering
- **Initialization**: Shows which models were loaded or trained

## How It Works (Simplified)

1. **Get CF Score**: Predict using Collaborative Filtering (normalized to 0-1)
2. **Get CB Score**: Predict using Content-Based Filtering (0-1 range)
3. **Combine**: Weighted average: `alpha × CF + (1-alpha) × CB`
4. **Return**: Final hybrid score

## Choosing the Right Alpha

### Use Higher Alpha (0.7-1.0) when:
- You have lots of user interactions
- User behavior is more important than item features
- You want recommendations based on "people like you"

### Use Lower Alpha (0.0-0.3) when:
- You have new users (cold start)
- Item features are very important
- You want recommendations based on item similarity

### Use Balanced Alpha (0.4-0.6) when:
- You want the best of both worlds
- You're not sure which is better
- You want diverse recommendations

## Advantages

- ✅ Combines strengths of both CF and CB
- ✅ Handles cold start problems better
- ✅ More accurate than individual methods
- ✅ Flexible (adjustable alpha parameter)

## Limitations

- ❌ More complex than individual methods
- ❌ Requires both user interactions and item features
- ❌ Alpha parameter needs tuning

## Common Issues

### Issue: "System not initialized"
**Cause**: You need to call `initialize()` or use the function interface
**Solution**: Use `hybrid_recommendation()` function or call `system.initialize()`

### Issue: Models not found
**Cause**: Models haven't been trained yet
**Solution**: Run `python run.py train` first

### Issue: Low scores for all recommendations
**Cause**: Alpha might not be tuned for your data
**Solution**: Try different alpha values (0.0, 0.3, 0.5, 0.7, 1.0)

## Tuning Alpha

Experiment with different alpha values to find what works best:

```python
from src.hybrid_model import hybrid_recommendation
from src.evaluation import evaluate_hybrid_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')

# Test different alpha values
best_alpha = 0.5
best_rmse = float('inf')

for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    results = evaluate_hybrid_model(df, alpha=alpha)
    rmse = results['rmse']
    print(f"Alpha={alpha:.1f}: RMSE={rmse:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha} (RMSE: {best_rmse:.4f})")
```

## Next Steps

1. Experiment with different alpha values
2. Evaluate the hybrid model (see `docs/EVALUATION.md`)
3. Compare hybrid vs individual methods
4. Tune alpha for your specific use case

## Additional Resources

- See `docs/COLLABORATIVE_FILTERING.md` for CF details
- See `docs/CONTENT_BASED_FILTERING.md` for CB details
- See `docs/EVALUATION.md` for evaluation metrics
- Check the Jupyter notebook: `notebooks/hybrid_model.ipynb`

