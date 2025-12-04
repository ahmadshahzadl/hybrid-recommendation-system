# Evaluation Documentation

## Overview

The evaluation module provides metrics to measure how well the recommendation models are performing. It helps you understand if your models are making good predictions.

## File Location

`src/evaluation.py`

## Why Evaluate?

Evaluation helps you:
- **Compare models**: See which approach works best
- **Tune parameters**: Find optimal settings (like alpha)
- **Monitor performance**: Track if models are improving
- **Debug issues**: Identify problems in predictions

## Key Metrics

### RMSE (Root Mean Squared Error)
- **What it measures**: Average prediction error
- **Range**: 0 to infinity (lower is better)
- **Interpretation**: 
  - 0.0 = Perfect predictions
  - < 0.5 = Very good
  - 0.5-1.0 = Good
  - > 1.0 = Needs improvement

### MAE (Mean Absolute Error)
- **What it measures**: Average absolute difference between predicted and actual
- **Range**: 0 to infinity (lower is better)
- **Interpretation**: Similar to RMSE but less sensitive to outliers

### Similarity Metrics (Content-Based)
- **Average Similarity**: Mean similarity between products
- **Std Similarity**: How varied the similarities are
- **Range**: -1.0 to 1.0 (higher is better for most cases)

## Key Functions

### 1. `evaluate_cf_model(model_cf, df, test_size=0.2, random_state=42)`

**What it does**: Evaluates the Collaborative Filtering model.

**Parameters**:
- `model_cf`: Trained CF model
- `df` (pd.DataFrame): Dataset with User_ID, Product_ID, Rating columns
- `test_size` (float): Proportion for testing (default: 0.2)
- `random_state` (int): For reproducibility (default: 42)

**Returns**:
- `dict`: Dictionary with 'rmse' and 'mae' keys

**Example**:
```python
from src.collaborative_filtering import train_collaborative_filtering
from src.evaluation import evaluate_cf_model
import pandas as pd

# Load and train
df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

# Evaluate
results = evaluate_cf_model(model, df)
print(f"RMSE: {results['rmse']:.4f}")
print(f"MAE: {results['mae']:.4f}")
```

### 2. `evaluate_cb_model(df, sample_size=100)`

**What it does**: Evaluates the Content-Based Filtering model.

**Parameters**:
- `df` (pd.DataFrame): Dataset with product features
- `sample_size` (int): Number of products to sample (default: 100)

**Returns**:
- `dict`: Dictionary with similarity metrics

**Example**:
```python
from src.evaluation import evaluate_cb_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
results = evaluate_cb_model(df)

print(f"Average Similarity: {results['average_similarity']:.4f}")
print(f"Std Similarity: {results['std_similarity']:.4f}")
```

### 3. `evaluate_hybrid_model(df, alpha=0.5, sample_size=50)`

**What it does**: Evaluates the Hybrid model.

**Parameters**:
- `df` (pd.DataFrame): Dataset
- `alpha` (float): Weight for CF (default: 0.5)
- `sample_size` (int): Number of samples to test (default: 50)

**Returns**:
- `dict`: Dictionary with 'rmse', 'mae', 'mse', 'n_samples'

**Example**:
```python
from src.evaluation import evaluate_hybrid_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
results = evaluate_hybrid_model(df, alpha=0.7)

print(f"RMSE: {results['rmse']:.4f}")
print(f"MAE: {results['mae']:.4f}")
print(f"Samples tested: {results['n_samples']}")
```

### 4. `comprehensive_evaluation(df, alpha=0.5)`

**What it does**: Evaluates all three models (CF, CB, Hybrid) at once.

**Parameters**:
- `df` (pd.DataFrame): Dataset
- `alpha` (float): Weight for CF in hybrid model (default: 0.5)

**Returns**:
- `dict`: Dictionary with results for all models

**Example**:
```python
from src.evaluation import comprehensive_evaluation, print_evaluation_results
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
results = comprehensive_evaluation(df, alpha=0.7)

# Print formatted results
print_evaluation_results(results)
```

### 5. `print_evaluation_results(results)`

**What it does**: Prints evaluation results in a nice format.

**Parameters**:
- `results` (dict): Results dictionary from evaluation functions

**Example**:
```python
from src.evaluation import comprehensive_evaluation, print_evaluation_results
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
results = comprehensive_evaluation(df)
print_evaluation_results(results)
```

## How to Run

### Method 1: Using the Main CLI (Recommended)

```bash
python run.py evaluate --data data/processed_data.csv --alpha 0.7
```

### Method 2: Python Script

```python
from src.evaluation import comprehensive_evaluation, print_evaluation_results
import pandas as pd

# Load data
df = pd.read_csv('data/processed_data.csv')

# Evaluate all models
results = comprehensive_evaluation(df, alpha=0.7)

# Print results
print_evaluation_results(results)
```

### Method 3: Evaluate Individual Models

```python
from src.collaborative_filtering import train_collaborative_filtering
from src.evaluation import evaluate_cf_model, evaluate_cb_model, evaluate_hybrid_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')

# Evaluate CF
model = train_collaborative_filtering(df)
cf_results = evaluate_cf_model(model, df)
print("CF Results:", cf_results)

# Evaluate CB
cb_results = evaluate_cb_model(df)
print("CB Results:", cb_results)

# Evaluate Hybrid
hybrid_results = evaluate_hybrid_model(df, alpha=0.7)
print("Hybrid Results:", hybrid_results)
```

## Testing Evaluation

### Test 1: Basic Evaluation

```python
from src.collaborative_filtering import train_collaborative_filtering
from src.evaluation import evaluate_cf_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')
model = train_collaborative_filtering(df)

results = evaluate_cf_model(model, df)

assert 'rmse' in results, "Should have RMSE"
assert 'mae' in results, "Should have MAE"
assert results['rmse'] >= 0, "RMSE should be non-negative"
assert results['mae'] >= 0, "MAE should be non-negative"

print("✓ Evaluation works correctly")
print(f"  RMSE: {results['rmse']:.4f}")
print(f"  MAE: {results['mae']:.4f}")
```

### Test 2: Compare Different Alpha Values

```python
from src.evaluation import evaluate_hybrid_model
import pandas as pd

df = pd.read_csv('data/processed_data.csv')

print("Comparing different alpha values:")
print("-" * 50)

for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    results = evaluate_hybrid_model(df, alpha=alpha, sample_size=30)
    print(f"Alpha={alpha:.1f}: RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")
```

### Test 3: Comprehensive Evaluation

```python
from src.evaluation import comprehensive_evaluation, print_evaluation_results
import pandas as pd

df = pd.read_csv('data/processed_data.csv')

results = comprehensive_evaluation(df, alpha=0.7)

# Check all models were evaluated
assert 'collaborative_filtering' in results
assert 'content_based' in results
assert 'hybrid' in results

print("✓ Comprehensive evaluation works")
print_evaluation_results(results)
```

## Expected Output

When running evaluation:

```
============================================================
EVALUATION RESULTS
============================================================

COLLABORATIVE FILTERING:
----------------------------------------
  RMSE: 0.4319
  MAE: 0.2169

CONTENT BASED:
----------------------------------------
  AVERAGE_SIMILARITY: 0.1840
  STD_SIMILARITY: 0.1620
  MIN_SIMILARITY: -0.1202
  MAX_SIMILARITY: 0.4260

HYBRID:
----------------------------------------
  RMSE: 3.4080
  MAE: 3.1967
  MSE: 11.6143
  N_SAMPLES: 50

============================================================
```

## Understanding the Results

### Good Results
- **Low RMSE/MAE**: Model predictions are close to actual values
- **High Similarity**: Products are well-clustered by features
- **Consistent Performance**: Similar results across different samples

### Bad Results
- **High RMSE/MAE**: Predictions are far from actual values
- **Low Similarity**: Products aren't well-clustered
- **Inconsistent Performance**: Results vary a lot

## Interpreting Metrics

### RMSE vs MAE

- **RMSE**: Penalizes large errors more (sensitive to outliers)
- **MAE**: Treats all errors equally (more robust)

**Example**:
- If RMSE is much higher than MAE: You have some very bad predictions (outliers)
- If RMSE ≈ MAE: Errors are fairly consistent

### Similarity Metrics

- **High Average Similarity**: Products are generally similar (might lack diversity)
- **Low Average Similarity**: Products are diverse (good for variety)
- **High Std Similarity**: Some products are very similar, others very different
- **Low Std Similarity**: All products have similar similarity levels

## Improving Results

### If RMSE/MAE is High:

1. **More Data**: Collect more user-item interactions
2. **Better Features**: Add more relevant product features
3. **Tune Alpha**: Experiment with different alpha values
4. **Preprocessing**: Check data quality and preprocessing steps

### If Similarity is Low:

1. **Feature Selection**: Use more relevant features
2. **Feature Engineering**: Create better feature representations
3. **Normalization**: Ensure features are properly normalized

## Common Issues

### Issue: "Could not generate any predictions"
**Cause**: Model can't make predictions for the test data
**Solution**: Check that your data has valid User_ID, Product_ID, and Rating columns

### Issue: Very high RMSE
**Cause**: Model isn't trained well or data has issues
**Solution**: 
- Retrain models with more data
- Check data quality
- Try different alpha values

### Issue: Evaluation takes too long
**Cause**: Testing on too many samples
**Solution**: Reduce `sample_size` parameter

## Best Practices

1. **Always Evaluate**: Don't deploy models without evaluation
2. **Use Multiple Metrics**: RMSE and MAE together give better picture
3. **Compare Models**: Evaluate all approaches to find the best
4. **Test on Hold-Out Data**: Don't evaluate on training data
5. **Monitor Over Time**: Re-evaluate as new data comes in

## Next Steps

1. Run evaluation on your data
2. Compare different alpha values
3. Identify areas for improvement
4. Tune models based on results

## Additional Resources

- See `docs/COLLABORATIVE_FILTERING.md` for CF details
- See `docs/CONTENT_BASED_FILTERING.md` for CB details
- See `docs/HYBRID_MODEL.md` for hybrid model details
- See `docs/TESTING_GUIDE.md` for more testing examples

