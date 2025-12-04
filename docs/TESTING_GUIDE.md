# Testing Guide

## Overview

This guide provides comprehensive testing examples for all components of the Hybrid Recommendation System. Use these tests to verify everything is working correctly.

## Prerequisites

Before testing, make sure:
1. Dependencies are installed: `pip install -r requirements.txt`
2. Data is preprocessed: `python src/data_processing.py`
3. You're in the project root directory

## Quick Test Suite

Run all basic tests at once:

```python
# test_all.py
import sys
sys.path.append('.')

from test_data_processing import *
from test_collaborative_filtering import *
from test_content_based_filtering import *
from test_hybrid_model import *

print("All tests passed! ✓")
```

## Testing Data Processing

### Test 1: Basic Functionality

```python
# test_data_processing.py
from src.data_processing import load_data, preprocess_data, save_processed_data
import pandas as pd
import os

def test_load_data():
    """Test loading data"""
    df = load_data('data/dataset.csv')
    assert not df.empty, "Data should not be empty"
    assert len(df) > 0, "Should have rows"
    print("✓ Data loading works")

def test_preprocess_data():
    """Test preprocessing"""
    df = load_data('data/dataset.csv')
    df_processed = preprocess_data(df)
    
    assert 'User_ID' in df_processed.columns, "Should have User_ID"
    assert 'Product_ID' in df_processed.columns, "Should have Product_ID"
    assert df_processed.isnull().sum().sum() == 0, "Should have no missing values"
    print("✓ Data preprocessing works")

def test_save_processed_data():
    """Test saving processed data"""
    df = load_data('data/dataset.csv')
    df_processed = preprocess_data(df)
    
    test_path = 'data/test_processed.csv'
    save_processed_data(df_processed, test_path)
    
    assert os.path.exists(test_path), "File should be created"
    
    # Cleanup
    os.remove(test_path)
    print("✓ Data saving works")

if __name__ == "__main__":
    test_load_data()
    test_preprocess_data()
    test_save_processed_data()
    print("\nAll data processing tests passed!")
```

**Run**: `python test_data_processing.py`

---

## Testing Collaborative Filtering

### Test 2: CF Model Training and Prediction

```python
# test_collaborative_filtering.py
from src.collaborative_filtering import (
    train_collaborative_filtering,
    predict_rating,
    get_top_recommendations,
    load_model,
    save_model
)
import pandas as pd
import os

def test_train_model():
    """Test model training"""
    df = pd.read_csv('data/processed_data.csv')
    model = train_collaborative_filtering(df)
    assert model is not None, "Model should be created"
    print("✓ Model training works")

def test_predict_rating():
    """Test rating prediction"""
    df = pd.read_csv('data/processed_data.csv')
    model = train_collaborative_filtering(df)
    
    user_id = df['User_ID'].iloc[0]
    product_id = df['Product_ID'].iloc[0]
    rating = predict_rating(model, user_id, product_id)
    
    assert 1.0 <= rating <= 5.0, f"Rating should be 1-5, got {rating}"
    print("✓ Rating prediction works")

def test_model_persistence():
    """Test saving and loading models"""
    df = pd.read_csv('data/processed_data.csv')
    model1 = train_collaborative_filtering(df)
    
    test_path = 'models/test_cf_model.pkl'
    save_model(model1, test_path)
    
    model2 = load_model(test_path)
    
    # Test same prediction
    user_id = df['User_ID'].iloc[0]
    product_id = df['Product_ID'].iloc[0]
    
    rating1 = predict_rating(model1, user_id, product_id)
    rating2 = predict_rating(model2, user_id, product_id)
    
    assert abs(rating1 - rating2) < 0.01, "Loaded model should give same predictions"
    
    # Cleanup
    os.remove(test_path)
    print("✓ Model persistence works")

def test_top_recommendations():
    """Test top recommendations"""
    df = pd.read_csv('data/processed_data.csv')
    model = train_collaborative_filtering(df)
    
    user_id = df['User_ID'].iloc[0]
    all_products = df['Product_ID'].unique().tolist()[:20]
    
    recommendations = get_top_recommendations(model, user_id, all_products, top_n=5)
    
    assert len(recommendations) == 5, "Should return 5 recommendations"
    assert all(1.0 <= r[1] <= 5.0 for r in recommendations), "Ratings should be valid"
    print("✓ Top recommendations work")

if __name__ == "__main__":
    test_train_model()
    test_predict_rating()
    test_model_persistence()
    test_top_recommendations()
    print("\nAll collaborative filtering tests passed!")
```

**Run**: `python test_collaborative_filtering.py`

---

## Testing Content-Based Filtering

### Test 3: CB Similarity and Recommendations

```python
# test_content_based_filtering.py
from src.content_based_filtering import (
    calculate_content_similarity,
    get_similar_products,
    get_content_prediction,
    save_similarity_matrix,
    load_similarity_matrix
)
import pandas as pd
import numpy as np
import os

def test_calculate_similarity():
    """Test similarity matrix calculation"""
    df = pd.read_csv('data/processed_data.csv')
    similarity_matrix = calculate_content_similarity(df)
    
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Should be square"
    assert np.allclose(np.diag(similarity_matrix.values), 1.0), "Diagonal should be 1.0"
    print("✓ Similarity calculation works")

def test_get_similar_products():
    """Test finding similar products"""
    df = pd.read_csv('data/processed_data.csv')
    similarity_matrix = calculate_content_similarity(df)
    
    product_id = df['Product_ID'].iloc[0]
    similar = get_similar_products(similarity_matrix, product_id, top_n=5)
    
    assert len(similar) == 5, "Should return 5 products"
    assert all(0 <= score <= 1 for score in similar.values), "Scores should be 0-1"
    print("✓ Get similar products works")

def test_content_prediction():
    """Test content-based prediction"""
    df = pd.read_csv('data/processed_data.csv')
    similarity_matrix = calculate_content_similarity(df)
    
    product_id = df['Product_ID'].iloc[0]
    score = get_content_prediction(similarity_matrix, product_id)
    
    assert 0.0 <= score <= 1.0, "Score should be 0-1"
    print("✓ Content prediction works")

def test_similarity_persistence():
    """Test saving and loading similarity matrix"""
    df = pd.read_csv('data/processed_data.csv')
    matrix1 = calculate_content_similarity(df)
    
    test_path = 'models/test_similarity.csv'
    save_similarity_matrix(matrix1, test_path)
    
    matrix2 = load_similarity_matrix(test_path)
    
    assert matrix1.shape == matrix2.shape, "Shapes should match"
    assert np.allclose(matrix1.values, matrix2.values), "Values should match"
    
    # Cleanup
    os.remove(test_path)
    print("✓ Similarity persistence works")

if __name__ == "__main__":
    test_calculate_similarity()
    test_get_similar_products()
    test_content_prediction()
    test_similarity_persistence()
    print("\nAll content-based filtering tests passed!")
```

**Run**: `python test_content_based_filtering.py`

---

## Testing Hybrid Model

### Test 4: Hybrid Recommendations

```python
# test_hybrid_model.py
from src.hybrid_model import (
    hybrid_recommendation,
    get_top_recommendations,
    HybridRecommendationSystem
)
import pandas as pd

def test_hybrid_recommendation():
    """Test hybrid recommendation"""
    df = pd.read_csv('data/processed_data.csv')
    user_id = df['User_ID'].iloc[0]
    product_id = df['Product_ID'].iloc[0]
    
    score = hybrid_recommendation(user_id, product_id, alpha=0.7)
    
    assert 0.0 <= score <= 1.0, "Score should be 0-1"
    print("✓ Hybrid recommendation works")

def test_different_alpha():
    """Test with different alpha values"""
    df = pd.read_csv('data/processed_data.csv')
    user_id = df['User_ID'].iloc[0]
    product_id = df['Product_ID'].iloc[0]
    
    scores = []
    for alpha in [0.0, 0.5, 1.0]:
        score = hybrid_recommendation(user_id, product_id, alpha=alpha)
        assert 0.0 <= score <= 1.0, f"Score should be 0-1 for alpha={alpha}"
        scores.append(score)
    
    print("✓ Different alpha values work")
    print(f"  Alpha 0.0: {scores[0]:.4f}")
    print(f"  Alpha 0.5: {scores[1]:.4f}")
    print(f"  Alpha 1.0: {scores[2]:.4f}")

def test_top_recommendations():
    """Test top hybrid recommendations"""
    df = pd.read_csv('data/processed_data.csv')
    user_id = df['User_ID'].iloc[0]
    
    recommendations = get_top_recommendations(user_id, top_n=5, alpha=0.7)
    
    assert len(recommendations) == 5, "Should return 5 recommendations"
    assert all(0.0 <= score <= 1.0 for _, score in recommendations), "Scores should be 0-1"
    print("✓ Top hybrid recommendations work")

def test_hybrid_system_class():
    """Test HybridRecommendationSystem class"""
    system = HybridRecommendationSystem(alpha=0.7)
    system.initialize()
    
    df = system.load_data()
    user_id = df['User_ID'].iloc[0]
    product_id = df['Product_ID'].iloc[0]
    
    score = system.predict(user_id, product_id)
    assert 0.0 <= score <= 1.0, "Score should be 0-1"
    print("✓ HybridRecommendationSystem class works")

if __name__ == "__main__":
    test_hybrid_recommendation()
    test_different_alpha()
    test_top_recommendations()
    test_hybrid_system_class()
    print("\nAll hybrid model tests passed!")
```

**Run**: `python test_hybrid_model.py`

---

## Testing Evaluation

### Test 5: Evaluation Metrics

```python
# test_evaluation.py
from src.evaluation import (
    evaluate_cf_model,
    evaluate_cb_model,
    evaluate_hybrid_model,
    comprehensive_evaluation
)
from src.collaborative_filtering import train_collaborative_filtering
import pandas as pd

def test_cf_evaluation():
    """Test CF model evaluation"""
    df = pd.read_csv('data/processed_data.csv')
    model = train_collaborative_filtering(df)
    
    results = evaluate_cf_model(model, df)
    
    assert 'rmse' in results, "Should have RMSE"
    assert 'mae' in results, "Should have MAE"
    assert results['rmse'] >= 0, "RMSE should be non-negative"
    print("✓ CF evaluation works")

def test_cb_evaluation():
    """Test CB model evaluation"""
    df = pd.read_csv('data/processed_data.csv')
    results = evaluate_cb_model(df)
    
    assert 'average_similarity' in results, "Should have average similarity"
    assert 'std_similarity' in results, "Should have std similarity"
    print("✓ CB evaluation works")

def test_hybrid_evaluation():
    """Test hybrid model evaluation"""
    df = pd.read_csv('data/processed_data.csv')
    results = evaluate_hybrid_model(df, alpha=0.7)
    
    assert 'rmse' in results, "Should have RMSE"
    assert 'mae' in results, "Should have MAE"
    assert results['n_samples'] > 0, "Should have samples"
    print("✓ Hybrid evaluation works")

def test_comprehensive_evaluation():
    """Test comprehensive evaluation"""
    df = pd.read_csv('data/processed_data.csv')
    results = comprehensive_evaluation(df, alpha=0.7)
    
    assert 'collaborative_filtering' in results
    assert 'content_based' in results
    assert 'hybrid' in results
    print("✓ Comprehensive evaluation works")

if __name__ == "__main__":
    test_cf_evaluation()
    test_cb_evaluation()
    test_hybrid_evaluation()
    test_comprehensive_evaluation()
    print("\nAll evaluation tests passed!")
```

**Run**: `python test_evaluation.py`

---

## Integration Tests

### Test 6: End-to-End Pipeline

```python
# test_integration.py
from src.data_processing import load_data, preprocess_data, save_processed_data
from src.hybrid_model import hybrid_recommendation, get_top_recommendations
from src.evaluation import comprehensive_evaluation
import pandas as pd
import os

def test_full_pipeline():
    """Test complete pipeline from data to recommendations"""
    
    # 1. Load and preprocess
    print("Step 1: Loading and preprocessing data...")
    df = load_data('data/dataset.csv')
    df_processed = preprocess_data(df)
    save_processed_data(df_processed, 'data/test_processed.csv')
    print("✓ Data preprocessed")
    
    # 2. Get recommendation
    print("Step 2: Getting recommendations...")
    user_id = df_processed['User_ID'].iloc[0]
    product_id = df_processed['Product_ID'].iloc[0]
    score = hybrid_recommendation(user_id, product_id, 
                                  data_path='data/test_processed.csv')
    assert 0.0 <= score <= 1.0, "Score should be valid"
    print(f"✓ Recommendation: {score:.4f}")
    
    # 3. Get top recommendations
    print("Step 3: Getting top recommendations...")
    recommendations = get_top_recommendations(user_id, top_n=5,
                                             data_path='data/test_processed.csv')
    assert len(recommendations) == 5, "Should return 5 recommendations"
    print("✓ Top recommendations retrieved")
    
    # 4. Evaluate
    print("Step 4: Evaluating models...")
    results = comprehensive_evaluation(df_processed, alpha=0.7)
    assert 'hybrid' in results, "Should have hybrid results"
    print("✓ Evaluation complete")
    
    # Cleanup
    if os.path.exists('data/test_processed.csv'):
        os.remove('data/test_processed.csv')
    
    print("\n✓ Full pipeline test passed!")

if __name__ == "__main__":
    test_full_pipeline()
```

**Run**: `python test_integration.py`

---

## Running All Tests

Create a master test file:

```python
# run_all_tests.py
import sys
import subprocess

tests = [
    'test_data_processing.py',
    'test_collaborative_filtering.py',
    'test_content_based_filtering.py',
    'test_hybrid_model.py',
    'test_evaluation.py',
    'test_integration.py'
]

print("Running all tests...\n")
print("=" * 60)

for test in tests:
    print(f"\nRunning {test}...")
    print("-" * 60)
    result = subprocess.run([sys.executable, test], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    if result.returncode != 0:
        print(f"❌ {test} failed!")
    else:
        print(f"✓ {test} passed!")

print("\n" + "=" * 60)
print("All tests completed!")
```

**Run**: `python run_all_tests.py`

---

## Testing Checklist

Use this checklist to verify everything works:

- [ ] Data can be loaded
- [ ] Data preprocessing works
- [ ] CF model can be trained
- [ ] CF predictions work
- [ ] CB similarity can be calculated
- [ ] CB recommendations work
- [ ] Hybrid recommendations work
- [ ] Models can be saved and loaded
- [ ] Evaluation metrics work
- [ ] CLI commands work
- [ ] End-to-end pipeline works

---

## Troubleshooting Tests

### Issue: Import errors
**Solution**: Make sure you're in the project root and dependencies are installed

### Issue: File not found errors
**Solution**: Run data preprocessing first: `python src/data_processing.py`

### Issue: Model not found errors
**Solution**: Train models first: `python run.py train`

---

## Next Steps

- Run tests regularly to catch issues early
- Add your own tests for specific use cases
- Use tests to verify changes don't break functionality
- See `docs/GETTING_STARTED.md` for more information

