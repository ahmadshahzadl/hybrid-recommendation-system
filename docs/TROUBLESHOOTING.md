# Troubleshooting Guide

Common issues and their solutions for the Hybrid Recommendation System.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Issues](#data-issues)
- [Runtime Errors](#runtime-errors)
- [Model Issues](#model-issues)
- [Performance Issues](#performance-issues)
- [Output Issues](#output-issues)

---

## Installation Issues

### Issue: "pip is not recognized"

**Symptoms**: Command prompt says 'pip' is not recognized as a command.

**Solutions**:
1. Make sure Python is installed: `python --version`
2. Use `python -m pip` instead: `python -m pip install -r requirements.txt`
3. Add Python to your PATH environment variable
4. On Windows, try: `py -m pip install -r requirements.txt`

---

### Issue: "surprise library not available"

**Symptoms**: Warning message about surprise library not being available.

**Solutions**:
- **This is normal!** The system automatically uses scikit-learn's TruncatedSVD as a fallback
- The warning can be ignored - functionality is not affected
- If you want to install surprise (optional):
  ```bash
  pip install scikit-surprise
  ```
  Note: May not work on Windows with Python 3.12+

---

### Issue: Package installation fails

**Symptoms**: Error when running `pip install -r requirements.txt`.

**Solutions**:
1. Update pip: `python -m pip install --upgrade pip`
2. Install packages one by one to identify the problematic one
3. Check Python version (need 3.8+): `python --version`
4. On Windows, you might need Visual C++ Build Tools for some packages

---

## Data Issues

### Issue: "Dataset file not found"

**Symptoms**: `FileNotFoundError: Dataset file not found: data/dataset.csv`

**Solutions**:
1. Check that `data/dataset.csv` exists
2. Verify you're running commands from the project root directory
3. Use absolute path: `python src/data_processing.py /full/path/to/dataset.csv`
4. Check file permissions

---

### Issue: "Dataset is empty"

**Symptoms**: Error saying dataset is empty.

**Solutions**:
1. Check your CSV file has data rows (not just headers)
2. Verify the file isn't corrupted
3. Open the file in a text editor to check contents
4. Ensure file encoding is UTF-8

---

### Issue: "Missing required columns"

**Symptoms**: Error about missing User_ID, Product_ID, or Rating columns.

**Solutions**:
1. Check your data has the required columns (see `docs/DATA_FORMAT.md`)
2. The system can auto-generate User_ID and Product_ID if missing
3. Ensure column names match exactly (case-sensitive)
4. Check for typos in column names

---

### Issue: "Insufficient data for training"

**Symptoms**: Error saying you need at least 10 ratings.

**Solutions**:
1. Ensure you have at least 10 user-item interactions
2. Check that ratings are not all missing
3. Verify User_ID and Product_ID columns exist
4. Use a larger dataset

---

## Runtime Errors

### Issue: "Module not found" or "ImportError"

**Symptoms**: Python can't find modules like `src.data_processing`.

**Solutions**:
1. Make sure you're in the project root directory
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Check that `src/` directory exists with `__init__.py`
4. Try: `python -m src.data_processing` instead

---

### Issue: "Can't get local object" (pickle error)

**Symptoms**: Error when saving/loading models.

**Solutions**:
- This should be fixed in the current version
- If you see this, update to the latest code
- Delete old model files and retrain: `rm -rf models/ && python run.py train`

---

### Issue: "Product ID not found"

**Symptoms**: Warning about Product ID not being in similarity matrix.

**Solutions**:
1. This is handled automatically - returns default score (0.5)
2. Ensure the product exists in your processed data
3. Retrain models if you've added new products: `python run.py train --retrain`
4. Check Product_ID type matches (int vs string)

---

### Issue: "System not initialized"

**Symptoms**: Error when calling predict without initialization.

**Solutions**:
1. Use the function interface: `hybrid_recommendation()` (auto-initializes)
2. Or call `system.initialize()` if using the class interface
3. Make sure models are trained: `python run.py train`

---

## Model Issues

### Issue: Models not found

**Symptoms**: Error loading models from `models/` directory.

**Solutions**:
1. Train models first: `python run.py train`
2. Check that `models/` directory exists
3. Verify model files exist: `models/cf_model.pkl` and `models/cb_similarity.csv`
4. Delete and retrain if corrupted: `rm -rf models/ && python run.py train`

---

### Issue: Models take too long to train

**Symptoms**: Training takes a very long time.

**Solutions**:
1. **This is normal for first run** - subsequent runs use cached models
2. Reduce dataset size for testing
3. Check your system has enough RAM
4. For very large datasets, consider sampling

---

### Issue: Low recommendation scores

**Symptoms**: All recommendations have low scores (< 0.3).

**Solutions**:
1. Check data quality - ensure ratings and features are valid
2. Try different alpha values: `--alpha 0.7` or `--alpha 0.3`
3. Verify you have enough user-item interactions
4. Check that preprocessing worked correctly

---

### Issue: Same recommendations for all users

**Symptoms**: All users get the same product recommendations.

**Solutions**:
1. Check that User_ID is being used correctly
2. Verify you have diverse user data
3. Try adjusting alpha parameter
4. Ensure CF model is trained properly

---

## Performance Issues

### Issue: Out of memory errors

**Symptoms**: System runs out of memory during training.

**Solutions**:
1. Reduce dataset size for testing
2. Close other applications
3. Use a machine with more RAM
4. Process data in batches (requires code modification)

---

### Issue: Slow predictions

**Symptoms**: Getting recommendations takes a long time.

**Solutions**:
1. First call is slower (loads models) - subsequent calls are faster
2. Ensure models are cached (don't retrain every time)
3. Reduce `top_n` parameter if getting many recommendations
4. Check system resources (CPU, RAM)

---

## Output Issues

### Issue: Unexpected recommendation scores

**Symptoms**: Scores seem wrong or inconsistent.

**Solutions**:
1. Check that data preprocessing completed successfully
2. Verify models were trained on the same data
3. Try retraining: `python run.py train --retrain`
4. Check alpha parameter is appropriate for your data

---

### Issue: Evaluation metrics seem wrong

**Symptoms**: RMSE/MAE values are very high or very low.

**Solutions**:
1. Check that evaluation is using the same data format as training
2. Verify ratings are in expected range (1-5)
3. Ensure enough samples for evaluation
4. Compare with individual model metrics

---

### Issue: No output or empty results

**Symptoms**: Commands run but produce no output.

**Solutions**:
1. Check that data was processed successfully
2. Verify models are trained
3. Ensure user_id and product_id exist in your data
4. Check log files: `recommendation_system.log`

---

## General Debugging Tips

### 1. Check Logs

Logs are saved to `recommendation_system.log`. Check this file for detailed error messages.

### 2. Verify Data

```python
import pandas as pd
df = pd.read_csv('data/processed_data.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())
```

### 3. Test Individual Components

Test each component separately:
```bash
# Test data processing
python src/data_processing.py

# Test training
python run.py train

# Test recommendation
python run.py recommend --user_id 5186 --product_id 9346
```

### 4. Check File Paths

Make sure you're running commands from the project root directory (where `run.py` is located).

### 5. Verify Dependencies

```bash
python -c "import pandas, sklearn, numpy; print('All OK')"
```

---

## Getting More Help

If you've tried the solutions above and still have issues:

1. **Check the logs**: Look at `recommendation_system.log` for detailed errors
2. **Review documentation**: Check relevant docs in `docs/` directory
3. **Test with sample data**: Use the provided dataset to verify setup
4. **Check Python version**: Ensure you're using Python 3.8 or higher
5. **Verify installation**: Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

---

## Common Error Messages

### "FileNotFoundError"
→ Check file paths and that files exist

### "ValueError"
→ Check data format and values

### "KeyError"
→ Check column names match exactly

### "AttributeError"
→ Check object types and method names

### "MemoryError"
→ Reduce dataset size or increase available memory

---

## Prevention Tips

1. **Always preprocess data first**: `python src/data_processing.py`
2. **Train models before using**: `python run.py train`
3. **Check data quality**: Validate your data before processing
4. **Use consistent paths**: Stick to default paths or document custom ones
5. **Keep dependencies updated**: Regularly update packages

---

## Still Stuck?

If none of these solutions work:

1. Check the specific error message in detail
2. Review the relevant documentation file
3. Verify your environment matches requirements
4. Try with the sample dataset first
5. Check GitHub issues (if applicable)

---

**Remember**: Most issues are related to:
- Missing or incorrect data
- Models not being trained
- Wrong file paths
- Missing dependencies

Start by checking these common causes!

