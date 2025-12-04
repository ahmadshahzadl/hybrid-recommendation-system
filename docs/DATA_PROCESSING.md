# Data Processing Documentation

## Overview

The `data_processing.py` module handles all data preparation tasks. It loads raw data, cleans it, handles missing values, encodes categorical variables, and normalizes numerical features.

## What Does Data Processing Do?

Think of data processing as preparing ingredients before cooking:
- **Loading**: Getting the raw data from a file
- **Cleaning**: Removing or fixing errors and inconsistencies
- **Encoding**: Converting text categories (like "male"/"female") into numbers
- **Normalizing**: Scaling numbers to a common range so they can be compared fairly

## File Location

`src/data_processing.py`

## Key Functions

### 1. `load_data(file_path)`

**What it does**: Loads a CSV file into a pandas DataFrame.

**Parameters**:
- `file_path` (str): Path to the CSV file (e.g., `'data/dataset.csv'`)

**Returns**: 
- `pd.DataFrame`: The loaded dataset

**Example**:
```python
from src.data_processing import load_data

df = load_data('data/dataset.csv')
print(f"Loaded {len(df)} rows")
```

**What happens if it fails**:
- Raises `FileNotFoundError` if the file doesn't exist
- Raises `ValueError` if the file is empty or invalid

### 2. `preprocess_data(df)`

**What it does**: Cleans and transforms the raw data.

**Steps it performs**:
1. Creates User_ID and Product_ID if missing (based on feature combinations)
2. Fills missing numerical values with the mean
3. Fills missing categorical values with the mode (most common value)
4. Encodes categorical variables using one-hot encoding
5. Normalizes numerical columns using StandardScaler

**Parameters**:
- `df` (pd.DataFrame): Raw dataset

**Returns**:
- `pd.DataFrame`: Preprocessed dataset

**Example**:
```python
from src.data_processing import load_data, preprocess_data

# Load raw data
df = load_data('data/dataset.csv')

# Preprocess
df_processed = preprocess_data(df)

print(f"Original shape: {df.shape}")
print(f"Processed shape: {df_processed.shape}")
```

**What gets encoded**:
- `Gender` → `Gender_male` (0 or 1)
- `Holiday` → `Holiday_Yes` (0 or 1)
- `Season` → `Season_spring`, `Season_summer`, `Season_winter` (0 or 1)
- `Geographical locations` → `Geographical locations_mountains`, `Geographical locations_plains` (0 or 1)

**What gets normalized**:
- Number of clicks on similar products
- Number of similar products purchased so far
- Median purchasing price (in rupees)
- Rating of the product
- Price of the product

### 3. `save_processed_data(df, output_path)`

**What it does**: Saves the processed dataset to a CSV file.

**Parameters**:
- `df` (pd.DataFrame): Processed dataset
- `output_path` (str): Path where to save the file

**Example**:
```python
from src.data_processing import save_processed_data

save_processed_data(df_processed, 'data/processed_data.csv')
```

## How to Run

### Method 1: Command Line (Recommended for Beginners)

```bash
python src/data_processing.py
```

This uses default paths:
- Input: `data/dataset.csv`
- Output: `data/processed_data.csv`

### Method 2: Command Line with Custom Paths

```bash
python src/data_processing.py data/dataset.csv data/processed_data.csv
```

### Method 3: Using the Main CLI

```bash
python run.py preprocess --input data/dataset.csv --output data/processed_data.csv
```

### Method 4: Python Script

```python
from src.data_processing import load_data, preprocess_data, save_processed_data

# Load data
df = load_data('data/dataset.csv')

# Preprocess
df_processed = preprocess_data(df)

# Save
save_processed_data(df_processed, 'data/processed_data.csv')

print("Data processing complete!")
```

## Testing the Data Processing

### Test 1: Basic Functionality

```python
from src.data_processing import load_data, preprocess_data

# Test loading
df = load_data('data/dataset.csv')
assert not df.empty, "Data should not be empty"
print("✓ Data loaded successfully")

# Test preprocessing
df_processed = preprocess_data(df)
assert 'User_ID' in df_processed.columns, "User_ID should be created"
assert 'Product_ID' in df_processed.columns, "Product_ID should be created"
print("✓ Data preprocessed successfully")
print(f"  Original columns: {len(df.columns)}")
print(f"  Processed columns: {len(df_processed.columns)}")
```

### Test 2: Check for Missing Values

```python
import pandas as pd
from src.data_processing import load_data, preprocess_data

df = load_data('data/dataset.csv')
df_processed = preprocess_data(df)

# Check for missing values
missing = df_processed.isnull().sum().sum()
assert missing == 0, f"Should have no missing values, but found {missing}"
print("✓ No missing values in processed data")
```

### Test 3: Verify Data Types

```python
from src.data_processing import load_data, preprocess_data

df = load_data('data/dataset.csv')
df_processed = preprocess_data(df)

# Check that User_ID and Product_ID are numeric
assert pd.api.types.is_numeric_dtype(df_processed['User_ID']), "User_ID should be numeric"
assert pd.api.types.is_numeric_dtype(df_processed['Product_ID']), "Product_ID should be numeric"
print("✓ User_ID and Product_ID are numeric")
```

### Test 4: Verify Normalization

```python
import numpy as np
from src.data_processing import load_data, preprocess_data

df = load_data('data/dataset.csv')
df_processed = preprocess_data(df)

# Check that normalized columns have mean ~0 and std ~1
normalized_cols = [
    'Number of clicks on similar products',
    'Number of similar products purchased so far',
    'Median purchasing price (in rupees)',
    'Rating of the product',
    'Price of the product'
]

for col in normalized_cols:
    if col in df_processed.columns:
        mean = df_processed[col].mean()
        std = df_processed[col].std()
        assert abs(mean) < 0.01, f"{col} should have mean ~0, got {mean}"
        assert abs(std - 1.0) < 0.01, f"{col} should have std ~1, got {std}"

print("✓ Numerical columns are properly normalized")
```

## Expected Output

When you run the data processing, you should see:

```
2025-12-04 20:46:18,233 - INFO - Loading data from data/dataset.csv...
2025-12-04 20:46:18,241 - INFO - Successfully loaded dataset from data/dataset.csv
2025-12-04 20:46:18,241 - INFO - Dataset shape: (1474, 13)
2025-12-04 20:46:18,242 - INFO - Preprocessing data...
2025-12-04 20:46:18,242 - INFO - Starting data preprocessing...
2025-12-04 20:46:18,243 - INFO - User_ID column not found. Creating based on user features...
2025-12-04 20:46:18,252 - INFO - Product_ID column not found. Creating based on product features...
2025-12-04 20:46:18,259 - INFO - Encoding categorical variables...
2025-12-04 20:46:18,267 - INFO - Scaling numerical columns: [...]
2025-12-04 20:46:18,274 - INFO - Preprocessing complete. Final shape: (1474, 18)
2025-12-04 20:46:18,320 - INFO - Processed data saved to data/processed_data.csv
2025-12-04 20:46:18,321 - INFO - Data preprocessing completed successfully!

Processed data saved to: data/processed_data.csv
Shape: (1474, 18)
Columns: [...]
```

## Understanding the Output

- **Shape**: Shows (number of rows, number of columns)
- **Columns**: Lists all column names in the processed data
- **Log messages**: Show what steps were performed

## Common Issues

### Issue: "Dataset file not found"
**Cause**: The input file path is incorrect
**Solution**: Check that `data/dataset.csv` exists, or provide the correct path

### Issue: "Dataset is empty"
**Cause**: The CSV file has no data rows
**Solution**: Check your dataset file and ensure it has data

### Issue: Missing columns after preprocessing
**Cause**: Some expected columns don't exist in the input data
**Solution**: The code handles missing columns gracefully, but check your data format

## Data Flow

```
Raw CSV File (dataset.csv)
    ↓
load_data()
    ↓
Raw DataFrame
    ↓
preprocess_data()
    ↓
Processed DataFrame
    ↓
save_processed_data()
    ↓
Processed CSV File (processed_data.csv)
```

## Next Steps

After data processing:
1. Check the processed data file was created
2. Verify the shape and columns look correct
3. Proceed to train models: `python run.py train`

## Additional Resources

- See `docs/DATA_FORMAT.md` for information about required data format
- See `docs/TESTING_GUIDE.md` for more testing examples
- Check the Jupyter notebook: `notebooks/data_preprocessing.ipynb`

