# Data Format Documentation

## Overview

This document describes the required data format for the Hybrid Recommendation System. Understanding the data format is crucial for getting good results.

## Required CSV Format

Your dataset should be a CSV file with the following columns:

## Column Descriptions

### User Features

These columns describe user characteristics and behavior:

1. **`Gender`** (Categorical)
   - Type: String
   - Values: `"male"`, `"female"`, or other categories
   - Example: `"male"`, `"female"`
   - **Required**: Yes (will be encoded during preprocessing)

2. **`Median purchasing price (in rupees)`** (Numerical)
   - Type: Float/Integer
   - Description: Average price the user typically pays
   - Example: `500`, `3000`, `1000`
   - **Required**: Yes (will be normalized)

3. **`Number of clicks on similar products`** (Numerical)
   - Type: Integer
   - Description: How many times user clicked on similar products
   - Example: `12`, `8`, `25`
   - **Required**: Yes (will be normalized)

4. **`Number of similar products purchased so far`** (Numerical)
   - Type: Integer
   - Description: Count of similar products user has bought
   - Example: `4`, `2`, `10`
   - **Required**: Yes (will be normalized)

5. **`Average rating given to similar products`** (Numerical)
   - Type: Float
   - Description: User's average rating for similar products
   - Example: `4.2`, `3.8`, `4.5`
   - **Required**: Optional (used for analysis)

### Product Features

These columns describe product characteristics:

6. **`Brand of the product`** (Categorical)
   - Type: String
   - Description: Product brand name
   - Example: `"PUMA"`, `"Lee"`, `"Head Hunters"`
   - **Required**: Optional (not used in similarity calculation by default)

7. **`Price of the product`** (Numerical)
   - Type: Float/Integer
   - Description: Product price
   - Example: `200`, `300`, `1000`
   - **Required**: Yes (used in content-based filtering)

8. **`Rating of the product`** (Numerical)
   - Type: Float
   - Range: Typically 0.0 to 5.0
   - Description: Product's average rating
   - Example: `4.5`, `3.2`, `4.8`
   - **Required**: Yes (used in content-based filtering and CF)

9. **`Customer review sentiment score (overall)`** (Numerical)
   - Type: Float
   - Range: Typically -1.0 to 1.0
   - Description: Overall sentiment from customer reviews
   - Example: `0.8`, `-0.4`, `0.6`
   - **Required**: Yes (used in content-based filtering)

### Context Features

These columns describe the context of the interaction:

10. **`Holiday`** (Categorical)
    - Type: String
    - Values: `"Yes"`, `"No"`
    - Description: Whether it's a holiday
    - Example: `"Yes"`, `"No"`
    - **Required**: Yes (will be encoded)

11. **`Season`** (Categorical)
    - Type: String
    - Values: `"spring"`, `"summer"`, `"winter"`, `"monsoon"`
    - Description: Current season
    - Example: `"winter"`, `"summer"`, `"spring"`
    - **Required**: Yes (will be encoded)

12. **`Geographical locations`** (Categorical)
    - Type: String
    - Values: `"plains"`, `"mountains"`, `"coastal"`
    - Description: User's geographical location
    - Example: `"plains"`, `"mountains"`, `"coastal"`
    - **Required**: Yes (will be encoded)

### Target Variable (Optional)

13. **`Probability for the product to be recommended to the person`** (Numerical)
    - Type: Float
    - Range: 0.0 to 1.0
    - Description: Ground truth recommendation probability (if available)
    - Example: `0.9`, `0.2`, `0.7`
    - **Required**: No (used for evaluation if available)

## Auto-Generated Columns

If these columns don't exist, they will be automatically created during preprocessing:

- **`User_ID`**: Unique identifier for each user (created from user features)
- **`Product_ID`**: Unique identifier for each product (created from product features)

## Example Data

Here's what a sample row looks like:

```csv
Number of clicks on similar products,Number of similar products purchased so far,Average rating given to similar products,Gender,Median purchasing price (in rupees),Rating of the product,Brand of the product,Customer review sentiment score (overall),Price of the product,Holiday,Season,Geographical locations,Probability for the product to be recommended to the person
12,4,4.2,male,500,4.5,PUMA,0.8,200,No,winter,plains,0.9
8,2,3.8,female,3000,3.2,Lee,-0.4,300,Yes,monsoon,mountains,0.2
```

## Data Requirements

### Minimum Requirements

- At least **10 rows** of data
- At least **10 unique users**
- At least **10 unique products**
- At least **10 user-item interactions** with ratings

### Recommended

- **100+ rows** for better results
- **50+ unique users**
- **50+ unique products**
- **Multiple interactions per user** (at least 2-3)

## Data Quality Guidelines

### Missing Values

- **Numerical columns**: Will be filled with the mean value
- **Categorical columns**: Will be filled with the mode (most common value)
- **Best practice**: Try to minimize missing values in your data

### Data Types

- Ensure numerical columns contain only numbers
- Ensure categorical columns contain consistent values
- Check for typos in categorical values (e.g., "male" vs "Male")

### Rating Values

- Ratings should be in a consistent range (typically 1-5 or 0-5)
- Invalid ratings will be clipped to valid range
- Missing ratings will be handled automatically

## Preprocessing Transformations

### What Happens to Your Data

1. **Missing Value Handling**:
   - Numerical: Filled with mean
   - Categorical: Filled with mode

2. **Encoding**:
   - `Gender`: `male` → `Gender_male` (0 or 1)
   - `Holiday`: `Yes` → `Holiday_Yes` (0 or 1)
   - `Season`: `spring` → `Season_spring` (0 or 1), etc.
   - `Geographical locations`: `plains` → `Geographical locations_plains` (0 or 1), etc.

3. **Normalization**:
   - Numerical columns are scaled to have mean=0 and std=1
   - This includes: clicks, purchases, prices, ratings

4. **ID Generation**:
   - If User_ID/Product_ID missing, they're created from feature combinations

## Customizing Features

### For Content-Based Filtering

You can specify which features to use for similarity:

```python
from src.content_based_filtering import calculate_content_similarity
import pandas as pd

df = pd.read_csv('data/processed_data.csv')

# Use custom features
custom_features = [
    'Price of the product',
    'Rating of the product',
    'Number of clicks on similar products'  # Custom feature
]

similarity_matrix = calculate_content_similarity(
    df,
    feature_columns=custom_features
)
```

## Data Validation

### Check Your Data Before Processing

```python
import pandas as pd

# Load your data
df = pd.read_csv('data/dataset.csv')

# Check shape
print(f"Shape: {df.shape}")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

# Check categorical values
print("\nCategorical values:")
print(f"Gender: {df['Gender'].unique()}")
print(f"Holiday: {df['Holiday'].unique()}")
print(f"Season: {df['Season'].unique()}")

# Check numerical ranges
print("\nNumerical ranges:")
print(df.describe())
```

## Common Data Issues

### Issue: "Missing required columns"
**Solution**: Check that all required columns exist in your CSV file

### Issue: "Insufficient data for training"
**Solution**: Ensure you have at least 10 user-item interactions

### Issue: Inconsistent categorical values
**Solution**: Standardize values (e.g., all "male" not "Male" or "MALE")

### Issue: Outlier values
**Solution**: Check for unrealistic values (e.g., negative prices, ratings > 5)

## Sample Data File

A minimal example dataset:

```csv
Number of clicks on similar products,Number of similar products purchased so far,Average rating given to similar products,Gender,Median purchasing price (in rupees),Rating of the product,Brand of the product,Customer review sentiment score (overall),Price of the product,Holiday,Season,Geographical locations
10,3,4.0,male,500,4.5,BrandA,0.7,200,No,winter,plains
15,5,4.2,female,300,4.3,BrandB,0.8,150,Yes,summer,coastal
8,2,3.8,male,1000,3.9,BrandC,0.5,500,No,spring,mountains
```

## Next Steps

1. Prepare your data in the required format
2. Validate your data using the checks above
3. Run preprocessing: `python src/data_processing.py`
4. Check the processed data output
5. Proceed with training models

## Additional Resources

- See `docs/DATA_PROCESSING.md` for preprocessing details
- See `docs/GETTING_STARTED.md` for setup instructions
- Check `notebooks/data_preprocessing.ipynb` for examples

