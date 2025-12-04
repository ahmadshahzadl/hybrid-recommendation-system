# Command Line Interface (CLI) Reference

## Overview

The `run.py` script provides a command-line interface for all operations in the Hybrid Recommendation System. It's the easiest way to use the system without writing Python code.

## File Location

`run.py` (in project root)

## Basic Usage

```bash
python run.py <command> [options]
```

## Available Commands

### 1. `preprocess` - Preprocess Data

Cleans and prepares your raw dataset.

**Usage**:
```bash
python run.py preprocess [--input INPUT] [--output OUTPUT]
```

**Options**:
- `--input`: Input dataset path (default: `data/dataset.csv`)
- `--output`: Output processed data path (default: `data/processed_data.csv`)

**Examples**:
```bash
# Use default paths
python run.py preprocess

# Use custom paths
python run.py preprocess --input my_data.csv --output my_processed_data.csv
```

**What it does**:
- Loads raw data
- Handles missing values
- Encodes categorical variables
- Normalizes numerical features
- Saves processed data

**Expected output**:
```
✓ Processed data saved to: data/processed_data.csv
  Shape: (1474, 18)
```

---

### 2. `train` - Train Models

Trains the Collaborative Filtering and Content-Based models.

**Usage**:
```bash
python run.py train [--data DATA] [--alpha ALPHA] [--retrain]
```

**Options**:
- `--data`: Processed data path (default: `data/processed_data.csv`)
- `--alpha`: Weight for Collaborative Filtering, 0-1 (default: `0.5`)
- `--retrain`: Force retrain even if models exist

**Examples**:
```bash
# Train with default settings
python run.py train

# Train with custom alpha
python run.py train --alpha 0.7

# Force retrain
python run.py train --retrain

# Custom data path and alpha
python run.py train --data my_data.csv --alpha 0.8
```

**What it does**:
- Trains Collaborative Filtering model
- Calculates Content-Based similarity matrix
- Saves models to `models/` directory

**Expected output**:
```
✓ Models trained and saved successfully!
```

---

### 3. `recommend` - Get Single Recommendation

Gets a recommendation score for a specific user-product pair.

**Usage**:
```bash
python run.py recommend --user_id USER_ID --product_id PRODUCT_ID [--alpha ALPHA] [--data DATA]
```

**Required Options**:
- `--user_id`: User ID (integer)
- `--product_id`: Product ID (integer)

**Optional Options**:
- `--alpha`: Weight for CF, 0-1 (default: `0.5`)
- `--data`: Processed data path (default: `data/processed_data.csv`)

**Examples**:
```bash
# Basic recommendation
python run.py recommend --user_id 5186 --product_id 9346

# With custom alpha
python run.py recommend --user_id 5186 --product_id 9346 --alpha 0.7

# With custom data
python run.py recommend --user_id 123 --product_id 456 --data my_data.csv
```

**What it does**:
- Loads trained models
- Calculates hybrid recommendation score
- Displays the score (0.0 to 1.0)

**Expected output**:
```
✓ Hybrid Recommendation Score: 0.3749
  User ID: 5186
  Product ID: 9346
  Alpha (CF weight): 0.5
```

---

### 4. `top` - Get Top Recommendations

Gets the top N product recommendations for a user.

**Usage**:
```bash
python run.py top --user_id USER_ID [--top_n TOP_N] [--alpha ALPHA] [--data DATA]
```

**Required Options**:
- `--user_id`: User ID (integer)

**Optional Options**:
- `--top_n`: Number of recommendations (default: `10`)
- `--alpha`: Weight for CF, 0-1 (default: `0.5`)
- `--data`: Processed data path (default: `data/processed_data.csv`)

**Examples**:
```bash
# Get top 10 recommendations
python run.py top --user_id 5186

# Get top 5 recommendations
python run.py top --user_id 5186 --top_n 5

# With custom alpha
python run.py top --user_id 5186 --top_n 10 --alpha 0.7
```

**What it does**:
- Finds top N products for the user
- Ranks by recommendation score
- Displays sorted list

**Expected output**:
```
✓ Top 10 Recommendations for User 5186:
--------------------------------------------------
  1. Product ID: 6408, Score: 0.4321
  2. Product ID: 7158, Score: 0.4320
  3. Product ID: 6201, Score: 0.4320
  ...
```

---

### 5. `evaluate` - Evaluate Models

Evaluates all models and shows performance metrics.

**Usage**:
```bash
python run.py evaluate [--data DATA] [--alpha ALPHA]
```

**Options**:
- `--data`: Processed data path (default: `data/processed_data.csv`)
- `--alpha`: Weight for CF in hybrid model, 0-1 (default: `0.5`)

**Examples**:
```bash
# Evaluate with default settings
python run.py evaluate

# With custom alpha
python run.py evaluate --alpha 0.7

# With custom data
python run.py evaluate --data my_data.csv --alpha 0.8
```

**What it does**:
- Evaluates Collaborative Filtering model
- Evaluates Content-Based model
- Evaluates Hybrid model
- Displays metrics (RMSE, MAE, etc.)

**Expected output**:
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
  ...

HYBRID:
----------------------------------------
  RMSE: 3.4080
  MAE: 3.1967
  ...
```

---

### 6. `pipeline` - Run Complete Pipeline

Runs the entire pipeline: preprocess, train, evaluate, and optionally show recommendations.

**Usage**:
```bash
python run.py pipeline [--input INPUT] [--output OUTPUT] [--alpha ALPHA] [--user_id USER_ID] [--product_id PRODUCT_ID]
```

**Options**:
- `--input`: Input dataset path (default: `data/dataset.csv`)
- `--output`: Output processed data path (default: `data/processed_data.csv`)
- `--alpha`: Weight for CF, 0-1 (default: `0.5`)
- `--user_id`: User ID for example recommendation (optional)
- `--product_id`: Product ID for example recommendation (optional)

**Examples**:
```bash
# Run complete pipeline
python run.py pipeline

# With example recommendation
python run.py pipeline --user_id 5186 --product_id 9346

# With custom settings
python run.py pipeline --input my_data.csv --alpha 0.7 --user_id 123 --product_id 456
```

**What it does**:
1. Preprocesses data
2. Trains all models
3. Shows example recommendation (if user_id and product_id provided)
4. Evaluates all models

**Expected output**:
```
============================================================
Running Complete Pipeline
============================================================

[1/4] Preprocessing data...
✓ Processed data saved to: data/processed_data.csv

[2/4] Training models...
✓ Models trained and saved successfully!

[3/4] Generating example recommendation...
✓ Hybrid Recommendation Score: 0.3749

[4/4] Evaluating models...
============================================================
EVALUATION RESULTS
============================================================
...

============================================================
Pipeline completed successfully!
============================================================
```

---

## Getting Help

### View All Commands

```bash
python run.py --help
```

### View Help for Specific Command

```bash
python run.py <command> --help
```

**Example**:
```bash
python run.py recommend --help
```

---

## Common Workflows

### Workflow 1: First Time Setup

```bash
# 1. Preprocess data
python run.py preprocess

# 2. Train models
python run.py train

# 3. Test with a recommendation
python run.py recommend --user_id 5186 --product_id 9346

# 4. Evaluate models
python run.py evaluate
```

### Workflow 2: Quick Start (All in One)

```bash
python run.py pipeline --user_id 5186 --product_id 9346
```

### Workflow 3: Experiment with Alpha

```bash
# Train with different alpha
python run.py train --alpha 0.7

# Get recommendations
python run.py recommend --user_id 5186 --product_id 9346 --alpha 0.7

# Evaluate
python run.py evaluate --alpha 0.7
```

### Workflow 4: Get Recommendations for Multiple Users

```bash
# User 1
python run.py top --user_id 5186 --top_n 10

# User 2
python run.py top --user_id 1234 --top_n 10

# User 3
python run.py top --user_id 5678 --top_n 10
```

---

## Tips and Tricks

### 1. Use Default Paths

If you use the standard file structure, you can omit path arguments:
```bash
python run.py preprocess  # Uses data/dataset.csv and data/processed_data.csv
```

### 2. Model Caching

Models are automatically cached. To force retrain:
```bash
python run.py train --retrain
```

### 3. Batch Processing

You can create a script to process multiple users:
```bash
# Create a file: batch_recommendations.sh
for user_id in 5186 1234 5678; do
    python run.py top --user_id $user_id --top_n 10
done
```

### 4. Save Output to File

Redirect output to a file:
```bash
python run.py evaluate > evaluation_results.txt
```

---

## Troubleshooting

### Issue: "command not found"
**Solution**: Make sure you're in the project root directory where `run.py` is located

### Issue: "No such file or directory"
**Solution**: Check that your data files exist, or provide correct paths with `--input` or `--data`

### Issue: "Models not found"
**Solution**: Run `python run.py train` first to train the models

### Issue: "Invalid user_id or product_id"
**Solution**: Make sure the IDs exist in your processed dataset

---

## Next Steps

- See `docs/GETTING_STARTED.md` for beginner guide
- See `docs/TESTING_GUIDE.md` for testing examples
- See individual module docs for Python API usage

