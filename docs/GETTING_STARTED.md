# Getting Started Guide

Welcome to the Hybrid Recommendation System! This guide will help you get started, even if you're new to machine learning and recommendation systems.

## What is a Recommendation System?

A recommendation system suggests items (like products, movies, or music) to users based on their preferences and behavior. This project combines two approaches:

1. **Collaborative Filtering**: Recommends based on what similar users liked
2. **Content-Based Filtering**: Recommends based on item features (price, rating, etc.)

## Prerequisites

Before you begin, make sure you have:

- **Python 3.8 or higher** installed on your computer
- Basic knowledge of using the command line/terminal
- A text editor or IDE (like VS Code, PyCharm, or even Notepad++)

## Step 1: Install Python Dependencies

Open your terminal/command prompt and navigate to the project folder, then run:

```bash
pip install -r requirements.txt
```

**What this does**: Installs all the Python libraries needed for the project (pandas, scikit-learn, etc.)

**Expected output**: You'll see packages being downloaded and installed. This may take a few minutes.

**Troubleshooting**:
- If you get "pip is not recognized", make sure Python is installed and added to your PATH
- On Windows, you might need to use `python -m pip install -r requirements.txt`

## Step 2: Prepare Your Data

Place your dataset CSV file in the `data/` folder and name it `dataset.csv`.

**What you need**: A CSV file with columns like:
- User information (Gender, purchase history, etc.)
- Product information (Price, Rating, Brand, etc.)
- User-Product interactions (Ratings, clicks, etc.)

See `docs/DATA_FORMAT.md` for detailed information about the required data format.

## Step 3: Run Data Preprocessing

This step cleans and prepares your data for the recommendation system.

```bash
python src/data_processing.py
```

**What this does**:
- Loads your raw dataset
- Handles missing values
- Encodes categorical variables (like Gender, Season)
- Normalizes numerical features
- Creates User_ID and Product_ID if missing
- Saves processed data to `data/processed_data.csv`

**Expected output**:
```
2025-12-04 20:46:18,233 - INFO - Loading data from data/dataset.csv...
2025-12-04 20:46:18,241 - INFO - Successfully loaded dataset from data/dataset.csv
...
Processed data saved to: data/processed_data.csv
Shape: (1474, 18)
```

**How to verify it worked**: Check that `data/processed_data.csv` was created.

## Step 4: Train the Models

Train the recommendation models:

```bash
python run.py train --data data/processed_data.csv
```

**What this does**:
- Trains the Collaborative Filtering model
- Calculates the Content-Based similarity matrix
- Saves models to the `models/` folder

**Expected output**:
```
✓ Models trained and saved successfully!
```

**How to verify it worked**: Check that `models/cf_model.pkl` and `models/cb_similarity.csv` were created.

## Step 5: Get Your First Recommendation

Test the system with a simple recommendation:

```bash
python run.py recommend --user_id 5186 --product_id 9346
```

**What this does**: 
- Loads the trained models
- Calculates a recommendation score for the given user-product pair
- Displays the score (0-1, where 1 is the best recommendation)

**Expected output**:
```
✓ Hybrid Recommendation Score: 0.3749
  User ID: 5186
  Product ID: 9346
  Alpha (CF weight): 0.5
```

## Step 6: Get Top Recommendations

Get the top 10 product recommendations for a user:

```bash
python run.py top --user_id 5186 --top_n 10
```

**What this does**: 
- Finds the top N products that the user would most likely like
- Ranks them by recommendation score

**Expected output**:
```
✓ Top 10 Recommendations for User 5186:
--------------------------------------------------
  1. Product ID: 6408, Score: 0.4321
  2. Product ID: 7158, Score: 0.4320
  ...
```

## Step 7: Evaluate the Models

Check how well the models are performing:

```bash
python run.py evaluate --data data/processed_data.csv
```

**What this does**: 
- Tests the models on your data
- Calculates accuracy metrics (RMSE, MAE)
- Shows performance for each model type

**Expected output**:
```
============================================================
EVALUATION RESULTS
============================================================

COLLABORATIVE FILTERING:
----------------------------------------
  RMSE: 0.4319
  MAE: 0.2169
...
```

## Quick Start: Run Everything at Once

If you want to run the complete pipeline in one command:

```bash
python run.py pipeline
```

This will:
1. Preprocess the data
2. Train all models
3. Evaluate the models
4. Show example recommendations

## Understanding the Output

### Recommendation Score
- **Range**: 0.0 to 1.0
- **0.0**: Very low recommendation (user probably won't like this)
- **0.5**: Neutral recommendation
- **1.0**: Very high recommendation (user will likely love this)

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Lower is better (measures prediction accuracy)
- **MAE (Mean Absolute Error)**: Lower is better (average prediction error)
- **Similarity Score**: Higher is better (how similar products are)

## Next Steps

1. **Experiment with Alpha**: Try different alpha values (0.0 to 1.0) to see how it affects recommendations
   ```bash
   python run.py recommend --user_id 5186 --product_id 9346 --alpha 0.7
   ```

2. **Explore the Notebooks**: Open the Jupyter notebooks in the `notebooks/` folder to see step-by-step examples
   ```bash
   jupyter notebook notebooks/
   ```

3. **Read the Detailed Documentation**: 
   - `docs/DATA_PROCESSING.md` - Learn about data preprocessing
   - `docs/COLLABORATIVE_FILTERING.md` - Understand collaborative filtering
   - `docs/CONTENT_BASED_FILTERING.md` - Learn about content-based filtering
   - `docs/HYBRID_MODEL.md` - Understand the hybrid approach
   - `docs/TESTING_GUIDE.md` - Learn how to test each component

## Common Issues and Solutions

### Issue: "File not found" error
**Solution**: Make sure you're running commands from the project root directory (where `run.py` is located)

### Issue: "Module not found" error
**Solution**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

### Issue: Models take too long to train
**Solution**: This is normal for the first run. Subsequent runs will be faster as models are cached.

### Issue: Low recommendation scores
**Solution**: 
- Check if your data has enough user-item interactions
- Try adjusting the alpha parameter
- Ensure your data is properly preprocessed

## Getting Help

If you encounter issues:
1. Check the `docs/TROUBLESHOOTING.md` guide
2. Review the error messages carefully
3. Make sure all prerequisites are installed
4. Verify your data format matches the requirements

## What's Next?

Now that you've got the basics working, you can:
- Customize the models for your specific use case
- Experiment with different features
- Tune the alpha parameter for better results
- Integrate the system into your own application

Happy recommending! 🎉

