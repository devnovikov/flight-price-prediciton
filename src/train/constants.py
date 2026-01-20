"""Constants for flight price prediction model training.

This module contains all constants used across training and inference
to ensure reproducibility and consistency.
"""

# Reproducibility
RANDOM_STATE = 42

# Target column
TARGET_COLUMN = "Price"

# Feature columns after preprocessing
FEATURE_COLUMNS = [
    'Journey_day',
    'Journey_month',
    'Journey_weekday',
    'Dep_hour',
    'Dep_min',
    'Arrival_hour',
    'Arrival_min',
    'Duration_minutes',
    'Airline_encoded',
    'Source_encoded',
    'Destination_encoded',
    'Total_Stops_encoded'
]

# Categorical columns from raw data
CATEGORICAL_COLUMNS = [
    'Airline',
    'Source',
    'Destination',
    'Total_Stops'
]

# Model paths
MODEL_PATH = "models/best_model.joblib"
DATASET_PATH = "dataset/Clean_Dataset.csv"

# Training configuration
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 20  # Number of iterations for RandomizedSearchCV

# Hyperparameter search spaces
RF_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

XGB_PARAMS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

CAT_PARAMS = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7]
}

# API configuration
API_VERSION = "1.0.0"
CURRENCY = "INR"
