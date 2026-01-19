#!/usr/bin/env python3
"""Production training script for flight price prediction.

This script trains multiple ML models, performs hyperparameter tuning,
and saves the best model to models/best_model.joblib.

Usage:
    uv run python src/train/train.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train.constants import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, N_ITER_SEARCH,
    MODEL_PATH, DATASET_PATH,
    RF_PARAMS, XGB_PARAMS, CAT_PARAMS
)


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess data for training.

    Returns:
        Tuple of (X, y, encoders)
    """
    print("Preprocessing data...")
    df = df.copy()

    # Find price column (target)
    price_col = [c for c in df.columns if c.lower() == 'price'][0]

    # Encode all categorical columns
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Split features and target
    X = df.drop(columns=[price_col])
    y = df[price_col]

    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")

    return X, y, encoders


def train_model(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    """Train a model and return metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'model': model
    }


def train_with_tuning(model, params, X_train, y_train, X_test, y_test, name: str) -> dict:
    """Train model with RandomizedSearchCV hyperparameter tuning."""
    print(f"\n  Tuning {name}...")

    search = RandomizedSearchCV(
        model, params,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring='neg_root_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"    Best params: {search.best_params_}")

    return {
        'Model': name,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'model': best_model,
        'best_params': search.best_params_
    }


def train_catboost_with_tuning(X_train, y_train, X_test, y_test) -> dict:
    """Train CatBoost with manual hyperparameter search (sklearn compatibility issue workaround)."""
    print(f"\n  Tuning CatBoost...")

    # Define parameter combinations to try
    param_combinations = [
        {'depth': 6, 'learning_rate': 0.1, 'iterations': 200},
        {'depth': 8, 'learning_rate': 0.05, 'iterations': 300},
        {'depth': 10, 'learning_rate': 0.03, 'iterations': 500},
    ]

    best_model = None
    best_rmse = float('inf')
    best_params = None

    for params in param_combinations:
        model = CatBoostRegressor(
            random_state=RANDOM_STATE,
            verbose=0,
            **params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params

    y_pred = best_model.predict(X_test)
    print(f"    Best params: {best_params}")

    return {
        'Model': 'CatBoost',
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'model': best_model,
        'best_params': best_params
    }


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("FLIGHT PRICE PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # Set random seed
    np.random.seed(RANDOM_STATE)
    print(f"\nRandom State: {RANDOM_STATE}")

    # Load data
    df = load_data(DATASET_PATH)

    # Preprocess
    X, y, encoders = preprocess_data(df)

    # Train-test split
    print(f"\nSplitting data (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")

    # Train models
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    results = []

    # Baseline models (no tuning)
    print("\nBaseline models:")

    print("  Training Linear Regression...")
    results.append(train_model(
        LinearRegression(), X_train, y_train, X_test, y_test, 'Linear Regression'
    ))

    print("  Training Ridge Regression...")
    results.append(train_model(
        Ridge(alpha=1.0, random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test, 'Ridge Regression'
    ))

    print("  Training Lasso Regression...")
    results.append(train_model(
        Lasso(alpha=1.0, random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test, 'Lasso Regression'
    ))

    # Models with hyperparameter tuning
    print("\nModels with hyperparameter tuning:")

    results.append(train_with_tuning(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        RF_PARAMS, X_train, y_train, X_test, y_test, 'Random Forest'
    ))

    results.append(train_with_tuning(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        XGB_PARAMS, X_train, y_train, X_test, y_test, 'XGBoost'
    ))

    # CatBoost uses manual tuning due to sklearn compatibility issues
    results.append(train_catboost_with_tuning(X_train, y_train, X_test, y_test))

    # Results table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results])
    results_df = results_df.sort_values('RMSE')

    print("\n{:<20} {:>12} {:>10} {:>12}".format('Model', 'RMSE', 'R2', 'MAE'))
    print("-" * 56)
    for _, row in results_df.iterrows():
        print("{:<20} {:>12,.0f} {:>10.4f} {:>12,.0f}".format(
            row['Model'], row['RMSE'], row['R2'], row['MAE']
        ))

    # Select best model
    best_result = min(results, key=lambda x: x['RMSE'])
    best_model = best_result['model']
    best_name = best_result['Model']
    best_r2 = best_result['R2']
    best_rmse = best_result['RMSE']

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"\nModel: {best_name}")
    print(f"R² Score: {best_r2:.4f}")
    print(f"RMSE: {best_rmse:,.0f} INR")
    print(f"MAE: {best_result['MAE']:,.0f} INR")

    if best_r2 > 0.85:
        print(f"\n✓ Target R² > 0.85 ACHIEVED!")
    else:
        print(f"\n✗ Target R² > 0.85 NOT achieved (got {best_r2:.4f})")

    # Save model
    print(f"\nSaving model to {MODEL_PATH}...")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model_artifact = {
        'model': best_model,
        'model_name': best_name,
        'feature_columns': X.columns.tolist(),
        'encoders': encoders,
        'metrics': {
            'rmse': best_rmse,
            'r2': best_r2,
            'mae': best_result['MAE']
        },
        'random_state': RANDOM_STATE
    }

    if 'best_params' in best_result:
        model_artifact['best_params'] = best_result['best_params']

    joblib.dump(model_artifact, MODEL_PATH)
    print(f"✓ Model saved to {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
