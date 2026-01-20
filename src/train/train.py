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
    """Train CatBoost with manual hyperparameter search matching notebook's RandomizedSearchCV behavior.

    Note: Due to CatBoost/sklearn compatibility issues with RandomizedSearchCV,
    we use manual cross-validation with the same parameter space and number of
    iterations as the notebook to ensure reproducible results.
    """
    from sklearn.model_selection import KFold
    import itertools
    import random

    print(f"\n  Tuning CatBoost...")

    # Same parameter space as notebook and constants.py
    param_space = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7]
    }

    # Generate all combinations
    keys = list(param_space.keys())
    all_combinations = list(itertools.product(*[param_space[k] for k in keys]))
    all_combinations = [dict(zip(keys, combo)) for combo in all_combinations]

    # Sample N_ITER_SEARCH combinations with same random state as notebook
    random.seed(RANDOM_STATE)
    n_iter = min(N_ITER_SEARCH, len(all_combinations))
    sampled_params = random.sample(all_combinations, n_iter)

    # 5-fold cross-validation
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_model = None
    best_cv_rmse = float('inf')
    best_params = None

    for i, params in enumerate(sampled_params):
        cv_rmses = []

        for train_idx, val_idx in kfold.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = CatBoostRegressor(
                random_state=RANDOM_STATE,
                verbose=0,
                **params
            )
            model.fit(X_tr, y_tr)
            y_pred_val = model.predict(X_val)
            cv_rmses.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))

        mean_cv_rmse = np.mean(cv_rmses)

        if mean_cv_rmse < best_cv_rmse:
            best_cv_rmse = mean_cv_rmse
            best_params = params

    # Train final model on full training set with best params
    best_model = CatBoostRegressor(
        random_state=RANDOM_STATE,
        verbose=0,
        **best_params
    )
    best_model.fit(X_train, y_train)
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
