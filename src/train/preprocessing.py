"""Preprocessing utilities for flight price prediction.

This module provides functions for feature engineering and data transformation
used during both training and inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def convert_duration(duration: str) -> int:
    """Convert duration string to minutes.

    Args:
        duration: Duration string in format "Xh Ym", "Xh", or "Ym"

    Returns:
        Total duration in minutes

    Examples:
        >>> convert_duration("2h 30m")
        150
        >>> convert_duration("1h")
        60
        >>> convert_duration("45m")
        45
    """
    if pd.isna(duration) or duration == '':
        return 0

    duration = str(duration).strip()
    total_minutes = 0

    if 'h' in duration and 'm' in duration:
        parts = duration.split('h')
        hours = int(parts[0].strip())
        minutes = int(parts[1].replace('m', '').strip())
        total_minutes = hours * 60 + minutes
    elif 'h' in duration:
        hours = int(duration.replace('h', '').strip())
        total_minutes = hours * 60
    elif 'm' in duration:
        total_minutes = int(duration.replace('m', '').strip())

    return total_minutes


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract datetime features from journey date and time columns.

    Args:
        df: DataFrame with Date_of_Journey, Dep_Time, and Arrival_Time columns

    Returns:
        DataFrame with extracted datetime features
    """
    df = df.copy()

    # Parse date of journey
    if 'Date_of_Journey' in df.columns:
        # Handle both DD/MM/YYYY and YYYY-MM-DD formats
        try:
            df['Journey_date'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
        except ValueError:
            df['Journey_date'] = pd.to_datetime(df['Date_of_Journey'])

        df['Journey_day'] = df['Journey_date'].dt.day
        df['Journey_month'] = df['Journey_date'].dt.month
        df['Journey_weekday'] = df['Journey_date'].dt.weekday
        df.drop('Journey_date', axis=1, inplace=True)

    # Parse departure time
    if 'Dep_Time' in df.columns:
        df['Dep_Time_parsed'] = pd.to_datetime(df['Dep_Time'], format='%H:%M', errors='coerce')
        df['Dep_hour'] = df['Dep_Time_parsed'].dt.hour
        df['Dep_min'] = df['Dep_Time_parsed'].dt.minute
        df.drop('Dep_Time_parsed', axis=1, inplace=True)

    # Parse arrival time
    if 'Arrival_Time' in df.columns:
        # Handle arrival time with potential date info
        arrival_time = df['Arrival_Time'].astype(str).str.split(' ').str[0]
        df['Arrival_Time_parsed'] = pd.to_datetime(arrival_time, format='%H:%M', errors='coerce')
        df['Arrival_hour'] = df['Arrival_Time_parsed'].dt.hour
        df['Arrival_min'] = df['Arrival_Time_parsed'].dt.minute
        df.drop('Arrival_Time_parsed', axis=1, inplace=True)

    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True) -> tuple:
    """Encode categorical columns using LabelEncoder.

    Args:
        df: DataFrame with categorical columns
        encoders: Dictionary of pre-fitted encoders (for inference)
        fit: Whether to fit new encoders (True for training, False for inference)

    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    df = df.copy()
    categorical_cols = ['Airline', 'Source', 'Destination']

    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            if fit:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders.get(col)
                if le is not None:
                    # Handle unseen labels by using -1
                    df[f'{col}_encoded'] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

    # Encode Total_Stops
    if 'Total_Stops' in df.columns:
        stops_mapping = {
            'non-stop': 0,
            '1 stop': 1,
            '2 stops': 2,
            '3 stops': 3,
            '4 stops': 4
        }
        df['Total_Stops_encoded'] = df['Total_Stops'].map(stops_mapping).fillna(0).astype(int)

    return df, encoders


def preprocess_training_data(df: pd.DataFrame) -> tuple:
    """Full preprocessing pipeline for training data.

    Args:
        df: Raw training DataFrame

    Returns:
        Tuple of (feature DataFrame, target Series, encoders dict)
    """
    df = df.copy()

    # Convert duration
    if 'Duration' in df.columns:
        df['Duration_minutes'] = df['Duration'].apply(convert_duration)

    # Extract datetime features
    df = extract_datetime_features(df)

    # Encode categoricals
    df, encoders = encode_categoricals(df, fit=True)

    # Define feature columns
    feature_cols = [
        'Journey_day', 'Journey_month', 'Journey_weekday',
        'Dep_hour', 'Dep_min', 'Arrival_hour', 'Arrival_min',
        'Duration_minutes',
        'Airline_encoded', 'Source_encoded', 'Destination_encoded',
        'Total_Stops_encoded'
    ]

    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df['Price'] if 'Price' in df.columns else None

    return X, y, encoders


def preprocess_inference_data(data: dict, encoders: dict) -> pd.DataFrame:
    """Preprocess single prediction request.

    Args:
        data: Dictionary with prediction request fields
        encoders: Pre-fitted encoders from training

    Returns:
        DataFrame ready for model prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([{
        'Airline': data.get('airline'),
        'Source': data.get('source'),
        'Destination': data.get('destination'),
        'Date_of_Journey': data.get('date_of_journey'),
        'Dep_Time': data.get('dep_time'),
        'Arrival_Time': data.get('arrival_time'),
        'Duration': data.get('duration'),
        'Total_Stops': data.get('total_stops')
    }])

    # Convert duration
    df['Duration_minutes'] = df['Duration'].apply(convert_duration)

    # Extract datetime features
    df = extract_datetime_features(df)

    # Encode categoricals using existing encoders
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)

    # Define feature columns in same order as training
    feature_cols = [
        'Journey_day', 'Journey_month', 'Journey_weekday',
        'Dep_hour', 'Dep_min', 'Arrival_hour', 'Arrival_min',
        'Duration_minutes',
        'Airline_encoded', 'Source_encoded', 'Destination_encoded',
        'Total_Stops_encoded'
    ]

    return df[feature_cols]
