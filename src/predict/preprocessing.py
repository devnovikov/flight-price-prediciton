"""Preprocessing utilities for inference.

Applies the same transformations as training to ensure consistency.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


def convert_duration(duration: str) -> int:
    """Convert duration string to minutes.

    Args:
        duration: Duration string in format "Xh Ym", "Xh", or "Ym"

    Returns:
        Total duration in minutes
    """
    if not duration or pd.isna(duration):
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


def preprocess_request(data: Dict[str, Any], encoders: Dict, feature_columns: list) -> pd.DataFrame:
    """Preprocess a prediction request into model-ready features.

    Args:
        data: Dictionary containing the request fields
        encoders: LabelEncoders from training
        feature_columns: List of feature column names in correct order

    Returns:
        DataFrame with preprocessed features
    """
    # Create a dictionary for features
    features = {}

    # Parse date of journey
    date_str = data.get('date_of_journey', '')
    try:
        if date_str:
            # Handle both YYYY-MM-DD and DD/MM/YYYY formats
            try:
                journey_date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                journey_date = datetime.strptime(date_str, '%d/%m/%Y')

            features['Journey_day'] = journey_date.day
            features['Journey_month'] = journey_date.month
            features['Journey_weekday'] = journey_date.weekday()
    except:
        features['Journey_day'] = 1
        features['Journey_month'] = 1
        features['Journey_weekday'] = 0

    # Parse departure time
    dep_time = data.get('dep_time', '00:00')
    try:
        dep_dt = datetime.strptime(dep_time, '%H:%M')
        features['Dep_hour'] = dep_dt.hour
        features['Dep_min'] = dep_dt.minute
    except:
        features['Dep_hour'] = 0
        features['Dep_min'] = 0

    # Parse arrival time
    arrival_time = data.get('arrival_time', '00:00')
    try:
        # Handle arrival time with potential date info
        arrival_clean = arrival_time.split(' ')[0] if ' ' in arrival_time else arrival_time
        arr_dt = datetime.strptime(arrival_clean, '%H:%M')
        features['Arrival_hour'] = arr_dt.hour
        features['Arrival_min'] = arr_dt.minute
    except:
        features['Arrival_hour'] = 0
        features['Arrival_min'] = 0

    # Convert duration
    features['Duration_minutes'] = convert_duration(data.get('duration', '0h 0m'))

    # Encode categorical features
    airline = data.get('airline', 'Unknown')
    source = data.get('source', 'Unknown')
    destination = data.get('destination', 'Unknown')
    total_stops = data.get('total_stops', 'non-stop')

    # Airline encoding
    if 'airline' in encoders:
        le = encoders['airline']
        if airline in le.classes_:
            features['Airline_encoded'] = le.transform([airline])[0]
        else:
            # Use most common class or 0 for unknown
            features['Airline_encoded'] = 0
    else:
        features['Airline_encoded'] = 0

    # Source encoding
    if 'source_city' in encoders:
        le = encoders['source_city']
        if source in le.classes_:
            features['Source_encoded'] = le.transform([source])[0]
        else:
            features['Source_encoded'] = 0
    else:
        features['Source_encoded'] = 0

    # Destination encoding
    if 'destination_city' in encoders:
        le = encoders['destination_city']
        if destination in le.classes_:
            features['Destination_encoded'] = le.transform([destination])[0]
        else:
            features['Destination_encoded'] = 0
    else:
        features['Destination_encoded'] = 0

    # Stops encoding
    stops_mapping = {
        'non-stop': 0, 'zero': 0,
        '1 stop': 1, 'one': 1,
        '2 stops': 2, 'two': 2,
        '3 stops': 3, 'three': 3,
        '4 stops': 4, 'four': 4
    }
    features['Total_Stops_encoded'] = stops_mapping.get(total_stops.lower(), 0)

    # Create DataFrame with feature columns
    # Fill missing columns with 0
    df_features = {}
    for col in feature_columns:
        if col in features:
            df_features[col] = features[col]
        else:
            # Try to find a matching column name (case-insensitive)
            matched = False
            for key, val in features.items():
                if key.lower() == col.lower():
                    df_features[col] = val
                    matched = True
                    break
            if not matched:
                df_features[col] = 0

    return pd.DataFrame([df_features])
