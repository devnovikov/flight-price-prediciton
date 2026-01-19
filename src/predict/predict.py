"""FastAPI application for Flight Price Prediction.

This module provides the REST API for predicting flight ticket prices
using the trained ML model.

Usage:
    uvicorn src.predict.predict:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)
from .model_loader import get_model_loader, DEFAULT_MODEL_PATH
from .preprocessing import preprocess_request


# API metadata
API_VERSION = "1.0.0"
API_TITLE = "Flight Price Prediction API"
API_DESCRIPTION = """
REST API for predicting flight ticket prices using machine learning.

This API serves predictions from a trained ML model that achieves
R2 > 0.85 on the Kaggle Flight Price Prediction dataset.

## Endpoints

- **GET /**: Health check endpoint
- **POST /predict**: Predict flight price

## Features Used

The model uses engineered features including:
- Datetime extraction (day, month, weekday, hour, minute)
- Duration conversion to minutes
- Categorical encoding for airline, source, destination, and stops
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - load model on startup."""
    # Startup
    loader = get_model_loader()
    success = loader.load(DEFAULT_MODEL_PATH)
    if success:
        print(f"Model loaded: {loader.model_name}")
        print(f"Metrics: {loader.metrics}")
    else:
        print("WARNING: Model not loaded. Predictions will fail.")
    yield
    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns the health status of the API and whether the model is loaded.
    """
    loader = get_model_loader()
    return HealthResponse(
        status="healthy" if loader.is_loaded else "unhealthy",
        model_loaded=loader.is_loaded,
        version=API_VERSION
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(request: PredictionRequest) -> PredictionResponse:
    """Predict flight price.

    Predicts the price of a flight ticket based on the provided flight details.

    The model uses feature engineering including:
    - Datetime extraction (day, month, weekday, hour, minute)
    - Duration conversion to minutes
    - Categorical encoding for airline, source, destination, and stops
    """
    loader = get_model_loader()

    # Check if model is loaded
    if not loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists at models/best_model.joblib"
        )

    try:
        # Convert request to dict
        request_data = {
            'airline': request.airline,
            'source': request.source,
            'destination': request.destination,
            'date_of_journey': request.date_of_journey,
            'dep_time': request.dep_time,
            'arrival_time': request.arrival_time,
            'duration': request.duration,
            'total_stops': request.total_stops
        }

        # Preprocess the request
        features_df = preprocess_request(
            request_data,
            loader.encoders,
            loader.feature_columns
        )

        # Make prediction
        prediction = loader.model.predict(features_df)[0]

        # Ensure prediction is positive
        prediction = max(0, float(prediction))

        return PredictionResponse(
            predicted_price=round(prediction, 2),
            currency="INR",
            model_version=API_VERSION,
            features_used=loader.feature_columns
        )

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input value: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# For running with `python -m src.predict.predict`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
