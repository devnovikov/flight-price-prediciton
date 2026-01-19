"""Pydantic schemas for Flight Price Prediction API.

These schemas define the request and response models for the API endpoints,
matching the OpenAPI contract specification.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class PredictionRequest(BaseModel):
    """Request model for flight price prediction.

    All fields are required and must match the training data format.
    """
    airline: str = Field(
        ...,
        description="Name of the airline carrier",
        examples=["IndiGo", "Air India", "Jet Airways"]
    )
    source: str = Field(
        ...,
        description="Departure city",
        examples=["Delhi", "Mumbai", "Bangalore"]
    )
    destination: str = Field(
        ...,
        description="Arrival city (must be different from source)",
        examples=["Cochin", "Hyderabad", "Kolkata"]
    )
    date_of_journey: str = Field(
        ...,
        description="Date of travel in YYYY-MM-DD format",
        examples=["2026-03-15"]
    )
    dep_time: str = Field(
        ...,
        description="Departure time in HH:MM format",
        examples=["10:30", "14:45"]
    )
    arrival_time: str = Field(
        ...,
        description="Arrival time in HH:MM format",
        examples=["14:45", "18:30"]
    )
    duration: str = Field(
        ...,
        description="Flight duration (e.g., '4h 15m', '2h', '45m')",
        examples=["4h 15m", "2h 30m"]
    )
    total_stops: str = Field(
        ...,
        description="Number of stops",
        examples=["non-stop", "1 stop", "2 stops"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "airline": "IndiGo",
                    "source": "Delhi",
                    "destination": "Cochin",
                    "date_of_journey": "2026-03-15",
                    "dep_time": "10:30",
                    "arrival_time": "14:45",
                    "duration": "4h 15m",
                    "total_stops": "1 stop"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for flight price prediction."""
    predicted_price: float = Field(
        ...,
        description="Predicted ticket price",
        examples=[8542.75]
    )
    currency: str = Field(
        default="INR",
        description="Currency of the predicted price"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Version of the ML model used for prediction"
    )
    features_used: List[str] = Field(
        ...,
        description="List of engineered features used by the model"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(
        ...,
        description="Current health status of the service",
        examples=["healthy", "unhealthy"]
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready for predictions"
    )
    version: str = Field(
        default="1.0.0",
        description="API version"
    )


class ErrorDetail(BaseModel):
    """Error detail model."""
    loc: List[str] = Field(..., description="Location of the error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    detail: List[ErrorDetail] = Field(..., description="List of validation errors")
