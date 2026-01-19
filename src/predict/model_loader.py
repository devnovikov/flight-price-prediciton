"""Model loading utility for Flight Price Prediction API.

Handles loading the trained model from disk on application startup.
"""

import os
import joblib
from pathlib import Path
from typing import Optional, Any

# Default model path
DEFAULT_MODEL_PATH = "models/best_model.joblib"


class ModelLoader:
    """Singleton class for loading and managing the ML model."""

    _instance: Optional['ModelLoader'] = None
    _model: Optional[Any] = None
    _model_artifact: Optional[dict] = None
    _is_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_path: str = DEFAULT_MODEL_PATH) -> bool:
        """Load the model from disk.

        Args:
            model_path: Path to the model file

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False

            print(f"Loading model from {model_path}...")
            self._model_artifact = joblib.load(model_path)
            self._model = self._model_artifact.get('model')

            if self._model is None:
                print("Model artifact does not contain 'model' key")
                return False

            self._is_loaded = True
            print(f"Model loaded successfully: {self._model_artifact.get('model_name', 'Unknown')}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self._is_loaded = False
            return False

    @property
    def model(self) -> Any:
        """Get the loaded model."""
        return self._model

    @property
    def model_artifact(self) -> Optional[dict]:
        """Get the full model artifact including metadata."""
        return self._model_artifact

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Get the model name."""
        if self._model_artifact:
            return self._model_artifact.get('model_name', 'Unknown')
        return 'Unknown'

    @property
    def feature_columns(self) -> list:
        """Get the feature columns used by the model."""
        if self._model_artifact:
            return self._model_artifact.get('feature_columns', [])
        return []

    @property
    def encoders(self) -> dict:
        """Get the label encoders."""
        if self._model_artifact:
            return self._model_artifact.get('encoders', {})
        return {}

    @property
    def metrics(self) -> dict:
        """Get the training metrics."""
        if self._model_artifact:
            return self._model_artifact.get('metrics', {})
        return {}


# Global model loader instance
model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return model_loader
