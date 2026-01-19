# Flight Price Prediction API Dockerfile
# Uses Python 3.13-slim with uv for dependency management

FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files and README (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (production only, no dev dependencies, skip project install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/

# Copy trained model (must exist before building)
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the API server
CMD ["uv", "run", "uvicorn", "src.predict.predict:app", "--host", "0.0.0.0", "--port", "8000"]
