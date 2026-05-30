FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8080

# Environment
ENV MODEL_PATH=/app/scoring/model.pkl
ENV THRESHOLD_PATH=/app/training/artifacts/threshold.pkl
ENV ENCODER_PATH=/app/feature_engineering/artifacts/encoder.pkl
ENV SCALER_PATH=/app/feature_engineering/artifacts/scaler.pkl
ENV PYTHONPATH=/app

# Run scoring API
CMD ["python", "scoring/score.py"]