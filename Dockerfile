# Cats vs Dogs CNN classifier - web API image
# Runs the FastAPI service that serves the trained PyTorch model.
FROM python:3.12-slim

WORKDIR /app

# Python prints straight to logs (no buffering issues in containers)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (config is required by src)
COPY config.py .
COPY src/ ./src/
COPY api/ ./api/

# The trained model must be mounted or baked into the image.
# By default the image only ships code; mount models/ to provide weights:
#   docker run -v $(pwd)/models:/app/models -p 8000:8000 cats-dogs-cnn
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]