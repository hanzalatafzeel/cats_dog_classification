# Cats vs Dogs CNN classifier - web API image
# Runs the FastAPI service that serves the trained PyTorch model.
# Uses CPU-only torch wheels so the image stays small enough for free tiers
# (Render free = 512 MB RAM / 0.1 CPU, 500 build-min/month).
FROM python:3.12-slim

WORKDIR /app

# Python prints straight to logs (no buffering issues in containers)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

# 1) CPU builds of torch/torchvision FIRST (PyPI defaults to CUDA bundles).
#    Pip skips re-installing them when -r requirements.txt is processed.
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Application code (config is required by src)
COPY config.py .
COPY src/ ./src/
COPY api/ ./api/

# 4) Runtime assets baked into the image (self-contained deploy):
#    - models/   trained checkpoint
#    - outputs/  committed dashboard data + diagnostic plots
#    - test_images/ quick-try samples used by the dashboard
COPY models/ ./models/
COPY outputs/ ./outputs/
COPY test_images/ ./test_images/

EXPOSE 8000

# Render injects $PORT (default 10000); fall back to 8000 locally.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]