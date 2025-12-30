# JengaHub Production Dockerfile
# Multi-stage build for optimized production deployment

# =============================================================================
# Stage 1: Base Dependencies
# =============================================================================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Python Dependencies
# =============================================================================
FROM base AS python-deps

# Install poetry for dependency management
RUN pip install poetry==1.6.1

# Configure poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main --no-root && rm -rf $POETRY_CACHE_DIR

# =============================================================================
# Stage 3: Model Optimization Dependencies (Optional)
# =============================================================================
FROM python-deps AS optimization-deps

# Install ONNX Runtime GPU
RUN pip install onnxruntime-gpu==1.15.1

# Install TensorRT (optional, for NVIDIA optimization)
RUN pip install nvidia-tensorrt==8.6.1

# Install optimization tools
RUN pip install \
    onnx==1.14.1 \
    onnxsim==0.4.33 \
    torch-pruning==1.3.0

# =============================================================================
# Stage 4: Application Code
# =============================================================================
FROM optimization-deps AS app-code

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Create non-root user for security
RUN groupadd -r jengahub && useradd -r -g jengahub jengahub

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache \
    && chown -R jengahub:jengahub /app

# Copy application code
COPY --chown=jengahub:jengahub jengahub/ ./jengahub/
COPY --chown=jengahub:jengahub examples/ ./examples/
COPY --chown=jengahub:jengahub scripts/ ./scripts/
COPY --chown=jengahub:jengahub configs/ ./configs/

# Install the package
RUN poetry install --only=main

# =============================================================================
# Stage 5: Production Image
# =============================================================================
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 AS production

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Create non-root user
RUN groupadd -r jengahub && useradd -r -g jengahub jengahub

# Set working directory
WORKDIR /app

# Copy virtual environment from deps stage
COPY --from=app-code /app/.venv /app/.venv

# Copy application code
COPY --from=app-code --chown=jengahub:jengahub /app/jengahub ./jengahub/
COPY --from=app-code --chown=jengahub:jengahub /app/scripts ./scripts/
COPY --from=app-code --chown=jengahub:jengahub /app/configs ./configs/

# Create runtime directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache /app/tmp \
    && chown -R jengahub:jengahub /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    JENGAHUB_HOME=/app \
    JENGAHUB_DATA_DIR=/app/data \
    JENGAHUB_MODEL_DIR=/app/models \
    JENGAHUB_CACHE_DIR=/app/cache \
    JENGAHUB_LOG_DIR=/app/logs

# Health check script
COPY --chown=jengahub:jengahub docker/healthcheck.py ./healthcheck.py
RUN chmod +x ./healthcheck.py

# Switch to non-root user
USER jengahub

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python healthcheck.py || exit 1

# Default command
CMD ["python", "-m", "jengahub.api.server", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Labels for metadata
# =============================================================================
LABEL maintainer="JengaHub Team"
LABEL version="1.0.0"
LABEL description="JengaHub - Unified Multimodal AI Framework for African Languages"
LABEL org.opencontainers.image.source="https://github.com/Jenga-AI/Jenga-AI"
LABEL org.opencontainers.image.documentation="https://jengahub.ai/docs"
LABEL org.opencontainers.image.vendor="JengaHub"
LABEL org.opencontainers.image.licenses="Apache-2.0"