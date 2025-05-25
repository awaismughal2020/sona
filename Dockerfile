# © 2025 Awais Mughal. All rights reserved.
# Unauthorized commercial use is prohibited.

# Multi-stage Dockerfile for SONA AI Assistant

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build dependencies
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # Audio processing dependencies
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    libsndfile1-dev \
    # Media processing
    ffmpeg \
    # Network tools
    curl \
    wget \
    # Git for potential package installations
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r sona && useradd -r -g sona -m -d /app -s /bin/bash sona

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories and __init__.py files for proper Python package structure
RUN mkdir -p config utils ai ai/speech_to_text ai/intent_detection ai/image_generation ai/web_search backend backend/middleware ui ui/components tests && \
    touch config/__init__.py && \
    touch utils/__init__.py && \
    touch ai/__init__.py && \
    touch ai/speech_to_text/__init__.py && \
    touch ai/intent_detection/__init__.py && \
    touch ai/image_generation/__init__.py && \
    touch ai/web_search/__init__.py && \
    touch backend/__init__.py && \
    touch backend/middleware/__init__.py && \
    touch ui/__init__.py && \
    touch ui/components/__init__.py && \
    touch tests/__init__.py

# Create necessary directories
RUN mkdir -p logs temp && \
    chmod 755 logs temp

# Set ownership to sona user
RUN chown -R sona:sona /app

# Switch to non-root user
USER sona

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py", "--mode", "dev"]

