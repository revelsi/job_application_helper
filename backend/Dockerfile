# Job Application Helper - Backend Dockerfile
# Optimized for Python 3.9+, FastAPI, and document processing

FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY start_api.py .
# Copy env.example as .env for Docker (will be overridden by docker-compose environment variables)
COPY env.example .env

# Create data directories
RUN mkdir -p data/{documents,cache} && \
    mkdir -p /home/appuser/.cache/torch/sentence_transformers && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Start the application
CMD ["python", "start_api.py"] 