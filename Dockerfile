# OpenSports Production Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy source code
COPY opensports/ ./opensports/
COPY README.md LICENSE ./

# Install the package
RUN pip install -e .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    OPENSPORTS_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r opensports \
    && useradd -r -g opensports opensports

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder opensports/ /app/opensports/
COPY --from=builder README.md LICENSE /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/exports && \
    chown -R opensports:opensports /app

# Switch to non-root user
USER opensports
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "opensports.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"] 