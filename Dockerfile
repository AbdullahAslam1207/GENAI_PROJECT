# Multi-stage build for RAG Chatbot deployment to EC2
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create necessary directories
RUN mkdir -p logs data/documents embeddings/cache indexes

# Copy application code
COPY main.py .
COPY quickstart.py .
COPY rag_pipeline.py .
COPY config.yaml .
COPY data/ ./data/
COPY embeddings/ ./embeddings/
COPY evaluation/ ./evaluation/
COPY examples/ ./examples/
COPY generation/ ./generation/
COPY indexing/ ./indexing/
COPY retrieval/ ./retrieval/

# Create a non-root user for security
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# Expose port for potential API/web interface (optional)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command - Build index on startup, then start interactive mode
CMD ["python", "main.py", "--config", "config.yaml", "build-index", "--documents-dir", "data/documents"]

# Alternative: Keep container running for manual interaction
# CMD ["tail", "-f", "/dev/null"]
