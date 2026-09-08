# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies if needed (for numpy/faiss etc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the source code
COPY . /app

# Create directory for persistent data
RUN mkdir -p /app/vsearch_data && chown -R 1000:1000 /app/vsearch_data
VOLUME ["/app/vsearch_data"]

# Set environment variables
ENV PYTHONPATH="/app"
ENV VSEARCH_PERSIST_DIR="/app/vsearch_data"
ENV VSEARCH_SNAPSHOT_INTERVAL="300"

EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "src.vsearch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
