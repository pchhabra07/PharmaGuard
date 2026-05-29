# ==============================================================
# PharmaGuard — Dockerfile (Multi-Stage Build)
# ==============================================================
# Builds a lightweight Docker image for the FastAPI prediction API.
#
# Stage 1 (builder): installs Python dependencies into a venv.
# Stage 2 (runtime): copies only the venv + application code,
#   resulting in a smaller final image (~400 MB vs ~1.2 GB).
#
# Usage:
#   docker build -f docker/Dockerfile -t pharmaguard-api .
#   docker run -p 8000:8000 pharmaguard-api
#
# The build context should be the project root (.).
# ==============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies needed for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code and configuration
COPY config.yaml .
COPY run_api.py .
COPY src/ src/

# Copy model artifacts (preprocessor + trained model)
COPY model_artifacts/ model_artifacts/

# Copy metrics history for the dashboard
COPY metrics/ metrics/

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
