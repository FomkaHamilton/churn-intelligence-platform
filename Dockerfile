# ─────────────────────────────────────────────────────────────────────────────
# Churn Intelligence Platform — Dockerfile
# Multi-stage build: keeps the runtime image lean.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps for packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# curl is required by the HEALTHCHECK command
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create data directories (volumes will be mounted here)
RUN mkdir -p data/raw data/processed data/sessions data/sample

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
