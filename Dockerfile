# ── Stage: runtime ────────────────────────────────────────────────────────────
# This Dockerfile is for deploying the FastAPI prediction service as a
# standalone cloud node. It is independent of the local pipeline.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs at runtime
COPY model/ ./model/
COPY api/ ./api/

EXPOSE 8000

# Run with a single worker; scale horizontally via your cloud provider
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
