FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    scikit-learn==1.6.1 \
    numpy==1.26.4 \
    scipy==1.14.1 \
    joblib==1.4.2 \
    requests==2.32.3 \
    pydantic==2.9.2 \
    google-cloud-storage==2.14.0 \
    google-cloud-secret-manager==2.20.0

# Copy source code
COPY src/ /app/src/

# Copy model artifacts (for serving mode)
COPY reports/training/model_final/ /app/model/

# Copy serving entrypoint
COPY src/serving/train.py /app/entrypoint.py

# Environment
ENV MODEL_DIR=/app/model
ENV PORT=8080
ENV PYTHONPATH=/app

EXPOSE 8080

# Default: serve mode. Override with --mode train for training jobs.
ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["--mode", "serve"]
