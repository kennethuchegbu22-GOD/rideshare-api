FROM python:3.11-slim

WORKDIR /app

# System deps for scientific python wheels build faster / more reliably
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY artifacts/ ./artifacts/

# Fails fast at build time if artifacts weren't generated/copied in
RUN test -f artifacts/model.joblib || \
    (echo "ERROR: artifacts/model.joblib is missing. Run train.py and copy artifacts/ in before building." && exit 1)

EXPOSE 5000

# 2 workers is a reasonable default for a small model; tune to your CPU/traffic
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app:app"]
