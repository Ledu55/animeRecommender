FROM python:3.12-slim

# LightGBM's wheel links against OpenMP at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer
COPY pyproject.toml README.md ./
COPY animerec ./animerec
RUN pip install --no-cache-dir .

# Recommendations produced by `animerec update`
COPY data ./data

ENV PORT=8000 DATA_PATH=/app/data/recommendations.json
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} animerec.web:app"]
