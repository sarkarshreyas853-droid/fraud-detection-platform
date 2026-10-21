# Fraud scoring API. Build:  docker build -t fraud-api:0.1 .
# The image bakes in the model and a snapshot of the Feast online store, so run
# "python scripts/prepare_feast.py" BEFORE building (it creates feature_repo/data/*.db).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY config.py .
COPY app/ app/
COPY feature_repo/ feature_repo/
COPY model/model.json model/model_card.json model/
RUN mkdir -p data && useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
