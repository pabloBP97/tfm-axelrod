# API de inferencia y demo del TFM (dilema del prisionero iterado).
#   docker build -t tfm-axelrod-api .
#   docker run -p 8000:8000 tfm-axelrod-api      -> http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TFM_ROOT=/app

WORKDIR /app

# torch (lo importa axelrod) en su variante solo-CPU: unos cientos de MB en vez de varios GB.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

# Solo lo necesario para servir: código y modelos pequeños (sin data/ ni mlruns/).
COPY tfm_axelrod/ tfm_axelrod/
COPY models/classifier_serving/ models/classifier_serving/
COPY models/qlearning/ models/qlearning/
COPY models/head_to_head.json models/head_to_head.json

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "tfm_axelrod.api:app", "--host", "0.0.0.0", "--port", "8000"]
