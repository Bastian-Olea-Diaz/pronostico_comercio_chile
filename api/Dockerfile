# Imagen de la API. Solo lleva lo necesario para servir: la API y el lector de artefactos, sin el pipeline
# ni LightGBM. Construir desde la raíz del repositorio:
#   docker build -f api/Dockerfile -t pronostico-api .
#   docker run --rm -p 8000:8000 pronostico-api
FROM python:3.12-slim

# Sin privilegios de root dentro del contenedor
RUN useradd -m -u 1000 app
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY src/pronostico/__init__.py src/pronostico/artifacts.py src/pronostico/
COPY api/__init__.py api/main.py api/

# Render define PORT al ejecutar el contenedor; 8000 es el valor para correrla localmente
ENV PYTHONPATH=/app/src \
    CACHE_DIR=/tmp/artefactos \
    PORT=8000
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
