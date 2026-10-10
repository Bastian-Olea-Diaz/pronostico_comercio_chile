"""Pronóstico mensual del comercio exterior de Chile en producción.

Toma el modelo del proyecto de análisis (paquete comercio_chile) y lo hace operar solo: cada mes detecta el
archivo nuevo de Aduanas, valida su calidad, evalúa el pronóstico anterior contra lo que realmente pasó,
monitorea drift, reentrena y publica una versión nueva de modelos y pronósticos.

Este archivo no importa nada pesado: la API y el tablero usan pronostico.artifacts sin instalar el pipeline.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# El paquete comercio_chile lee esta variable al importarse: sus descargas y archivos intermedios quedan en
# work/ de este repositorio, no en el repositorio de análisis.
os.environ.setdefault("COMERCIO_CHILE_DATA", str(ROOT / "work"))
