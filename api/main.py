"""API de pronósticos del comercio exterior de Chile.

Sirve los pronósticos que el pipeline calcula una vez al mes (pronóstico por lotes): los datos cambian
mensualmente, así que calcular en cada consulta solo agregaría latencia y dependencias. Lee los artefactos
de una carpeta local (ARTIFACTS_DIR) o de la última release de GitHub, y los vuelve a descargar cada
REFRESH_HOURS horas para tomar el mes nuevo sin reiniciar el servicio.
"""
import os
import threading
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from pronostico import artifacts as A

LOCAL = os.environ.get("ARTIFACTS_DIR")
CACHE = Path(os.environ.get("CACHE_DIR", "/tmp/artefactos"))
REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", 6))
FILES = {A.STATUS, A.FORECASTS, A.TOTALS, A.PERFORMANCE, A.LABELS, A.COUNTRIES}
Flow = Literal["exportaciones", "importaciones"]

app = FastAPI(
    title="Pronóstico del comercio exterior de Chile",
    description="Pronósticos mensuales a 1, 2 y 3 meses por producto (HS6) y país, con su monitoreo. "
                "Datos: Servicio Nacional de Aduanas (datos.gob.cl).",
    version="1.0",
)


class Store:
    """Artefactos en memoria, recargados cuando pasan REFRESH_HOURS desde la última carga."""

    def __init__(self):
        self.loaded_at, self.lock, self.data = 0.0, threading.Lock(), None

    def get(self) -> dict:
        with self.lock:
            if self.data is None or (not LOCAL and time.time() - self.loaded_at > REFRESH_HOURS * 3600):
                self.data = self._load()
                self.loaded_at = time.time()
            return self.data

    @staticmethod
    def _load() -> dict:
        folder = Path(LOCAL) if LOCAL else CACHE
        release = "local"
        if not LOCAL:
            release = A.download_release(folder, only=FILES)
        labels = A.read(folder, A.LABELS).set_index("hs6")["etiqueta"]
        countries = A.read(folder, A.COUNTRIES).set_index("codigo")["glosa"]
        return {"release": release, "status": A.read_status(folder), "forecasts": A.read(folder, A.FORECASTS),
                "totals": A.read(folder, A.TOTALS), "performance": A.read(folder, A.PERFORMANCE),
                "labels": labels, "countries": countries}


store = Store()


def _records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/", summary="Descripción del servicio")
def root():
    s = store.get()
    return {"servicio": app.title, "datos_hasta": s["status"]["ultimo_mes"], "release": s["release"],
            "documentacion": "/docs",
            "rutas": ["/salud", "/estado", "/pronosticos/totales", "/pronosticos", "/productos", "/monitoreo/desempeno"]}


@app.get("/salud", summary="Estado del servicio y antigüedad de los datos")
def health():
    s = store.get()
    return {"estado": "ok", "datos_hasta": s["status"]["ultimo_mes"], "actualizado": s["status"]["actualizado"],
            "alertas": len(s["status"]["alertas"])}


@app.get("/estado", summary="Versión de los modelos, calidad de datos, desempeño y alertas del último mes")
def status():
    return store.get()["status"]


@app.get("/pronosticos/totales", summary="Total pronosticado por mes (series regulares)")
def totals(flujo: Flow | None = None):
    df = store.get()["totals"]
    return _records(df[df.flujo == flujo] if flujo else df)


@app.get("/pronosticos", summary="Pronóstico de una serie producto–país")
def forecast(flujo: Flow, hs6: str = Query(pattern=r"^\d{6}$"), pais: str = Query(pattern=r"^\d{1,3}$")):
    s = store.get()
    fc = s["forecasts"]
    sel = fc[(fc.flujo == flujo) & (fc.hs6 == hs6) & (fc.pais == pais)]
    if sel.empty:
        raise HTTPException(404, "Sin pronóstico para esa serie: solo se pronostican las series regulares "
                                 "(comercio en al menos 10 de los últimos 12 meses).")
    return {"flujo": flujo, "hs6": hs6, "producto": s["labels"].get(hs6), "pais": pais,
            "pais_nombre": s["countries"].get(pais),
            "pronosticos": _records(sel[["origen", "h", "objetivo", "pronostico", "pronostico_media", "media_movil_3"]])}


@app.get("/productos", summary="Busca productos con pronóstico por nombre o código")
def products(buscar: str = Query(min_length=2), flujo: Flow = "exportaciones", limite: int = Query(20, le=100)):
    s = store.get()
    available = s["forecasts"].loc[s["forecasts"].flujo == flujo, "hs6"].unique()
    labels = s["labels"].reindex(available).fillna("")
    hit = labels[labels.str.contains(buscar, case=False, regex=False) | labels.index.str.startswith(buscar)]
    return [{"hs6": h, "producto": p} for h, p in hit.head(limite).items()]


@app.get("/monitoreo/desempeno", summary="Error del modelo y de la media móvil, mes a mes")
def performance(flujo: Flow | None = None, h: int = Query(1, ge=1, le=3)):
    df = store.get()["performance"]
    df = df[(df.h == h) & ((df.flujo == flujo) if flujo else True)]
    return _records(df)
