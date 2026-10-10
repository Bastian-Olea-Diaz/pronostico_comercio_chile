"""Artefactos publicados cada mes y cómo leerlos.

Cada ejecución del pipeline publica una release de GitHub con estos archivos. La release funciona como
registro de modelos y datos: cada versión queda identificada, con sus modelos, pronósticos y reportes. La API
y el tablero leen la última release; el pipeline la descarga para partir desde el estado anterior.

Solo depende de pandas y requests, para que la API y el tablero no necesiten instalar el pipeline.
"""
import json
import os
from pathlib import Path

import pandas as pd
import requests

REPO = os.environ.get("PRONOSTICO_REPO", "Bastian-Olea-Diaz/pronostico_comercio_chile")

STATUS = "estado.json"
FORECASTS = "pronosticos.parquet"
FORECAST_HISTORY = "pronosticos_historial.parquet"
TOTALS = "totales.parquet"
PERFORMANCE = "desempeno.csv"
DRIFT = "drift.csv"
INGEST = "ingesta.csv"
LABELS = "etiquetas_hs6.csv"
COUNTRIES = "paises.csv"
REPORT = "reporte.md"
MODELS = "modelos.zip"
PANEL = "panel_{flow}.parquet"


def latest_release(repo: str = REPO, token: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token or os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token or os.environ['GITHUB_TOKEN']}"
    resp = requests.get(f"https://api.github.com/repos/{repo}/releases/latest", headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_release(dest: Path, repo: str = REPO, only: set[str] | None = None) -> str:
    """Descarga los archivos de la última release en `dest` y devuelve su etiqueta."""
    release = latest_release(repo)
    dest.mkdir(parents=True, exist_ok=True)
    for asset in release["assets"]:
        if only and asset["name"] not in only:
            continue
        with requests.get(asset["browser_download_url"], stream=True, timeout=300) as r:
            r.raise_for_status()
            tmp = dest / (asset["name"] + ".tmp")
            with tmp.open("wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
            tmp.replace(dest / asset["name"])
    return release["tag_name"]


def read_status(folder: Path) -> dict:
    return json.loads((folder / STATUS).read_text(encoding="utf-8"))


def read(folder: Path, name: str) -> pd.DataFrame:
    path = folder / name
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"hs6": str, "pais": str, "codigo": str})
