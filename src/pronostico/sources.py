"""Detección y descarga de meses nuevos en datos.gob.cl, usando el paquete del proyecto de análisis.

Aduanas publica cada mes cerca del día 30 del mes siguiente. Un mes se considera disponible cuando están
publicados los archivos de exportaciones y de importaciones.
"""
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from comercio_chile import clean, download, ingest

FLOWS = ("exportaciones", "importaciones")


def is_published(month: str) -> bool:
    p = pd.Period(month, "M")
    return all(any(r["month"] == p.month for r in download.list_resources(flow, p.year)) for flow in FLOWS)


def fetch_month(flow: str, month: str) -> tuple[pd.DataFrame, dict]:
    """Descarga, repara e ingiere el archivo del mes y devuelve (panel del mes, estadísticas de ingesta)."""
    p = pd.Period(month, "M")
    parts = [r for r in download.list_resources(flow, p.year) if r["month"] == p.month]
    if not parts:
        raise FileNotFoundError(f"{flow} {month}: no está publicado")
    paths = [download.download(r) for r in parts]   # un mes puede venir en varios volúmenes (.part01, .part02)
    stats = ingest.process(flow, p.year, p.month, paths[0], force=True)
    panel, _ = clean.build_panel(flow, [month])
    return panel, asdict(stats)


def local_fetch(source_data: Path):
    """Variante para pruebas locales: arma el panel del mes desde archivos ya ingeridos en otra carpeta
    (por ejemplo, la del proyecto de análisis), sin descargar nada."""
    def fetch(flow: str, month: str) -> tuple[pd.DataFrame, dict]:
        log = pd.read_csv(source_data / "interim" / "_ingest_log.csv")
        p = pd.Period(month, "M")
        stats = log[(log.kind == flow) & (log.year == p.year) & (log.month == p.month)].iloc[0].to_dict()
        panel = pd.read_parquet(source_data / "processed" / f"panel_{flow}.parquet")
        return panel[panel.periodo == p].reset_index(drop=True), stats
    return fetch
