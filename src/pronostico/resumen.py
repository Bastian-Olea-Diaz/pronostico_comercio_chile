"""resumen.json: lo que muestra el portafolio, en un archivo chico y estable.

Uso:
    python -m pronostico.resumen --estado state --salida site

El portafolio (bastian-olea-diaz.github.io) lo lee desde GitHub Pages de este repositorio, que comparte su
dominio: el navegador no necesita permisos extra (CORS) y no hay servidor que mantener. El esquema tiene
versión: un cambio incompatible sube ESQUEMA y el portafolio puede detectarlo en vez de dibujar mal.

Todos los montos van en US$. Los totales son de las series regulares (las que el modelo pronostica).

Campos agregados sin cambiar el esquema (el portafolio ignora lo que no conoce): `inferior` y `superior` en
cada pronóstico (intervalo conformal del 80%), `intervalo` por flujo, `dentro` en cada acierto y `cobertura`
en el resumen. `error_tipico` se mantiene por compatibilidad.
"""
import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts as A

ESQUEMA = 1
HISTORY_MONTHS = 24
BAND_MONTHS = 12
TOP_PRODUCTS = 5
FLOWS = ("exportaciones", "importaciones")


def _title(text: str) -> str:
    """Los nombres nuevos ya vienen en formato de oración ("Cátodo de cobre"); solo se transforman los que
    llegan en mayúsculas (releases anteriores)."""
    text = str(text)
    if not text.isupper():
        return text
    return re.sub(r" (De|Del|La|Las|Los|Y|E|En|Con|Para|Sin) ", lambda m: m.group(0).lower(), text.title())


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v))


def _coverage(live: pd.DataFrame) -> dict:
    """Cuántos totales publicados (todos los horizontes) cayeron dentro de su intervalo, en vivo."""
    if "dentro_intervalo" not in live:
        return {"evaluados": 0, "dentro": 0}
    known = live.dentro_intervalo.dropna().astype(bool)
    return {"evaluados": int(len(known)), "dentro": int(known.sum())}


def _flow_summary(state: Path, flow: str, fc: pd.DataFrame, totals: pd.DataFrame, perf: pd.DataFrame,
                  labels: pd.Series) -> dict:
    fc = fc[fc.flujo == flow]
    panel = pd.read_parquet(state / A.PANEL.format(flow=flow))
    panel = panel[(panel.segmento == "bienes") & ~panel.pseudo_pais & (panel.hs6 != "000000")]
    panel = panel.merge(fc[["hs6", "pais"]].drop_duplicates())          # mismas series que el pronóstico
    monthly = panel.groupby(panel.periodo.astype(str)).valor_usd.sum().sort_index()

    tot = totals[totals.flujo == flow].sort_values("h")
    p1 = perf[(perf.flujo == flow) & (perf.h == 1)].sort_values("periodo")
    live = p1[p1.fuente == "producción"]
    band_rows = p1.dropna(subset=["total_pronosticado"]).tail(BAND_MONTHS)
    band = float((band_rows.total_pronosticado / band_rows.total_observado - 1).abs().mean()) if len(band_rows) else None

    # productos con mayor pronóstico a 1 mes, comparados con su promedio de los últimos 3 meses observados
    h1 = fc[fc.h == 1].groupby("hs6").pronostico_media.sum().nlargest(TOP_PRODUCTS)
    last3 = sorted(panel.periodo.astype(str).unique())[-3:]
    recent = panel[panel.periodo.astype(str).isin(last3)].groupby("hs6").valor_usd.sum() / 3

    return {
        "historia": [{"mes": m, "valor": round(float(v))} for m, v in monthly.tail(HISTORY_MONTHS).items()],
        "pronostico": [{"mes": r.objetivo, "h": int(r.h), "valor": round(float(r.total)),
                        "media_movil": round(float(r.total_media_movil_3)),
                        "inferior": _num(getattr(r, "inferior", None)), "superior": _num(getattr(r, "superior", None))}
                       for r in tot.itertuples()],
        "intervalo": ({"nivel": float(tot.nivel.iloc[0]), "metodo": "conformal",
                       "meses_calibracion": int(tot.calibracion_meses.min())}
                      if "nivel" in tot and tot.inferior.notna().any() else None),
        "error_tipico": None if band is None or np.isnan(band) else round(band, 4),
        "aciertos": [{"mes": r.periodo, "pronosticado": round(float(r.total_pronosticado)), "observado": round(float(r.total_observado)),
                      "error_modelo": round(float(r.WAPE), 4), "error_media_movil": round(float(r.WAPE_media_movil_3), 4),
                      "gana": bool(r.WAPE < r.WAPE_media_movil_3),
                      "dentro": None if pd.isna(getattr(r, "dentro_intervalo", None)) else bool(r.dentro_intervalo)}
                     for r in live.itertuples()],
        "resumen": {"meses_evaluados": int(len(live)), "meses_gana": int((live.WAPE < live.WAPE_media_movil_3).sum()),
                    "error_modelo": round(float(live.WAPE.mean()), 4) if len(live) else None,
                    "error_media_movil": round(float(live.WAPE_media_movil_3.mean()), 4) if len(live) else None,
                    "error_modelo_backtest": round(float(p1[p1.fuente == "backtest"].WAPE.mean()), 4),
                    "error_media_movil_backtest": round(float(p1[p1.fuente == "backtest"].WAPE_media_movil_3.mean()), 4),
                    "cobertura": _coverage(perf[(perf.flujo == flow) & (perf.fuente == "producción")])},
        "destacados": [{"hs6": h, "producto": _title(labels.get(h, h)), "valor": round(float(v)),
                        "variacion": round(float(v / recent[h] - 1), 4) if recent.get(h, 0) > 0 else None}
                       for h, v in h1.items()],
    }


def build(state: Path, links: dict | None = None, labels_csv: Path | None = None) -> dict:
    """labels_csv: hs6_nombres.csv del proyecto de análisis, para refrescar los nombres de productos sin
    esperar a la próxima actualización mensual."""
    status = A.read_status(state)
    fc, totals, perf = A.read(state, A.FORECASTS), A.read(state, A.TOTALS), A.read(state, A.PERFORMANCE)
    if labels_csv:
        names = pd.read_csv(labels_csv, dtype={"hs6": str})
        labels = names.set_index("hs6")["nombre"]
    else:
        labels = A.read(state, A.LABELS).set_index("hs6")["etiqueta"]
    level = "alerta" if status["alertas"] else "aviso" if status.get("avisos") else "normal"
    next_data = status["proximo_mes"]
    return {
        "esquema": ESQUEMA,
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos_hasta": status["ultimo_mes"],
        "proximo_dato": next_data,
        # Aduanas publica cada mes cerca del día 30 del mes siguiente
        "proxima_actualizacion": str(pd.Period(next_data, "M") + 1),
        "estado": level,
        "alertas": len(status["alertas"]),
        "avisos": len(status.get("avisos", [])),
        "enlaces": {k: v for k, v in (links or {}).items() if v},
        "flujos": {flow: _flow_summary(state, flow, fc, totals, perf, labels) for flow in FLOWS},
    }


def write_site(state: Path, out: Path, links: dict | None = None, labels_csv: Path | None = None) -> Path:
    """Carpeta que se publica en GitHub Pages: resumen.json y una portada mínima que explica qué es."""
    out.mkdir(parents=True, exist_ok=True)
    summary = build(state, links, labels_csv)
    path = out / "resumen.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shutil.copy(state / A.REPORT, out / "reporte.md")
    (out / "index.html").write_text(
        "<!doctype html><html lang='es'><meta charset='utf-8'><title>Pronóstico del comercio exterior de Chile</title>"
        "<p>Datos publicados automáticamente por el pipeline "
        f"<a href='https://github.com/{A.REPO}'>{A.REPO}</a>: <a href='resumen.json'>resumen.json</a> "
        f"(datos hasta {summary['datos_hasta']}) y <a href='reporte.md'>reporte del último mes</a>.</p></html>",
        encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estado", type=Path, default=Path(os.environ.get("PRONOSTICO_STATE", "state")))
    parser.add_argument("--salida", type=Path, default=Path("site"))
    parser.add_argument("--etiquetas", type=Path, help="hs6_nombres.csv para refrescar los nombres de productos")
    args = parser.parse_args()
    links = {"tablero": os.environ.get("TABLERO_URL"), "api": os.environ.get("API_URL"),
             "codigo": f"https://github.com/{A.REPO}"}
    path = write_site(args.estado, args.salida, links, args.etiquetas)
    print(f"{path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
