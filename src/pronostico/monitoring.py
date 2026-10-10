"""Monitoreo en producción: las capas del notebook 06, ahora sobre datos que llegan cada mes.

    capa 0  calidad de datos       al recibir el archivo; si falla, el pipeline se detiene
    capa 1  data drift             PSI de las features y del mix por capítulo; índice de precios implícito
    capa 3  desempeño              el pronóstico publicado el mes anterior contra lo que realmente pasó,
                                   con un gráfico de control calibrado en el backtest inicial

Los umbrales son los del proyecto de análisis, que se calibraron en un período de control sin falsas alarmas.
"""
import numpy as np
import pandas as pd
from comercio_chile import drift as D
from comercio_chile import forecast as F
from comercio_chile.features import Grid

OK, WARN, ALERT = D.OK, D.WARN, D.ALERT
REFERENCE_ORIGINS = 12      # meses de referencia para el data drift
UNDER_BASELINE_STREAK = 2   # meses seguidos peor que la media móvil que disparan una alerta


def worst(states) -> str:
    states = list(states)
    return ALERT if ALERT in states else WARN if WARN in states else OK


# ------------------------------------------------------------------------- capa 0: calidad de datos
def data_quality(stats: dict, ingest_log: pd.DataFrame, month_panel: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Mismas reglas que comercio_chile.drift.data_quality, sobre el archivo recién ingerido.

    stats         estadísticas de ingesta del mes (comercio_chile.ingest.IngestStats)
    ingest_log    estadísticas de los meses anteriores (para comparar volumen con el mismo mes del año pasado)
    month_panel   panel del mes nuevo; history: panel de los 12 meses anteriores
    """
    month = pd.Period(year=stats["year"], month=stats["month"], freq="M")
    prev = ingest_log[(ingest_log.kind == stats["kind"]) & (ingest_log.year == month.year - 1) & (ingest_log.month == month.month)]
    volume = 100 * (stats["rows"] / prev.rows.iloc[0] - 1) if len(prev) else np.nan
    new = ~month_panel.hs6.isin(set(history.hs6)) | ~month_panel.pais.isin(set(history.pais))
    new_share = 100 * month_panel.valor_usd[new].sum() / month_panel.valor_usd.sum()
    checks = {
        "filas descartadas": (stats["invalid_rows"], ALERT if stats["invalid_rows"] > 0 else OK),
        "valores no convertibles": (stats["numeric_coerced"] + stats["date_coerced"],
                                    ALERT if stats["numeric_coerced"] + stats["date_coerced"] > 0 else OK),
        "Δ filas vs mismo mes del año anterior (%)": (volume, OK if np.isnan(volume) or abs(volume) <= 25
                                                      else WARN if abs(volume) <= 50 else ALERT),
        "% del valor con códigos nunca vistos": (new_share, ALERT if new_share > 1 else OK),
    }
    return {"flujo": stats["kind"], "periodo": str(month), "estado": worst(s for _, s in checks.values()),
            "chequeos": {k: {"valor": None if pd.isna(v) else round(float(v), 3), "estado": s} for k, (v, s) in checks.items()}}


# ------------------------------------------------------------------------------- capa 3: desempeño
def evaluate(previous: pd.DataFrame, grid: Grid, month: str) -> pd.DataFrame:
    """Compara los pronósticos que apuntaban a `month` con el valor observado (un par sin comercio vale 0)."""
    t = grid.t_of(month)
    actual = pd.Series(grid.valor[:, t], index=pd.MultiIndex.from_frame(grid.keys[["hs6", "pais"]]))
    rows = []
    for (flow, h), g in previous[previous.objetivo == month].groupby(["flujo", "h"]):
        y = actual.reindex(pd.MultiIndex.from_frame(g[["hs6", "pais"]])).fillna(0).to_numpy()
        model, base = F.metrics(y, g.pronostico.to_numpy()), F.metrics(y, g.media_movil_3.to_numpy())
        rows.append({"flujo": flow, "periodo": month, "h": h, "fuente": "producción", "WAPE": model["WAPE"],
                     "WAPE_media_movil_3": base["WAPE"], "sesgo": model["sesgo_total"], "series": len(g),
                     "total_observado": y.sum(), "total_pronosticado": g.pronostico_media.sum()})
    return pd.DataFrame(rows)


def backtest(grid: Grid, flow: str, months: list[str]) -> pd.DataFrame:
    """Desempeño a 1 mes en meses ya observados, reentrenando cada mes como en producción. Sirve de
    referencia para el gráfico de control: lo que es normal para este modelo, no un umbral inventado."""
    from .modeling import CONFIGS

    wf = F.walk_forward(grid, 1, months, CONFIGS["l1"])
    totals = F.walk_forward(grid, 1, months, CONFIGS["tweedie"]).groupby("periodo").modelo.sum()   # total: modelo de media
    rows = []
    for month, g in wf.groupby("periodo"):
        model, base = F.metrics(g.y.values, g.modelo.values), F.metrics(g.y.values, g.media_movil_3.values)
        rows.append({"flujo": flow, "periodo": month, "h": 1, "fuente": "backtest", "WAPE": model["WAPE"],
                     "WAPE_media_movil_3": base["WAPE"], "sesgo": model["sesgo_total"], "series": len(g),
                     "total_observado": g.y.sum(), "total_pronosticado": totals[month]})
    return pd.DataFrame(rows)


def performance_status(history: pd.DataFrame) -> pd.DataFrame:
    """Estado de cada mes en producción (h = 1): gráfico de control sobre WAPE y sesgo con μ y σ del
    backtest, y una regla de negocio: si el modelo pierde contra la media móvil varios meses seguidos,
    alerta aunque el gráfico de control no lo marque."""
    out = []
    for flow, g in history[history.h == 1].sort_values("periodo").groupby("flujo"):
        ref, live = g[g.fuente == "backtest"], g[g.fuente == "producción"].copy()
        if live.empty:
            continue
        for col in ("WAPE", "sesgo"):
            mu, sd = D.control_limits(ref[col])
            live[f"z_{col}"] = (live[col] - mu) / sd
            live[f"control_{col}"] = D.control_status(live[col], mu, sd).to_numpy()
        losing = (live.WAPE > live.WAPE_media_movil_3).astype(int)
        streak = losing.groupby((losing == 0).cumsum()).cumsum()
        live["pierde_vs_media_movil"] = np.where(streak >= UNDER_BASELINE_STREAK, ALERT, np.where(losing == 1, WARN, OK))
        live["estado"] = [worst(r) for r in live[["control_WAPE", "control_sesgo", "pierde_vs_media_movil"]].to_numpy()]
        out.append(live)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------------------- capa 1: data drift
def data_drift(grid: Grid, flow: str, month: str) -> pd.DataFrame:
    """Drift del pronóstico que apuntaba a `month`, el que se acaba de evaluar: sus features (origen
    month − 1) y su objetivo (month, ya observado) contra los 12 orígenes anteriores. El índice de precios
    se mide en el mes recién llegado."""
    t = grid.t_of(month)
    origin = t - 1
    ref_origins = list(range(max(D.FULL_ORIGIN, origin - REFERENCE_ORIGINS), origin))
    dd = D.data_drift(D.feature_frame(grid, ref_origins), D.feature_frame(grid, [origin]))
    prices = D.price_index(grid, ref_origins, t)
    extra = pd.DataFrame([{"feature": f"índice de precios ({seg})", "tipo": "precio", "PSI": np.nan, "KS": np.nan,
                           "nivel_actual": v, "estado": D.level(v, *D.PRICE_THRESHOLDS)} for seg, v in prices.items()])
    out = pd.concat([dd, extra], ignore_index=True)
    out.insert(0, "periodo", month)
    out.insert(0, "flujo", flow)
    return out
