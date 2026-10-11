"""Entrenamiento, persistencia y pronóstico con los modelos del proyecto de análisis.

Por flujo y horizonte hay dos modelos, los mismos que se validaron en el notebook 03:
    l1        mediana del cambio relativo al nivel reciente: el mejor pronóstico por serie
    tweedie   media con offset en el nivel: para sumar series y pronosticar totales sin sesgo
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from comercio_chile import forecast as F
from comercio_chile.features import Grid, regular_mask

from .artifacts import FORECASTS, TOTALS

HORIZONS = (1, 2, 3)
INTERVAL_LEVEL = 0.80
INTERVAL_WINDOW = 12      # meses de errores pasados que calibran el intervalo
_BASE = F.ModelConfig().params
CONFIGS = {
    "l1": F.ModelConfig(params={**_BASE, "objective": "l1"}, target="delta"),
    "tweedie": F.ModelConfig(params={**_BASE, "objective": "tweedie", "tweedie_variance_power": 1.5}, target="raw_offset"),
}


class SavedModel:
    """Modelo cargado desde disco con la misma interfaz que usa comercio_chile.forecast.predict."""

    def __init__(self, booster: lgb.Booster, target: str, categories: dict):
        self.booster, self.target_, self.categories_ = booster, target, categories

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X)


def name(flow: str, h: int, kind: str) -> str:
    return f"{flow}_h{h}_{kind}"


def train_all(grid: Grid, t0: int, flow: str) -> dict:
    """Entrena los modelos de un flujo con toda la información disponible en el origen t0."""
    return {name(flow, h, kind): F.fit(grid, t0, h, cfg) for h in HORIZONS for kind, cfg in CONFIGS.items()}


def save(models: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for key, m in models.items():
        booster = m.booster_ if hasattr(m, "booster_") else m.booster
        booster.save_model(folder / f"{key}.txt")
        meta = {"target": m.target_, "categories": {c: [str(v) for v in cats] for c, cats in m.categories_.items()}}
        (folder / f"{key}.json").write_text(json.dumps(meta), encoding="utf-8")


def load(folder: Path) -> dict:
    out = {}
    for path in sorted(folder.glob("*.txt")):
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        cats = {c: pd.Index(v) for c, v in meta["categories"].items()}
        out[path.stem] = SavedModel(lgb.Booster(model_file=str(path)), meta["target"], cats)
    return out


def forecast(models: dict, grid: Grid, t0: int, flow: str) -> pd.DataFrame:
    """Pronósticos a 1, 2 y 3 meses desde el origen t0 para las series regulares, junto al baseline
    (media móvil de 3 meses) con que se comparan después."""
    rows = np.where(regular_mask(grid, t0))[0]
    keys = grid.keys.iloc[rows]
    baseline = F.BASELINES["media_movil_3"](grid.valor[rows], t0, 1).clip(0)
    out = []
    for h in HORIZONS:
        out.append(pd.DataFrame({
            "flujo": flow, "hs6": keys.hs6.to_numpy(), "pais": keys.pais.to_numpy(),
            "commodity": keys.commodity.to_numpy(), "origen": str(grid.months[t0]), "h": h,
            "objetivo": str(grid.months[t0] + h),
            "pronostico": F.predict(models[name(flow, h, "l1")], grid, t0, h, rows),
            "pronostico_media": F.predict(models[name(flow, h, "tweedie")], grid, t0, h, rows),
            "media_movil_3": baseline,
        }))
    return pd.concat(out, ignore_index=True)


def totals(fc: pd.DataFrame) -> pd.DataFrame:
    """Totales por flujo y mes objetivo. El total usa el modelo de media (Tweedie): sumar medianas
    subestima el total, como se mostró en el notebook 03."""
    g = fc.groupby(["flujo", "origen", "objetivo", "h"], as_index=False)
    return g.agg(total=("pronostico_media", "sum"), total_media_movil_3=("media_movil_3", "sum"), series=("hs6", "size"))


def sanity_check(fc: pd.DataFrame, grid: Grid, t0: int) -> list[str]:
    """Controles sobre un modelo recién entrenado, antes de publicarlo. Un modelo roto (datos corruptos,
    error de código) suele producir valores imposibles o totales desproporcionados; un modelo malo pero
    plausible se detecta después, con el monitoreo de desempeño."""
    problems = []
    for col in ("pronostico", "pronostico_media"):
        if fc[col].isna().any() or np.isinf(fc[col]).any():
            problems.append(f"{col}: valores faltantes o infinitos")
        if (fc[col] < 0).any():
            problems.append(f"{col}: valores negativos")
    recent = grid.valor[:, max(0, t0 - 2): t0 + 1].sum(axis=0).mean()   # total mensual reciente (todas las series)
    for (flow, h), g in fc.groupby(["flujo", "h"]):
        ratio = g.pronostico_media.sum() / recent
        if not 0.3 <= ratio <= 2.0:
            problems.append(f"{flow} h={h}: el total pronosticado es {ratio:.2f} veces el total reciente")
    return problems


def add_intervals(tot: pd.DataFrame, performance: pd.DataFrame, level: float = INTERVAL_LEVEL,
                  window: int = INTERVAL_WINDOW) -> pd.DataFrame:
    """Intervalo conformal para cada total: pronóstico × e^(±q), donde q es el error |log(observado / pronosticado)|
    en la posición ⌈(n + 1) · nivel⌉ de los últimos `window` meses evaluados con el mismo horizonte (backtest y
    producción). El notebook 09 del proyecto de análisis mostró que la banda anterior, el error relativo
    promedio, cubría menos de lo que aparentaba, porque un promedio no es un cuantil."""
    from comercio_chile.intervals import conformal_quantile, log_ratio_score, total_interval

    perf = performance.dropna(subset=["total_pronosticado", "total_observado"])
    perf = perf[(perf.total_pronosticado > 0) & (perf.total_observado > 0)]
    out = tot.copy()
    for col in ("inferior", "superior", "calibracion_meses"):
        out[col] = np.nan
    for i, r in out.iterrows():
        past = perf[(perf.flujo == r.flujo) & (perf.h == r.h)].sort_values("periodo")
        past = past.drop_duplicates("periodo", keep="last").tail(window)   # producción reemplaza al backtest del mismo mes
        q = conformal_quantile(log_ratio_score(past.total_observado, past.total_pronosticado), level)
        if np.isfinite(q):
            out.loc[i, ["inferior", "superior"]] = total_interval([r.total], q)[0][0], total_interval([r.total], q)[1][0]
        out.loc[i, "calibracion_meses"] = len(past)
    out["nivel"] = level
    return out


def write_forecasts(fc: pd.DataFrame, folder: Path, performance: pd.DataFrame | None = None) -> pd.DataFrame:
    fc.to_parquet(folder / FORECASTS, index=False)
    tot = totals(fc)
    if performance is not None:
        tot = add_intervals(tot, performance)
    tot.to_parquet(folder / TOTALS, index=False)
    return tot
