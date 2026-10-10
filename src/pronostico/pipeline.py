"""Orquestación del pipeline mensual.

Uso:
    python -m pronostico.pipeline bootstrap --hasta 2026-08 [--datos-locales RUTA]
    python -m pronostico.pipeline run [--mes 2026-09] [--datos-locales RUTA]

bootstrap construye el estado inicial: paneles históricos, backtest de referencia para el monitoreo, primeros
modelos y pronósticos. run procesa el mes siguiente al último del estado: si Aduanas aún no lo publica,
termina sin cambios (código 10); si los datos no pasan los controles de calidad, se detiene sin tocar el
estado (código 2); si todo resulta, actualiza el estado (código 0) y deja un reporte para la release.

--datos-locales usa archivos ya ingeridos en otra carpeta en vez de descargar, para probar el pipeline
completo sin red (por ejemplo, con data/ del proyecto de análisis).
"""
import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import ROOT
from . import artifacts as A
from . import monitoring as M
from . import sources

STATE = Path(os.environ.get("PRONOSTICO_STATE", ROOT / "state"))
EXIT_UPDATED, EXIT_STOPPED, EXIT_NO_NEWS = 0, 2, 10
BACKTEST_MONTHS = 12


# ------------------------------------------------------------------------------------------- estado
def _panel(state: Path, flow: str) -> pd.DataFrame:
    return pd.read_parquet(state / A.PANEL.format(flow=flow))


def _models_dir(state: Path) -> Path:
    folder = state / "modelos"
    if not folder.exists() and (state / A.MODELS).exists():   # la release guarda los modelos comprimidos
        with zipfile.ZipFile(state / A.MODELS) as z:
            z.extractall(folder)
    return folder


def _zip_models(state: Path) -> None:
    with zipfile.ZipFile(state / A.MODELS, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((state / "modelos").iterdir()):
            z.write(f, f.name)


def _append(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Agrega filas a un CSV de historial reemplazando las que ya existían con las mismas claves
    (reprocesar un mes no duplica filas)."""
    if path.exists():
        old = pd.read_csv(path, dtype={k: str for k in keys if k not in ("h", "year", "month")})
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(keys, keep="last")
    new.to_csv(path, index=False)
    return new


def _keep_history(fc: pd.DataFrame, state: Path, origins: int = 6) -> None:
    """Pronósticos de los últimos orígenes: los de 2 y 3 meses se evalúan cuando llega su mes objetivo,
    después de que pronosticos.parquet ya se reemplazó por los del mes siguiente."""
    path = state / A.FORECAST_HISTORY
    hist = pd.concat([pd.read_parquet(path), fc], ignore_index=True) if path.exists() else fc
    hist = hist.drop_duplicates(["flujo", "hs6", "pais", "origen", "h"], keep="last")
    keep = sorted(hist.origen.unique())[-origins:]
    hist[hist.origen.isin(keep)].to_parquet(path, index=False)


def _versions() -> dict:
    return {"pipeline": os.environ.get("GITHUB_SHA", "local")[:7], "comercio_chile": os.environ.get("BASE_SHA", "local")[:7]}


# ------------------------------------------------------------------------------------------ pasos
def _train_and_forecast(panels: dict, month: str, state: Path, previous_models: dict | None) -> tuple[pd.DataFrame, dict]:
    """Entrena modelos nuevos por flujo y los publica si pasan los controles; si no, reutiliza los anteriores."""
    from comercio_chile.features import grid_from_panel

    from . import modeling

    folder = state / "modelos"
    forecasts, decisions = [], {}
    for flow, panel in panels.items():
        grid = grid_from_panel(panel, flow)
        t = grid.t_of(month)
        models = modeling.train_all(grid, t, flow)
        fc = modeling.forecast(models, grid, t, flow)
        problems = modeling.sanity_check(fc, grid, t)
        if problems and previous_models:
            old = {k: v for k, v in previous_models.items() if k.startswith(flow)}
            fc = modeling.forecast(old, grid, t, flow)
            decisions[flow] = {"modelo": "anterior", "motivo": problems}
        elif problems:
            raise RuntimeError(f"{flow}: el modelo no pasa los controles y no hay uno anterior: {problems}")
        else:
            modeling.save(models, folder)
            decisions[flow] = {"modelo": "nuevo", "entrenado_hasta": month}
        forecasts.append(fc)
    return pd.concat(forecasts, ignore_index=True), decisions


def _labels(state: Path) -> None:
    from comercio_chile.config import REFERENCES

    shutil.copy(REFERENCES / "codigos" / "hs6_etiquetas_2025.csv", state / A.LABELS)
    shutil.copy(REFERENCES / "codigos" / "paises.csv", state / A.COUNTRIES)


def _status(state: Path, month: str, decisions: dict, quality: list, drift: pd.DataFrame,
            alerts: list, notices: list) -> dict:
    """alertas: requieren que alguien revise (abren un issue). avisos: informan un cambio en los datos que
    no exige acción inmediata; el notebook 06 mostró que el drift de precios de 2026 fue persistente sin
    degradar el modelo, y alertar cada mes por él solo generaría fatiga de alertas."""
    perf = M.performance_status(pd.read_csv(state / A.PERFORMANCE))
    latest = perf[perf.periodo == month] if not perf.empty else perf
    alerts, notices = list(alerts), list(notices)
    for r in latest.itertuples():
        if r.estado == M.ALERT:
            alerts.append(f"Desempeño {r.flujo}: WAPE {r.WAPE:.1%} (media móvil {r.WAPE_media_movil_3:.1%}), "
                          f"sesgo {r.sesgo:+.1%}")
    for r in drift[(drift.estado == M.ALERT) & drift.tipo.isin(["composición", "precio", "objetivo"])].itertuples():
        notices.append(f"Drift {r.flujo}: {r.feature}")
    status = {
        "ultimo_mes": month,
        "proximo_mes": str(pd.Period(month, "M") + 1),
        "actualizado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versiones": _versions(),
        "modelos": decisions,
        "calidad": quality,
        "desempeno": latest.drop(columns=["fuente"], errors="ignore").to_dict("records"),
        "drift": {flow: {"alertas": int((g.estado == M.ALERT).sum()), "advertencias": int((g.estado == M.WARN).sum())}
                  for flow, g in drift.groupby("flujo")},
        "alertas": alerts,
        "avisos": notices,
    }
    (state / A.STATUS).write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return status


def _report(status: dict, totals: pd.DataFrame) -> str:
    lines = [f"## Pronóstico del comercio exterior de Chile · datos hasta {status['ultimo_mes']}", ""]
    lines += ["### Alertas", *(f"- {a}" for a in status["alertas"])] if status["alertas"] else ["Sin alertas."]
    if status["avisos"]:
        lines += ["", "### Avisos (informativos)", *(f"- {a}" for a in status["avisos"])]
    lines += ["", "### Desempeño del pronóstico anterior (1 mes)", "", "| Flujo | WAPE modelo | WAPE media móvil | Sesgo | Estado |",
              "|---|---|---|---|---|"]
    lines += [f"| {r['flujo']} | {r['WAPE']:.1%} | {r['WAPE_media_movil_3']:.1%} | {r['sesgo']:+.1%} | {r['estado']} |"
              for r in status["desempeno"]] or ["| (primer mes en producción) | | | | |"]
    lines += ["", "### Totales pronosticados (series regulares, US$ millones)", "", "| Flujo | Mes | Pronóstico | Media móvil 3 |", "|---|---|---|---|"]
    lines += [f"| {r.flujo} | {r.objetivo} | {r.total / 1e6:,.0f} | {r.total_media_movil_3 / 1e6:,.0f} |" for r in totals.itertuples()]
    lines += ["", "### Modelos", *(f"- {flow}: {d['modelo']}" + (f" (motivo: {'; '.join(d['motivo'])})" if "motivo" in d else "")
                                   for flow, d in status["modelos"].items())]
    lines += ["", f"Versiones: pipeline `{status['versiones']['pipeline']}`, comercio_chile `{status['versiones']['comercio_chile']}`"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------- comandos
def bootstrap(through: str, state: Path = STATE, local_data: Path | None = None) -> int:
    """Estado inicial con datos hasta `through` (incluido)."""
    from comercio_chile import clean, download, ingest
    from comercio_chile.features import grid_from_panel

    state.mkdir(parents=True, exist_ok=True)
    end = pd.Period(through, "M")
    if local_data:
        log = pd.read_csv(local_data / "interim" / "_ingest_log.csv")
        panels = {f: pd.read_parquet(local_data / "processed" / f"panel_{f}.parquet") for f in sources.FLOWS}
    else:   # reconstrucción completa desde datos.gob.cl (~1 GB, ~30 min)
        download.main()
        ingest.main()
        log = pd.read_csv(ingest.INTERIM / "_ingest_log.csv")
        panels = {f: clean.build_panel(f)[0] for f in sources.FLOWS}
    log = log[pd.PeriodIndex.from_fields(year=log.year, month=log.month, freq="M") <= end]
    log.drop(columns=["invalid_samples"], errors="ignore").to_csv(state / A.INGEST, index=False)
    panels = {f: p[p.periodo <= end].reset_index(drop=True) for f, p in panels.items()}
    for f, p in panels.items():
        p.to_parquet(state / A.PANEL.format(flow=f), index=False)
    _labels(state)

    months = [str(end - k) for k in range(BACKTEST_MONTHS - 1, -1, -1)]
    perf = pd.concat([M.backtest(grid_from_panel(p, f), f, months) for f, p in panels.items()], ignore_index=True)
    perf.to_csv(state / A.PERFORMANCE, index=False)
    drift = pd.concat([M.data_drift(grid_from_panel(p, f), f, through) for f, p in panels.items()], ignore_index=True)
    drift.to_csv(state / A.DRIFT, index=False)

    fc, decisions = _train_and_forecast(panels, through, state, None)
    from .modeling import write_forecasts, totals
    write_forecasts(fc, state)
    _zip_models(state)
    _keep_history(fc, state)
    status = _status(state, through, decisions, [], drift, [], [])
    (state / A.REPORT).write_text(_report(status, totals(fc)), encoding="utf-8")
    print(f"Estado inicial listo: datos hasta {through}, {len(fc):,} pronósticos")
    return EXIT_UPDATED


def run(state: Path = STATE, month: str | None = None, fetch=sources.fetch_month, published=sources.is_published) -> int:
    from . import modeling

    current = A.read_status(state)
    month = month or current["proximo_mes"]
    if pd.Period(month, "M") <= pd.Period(current["ultimo_mes"], "M"):
        print(f"{month} ya está procesado (último mes: {current['ultimo_mes']})")
        return EXIT_NO_NEWS
    if pd.Period(month, "M") != pd.Period(current["ultimo_mes"], "M") + 1:
        raise ValueError(f"Los meses se procesan en orden: el siguiente es {current['proximo_mes']}, no {month}")
    if not published(month):
        print(f"{month} aún no está publicado en datos.gob.cl")
        return EXIT_NO_NEWS

    # capa 0: calidad de datos, antes de tocar el estado
    ingest_log = pd.read_csv(state / A.INGEST)
    panels, stats, quality = {}, {}, []
    for flow in sources.FLOWS:
        new_rows, stats[flow] = fetch(flow, month)
        panel = _panel(state, flow)
        history = panel[panel.periodo > pd.Period(month, "M") - 13]
        quality.append(M.data_quality(stats[flow], ingest_log, new_rows, history))
        panels[flow] = pd.concat([panel[panel.periodo != pd.Period(month, "M")], new_rows], ignore_index=True)
    if any(q["estado"] == M.ALERT for q in quality):
        report = ["## Pipeline detenido por calidad de datos", "", f"Mes: {month}. El estado no se modificó.", ""]
        for q in quality:
            report += [f"### {q['flujo']}: {q['estado']}", *(f"- {k}: {v['valor']} ({v['estado']})" for k, v in q["chequeos"].items()), ""]
        (state / A.REPORT).write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report))
        return EXIT_STOPPED

    from comercio_chile.features import grid_from_panel

    for flow, panel in panels.items():
        panel.to_parquet(state / A.PANEL.format(flow=flow), index=False)
    new_log = pd.DataFrame([{k: v for k, v in s.items() if k != "invalid_samples"} for s in stats.values()])
    _append(state / A.INGEST, new_log, ["kind", "year", "month"])

    # capa 3: los pronósticos que apuntaban a este mes (hechos hace 1, 2 y 3 meses) contra lo observado;
    # capa 1: drift de las entradas de esos pronósticos
    previous = pd.read_parquet(state / A.FORECAST_HISTORY)
    grids = {flow: grid_from_panel(p, flow) for flow, p in panels.items()}
    perf = pd.concat([M.evaluate(previous[previous.flujo == f], g, month) for f, g in grids.items()], ignore_index=True)
    _append(state / A.PERFORMANCE, perf, ["flujo", "periodo", "h", "fuente"])
    drift = pd.concat([M.data_drift(g, f, month) for f, g in grids.items()], ignore_index=True)
    _append(state / A.DRIFT, drift, ["flujo", "periodo", "feature"])

    fc, decisions = _train_and_forecast(panels, month, state, modeling.load(_models_dir(state)))
    modeling.write_forecasts(fc, state)
    _zip_models(state)
    _keep_history(fc, state)
    notices = [f"Calidad {q['flujo']}: {k} = {v['valor']}" for q in quality for k, v in q["chequeos"].items() if v["estado"] == M.WARN]
    alerts = [f"Modelo {f}: se reutilizó el anterior ({'; '.join(d['motivo'])})" for f, d in decisions.items() if "motivo" in d]
    status = _status(state, month, decisions, quality, drift, alerts, notices)
    (state / A.REPORT).write_text(_report(status, modeling.totals(fc)), encoding="utf-8")
    print(f"{month} procesado. Alertas: {len(status['alertas'])}")
    return EXIT_UPDATED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--hasta", required=True)
    b.add_argument("--datos-locales", type=Path)
    r = sub.add_parser("run")
    r.add_argument("--mes")
    r.add_argument("--datos-locales", type=Path)
    args = parser.parse_args()
    if args.cmd == "bootstrap":
        code = bootstrap(args.hasta, local_data=args.datos_locales)
    else:
        kwargs = {"fetch": sources.local_fetch(args.datos_locales), "published": lambda m: True} if args.datos_locales else {}
        code = run(month=args.mes, **kwargs)
    sys.exit(code)


if __name__ == "__main__":
    main()
