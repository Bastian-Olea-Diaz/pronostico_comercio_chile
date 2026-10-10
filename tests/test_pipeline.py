"""El pipeline completo sobre datos sintéticos: estado inicial, mes nuevo, reprocesos y parada por calidad."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from pronostico import artifacts as A
from pronostico import modeling, monitoring, pipeline, sources


@pytest.fixture(scope="module")
def state(local_data, tmp_path_factory):
    folder = tmp_path_factory.mktemp("estado")
    assert pipeline.bootstrap("2026-04", state=folder, local_data=local_data) == pipeline.EXIT_UPDATED
    return folder


def run(folder, local_data, **kw):
    return pipeline.run(folder, fetch=sources.local_fetch(local_data), published=lambda m: True, **kw)


def test_bootstrap_creates_every_artifact(state):
    for name in [A.STATUS, A.FORECASTS, A.TOTALS, A.PERFORMANCE, A.DRIFT, A.INGEST, A.MODELS, A.REPORT,
                 A.FORECAST_HISTORY, A.LABELS, A.COUNTRIES]:
        assert (state / name).exists(), name
    status = A.read_status(state)
    assert status["ultimo_mes"] == "2026-04" and status["proximo_mes"] == "2026-05"
    fc = pd.read_parquet(state / A.FORECASTS)
    assert set(fc.objetivo) == {"2026-05", "2026-06", "2026-07"}
    assert (fc[["pronostico", "pronostico_media", "media_movil_3"]] >= 0).all().all()
    assert (pd.read_csv(state / A.PERFORMANCE).fuente == "backtest").all()


def test_quality_alert_stops_without_touching_state(state, local_data, tmp_path):
    folder = tmp_path / "estado"
    shutil.copytree(state, folder)
    before = (folder / A.STATUS).read_text(encoding="utf-8")
    good = sources.local_fetch(local_data)

    def corrupted(flow, month):
        panel, stats = good(flow, month)
        return panel, {**stats, "invalid_rows": 5}

    assert pipeline.run(folder, fetch=corrupted, published=lambda m: True) == pipeline.EXIT_STOPPED
    assert (folder / A.STATUS).read_text(encoding="utf-8") == before
    assert pd.read_parquet(folder / A.PANEL.format(flow="exportaciones")).periodo.max() == pd.Period("2026-04", "M")
    assert "detenido" in (folder / A.REPORT).read_text(encoding="utf-8")


def test_unpublished_month_is_a_no_op(state, local_data, tmp_path):
    folder = tmp_path / "estado"
    shutil.copytree(state, folder)
    code = pipeline.run(folder, fetch=sources.local_fetch(local_data), published=lambda m: False)
    assert code == pipeline.EXIT_NO_NEWS


def test_monthly_runs_update_state_and_evaluate_every_horizon(state, local_data, tmp_path):
    folder = tmp_path / "estado"
    shutil.copytree(state, folder)
    assert run(folder, local_data) == pipeline.EXIT_UPDATED          # 2026-05
    assert run(folder, local_data) == pipeline.EXIT_UPDATED          # 2026-06
    status = A.read_status(folder)
    assert status["ultimo_mes"] == "2026-06"
    perf = pd.read_csv(folder / A.PERFORMANCE)
    live = perf[perf.fuente == "producción"]
    assert set(map(tuple, live[["periodo", "h"]].to_numpy())) == {("2026-05", 1), ("2026-06", 1), ("2026-06", 2)}
    assert all(d["modelo"] == "nuevo" for d in status["modelos"].values())

    # reprocesar un mes ya procesado no cambia nada; saltarse un mes es un error
    assert run(folder, local_data, month="2026-06") == pipeline.EXIT_NO_NEWS
    with pytest.raises(ValueError):
        run(folder, local_data, month="2026-09")
    assert len(pd.read_csv(folder / A.PERFORMANCE)) == len(perf)
    panel = pd.read_parquet(folder / A.PANEL.format(flow="exportaciones"))
    assert not panel.duplicated(["periodo", "segmento", "hs6", "pais"]).any()


def test_broken_model_falls_back_to_previous(state, local_data, tmp_path, monkeypatch):
    folder = tmp_path / "estado"
    shutil.copytree(state, folder)
    real_forecast = modeling.forecast
    calls = {"n": 0}

    def broken_first_time(models, grid, t0, flow):
        fc = real_forecast(models, grid, t0, flow)
        calls["n"] += 1
        if calls["n"] == 1:   # el primer pronóstico (modelo nuevo de exportaciones) sale roto
            fc["pronostico"] = -1.0
        return fc

    monkeypatch.setattr(modeling, "forecast", broken_first_time)
    assert run(folder, local_data) == pipeline.EXIT_UPDATED
    status = json.loads((folder / A.STATUS).read_text(encoding="utf-8"))
    assert status["modelos"]["exportaciones"]["modelo"] == "anterior"
    assert status["modelos"]["importaciones"]["modelo"] == "nuevo"
    assert any("reutilizó" in a for a in status["alertas"])
    assert (pd.read_parquet(folder / A.FORECASTS).pronostico >= 0).all()


def test_saved_models_predict_like_the_originals(state, local_data):
    from comercio_chile.features import grid_from_panel

    panel = pd.read_parquet(local_data / "processed" / "panel_exportaciones.parquet")
    grid = grid_from_panel(panel[panel.periodo <= pd.Period("2026-04", "M")], "exportaciones")
    t0 = grid.t_of("2026-04")
    fresh = modeling.train_all(grid, t0, "exportaciones")
    folder = state / "prueba_modelos"
    modeling.save(fresh, folder)
    loaded = modeling.load(folder)
    a, b = modeling.forecast(fresh, grid, t0, "exportaciones"), modeling.forecast(loaded, grid, t0, "exportaciones")
    np.testing.assert_allclose(a.pronostico, b.pronostico, rtol=1e-9)
    np.testing.assert_allclose(a.pronostico_media, b.pronostico_media, rtol=1e-9)


def test_losing_to_baseline_two_months_in_a_row_alerts():
    ref = pd.DataFrame({"flujo": "x", "h": 1, "fuente": "backtest", "periodo": [f"2025-{m:02d}" for m in range(1, 13)],
                        "WAPE": np.linspace(0.30, 0.34, 12), "WAPE_media_movil_3": 0.40, "sesgo": np.linspace(-0.05, 0.05, 12)})
    live = pd.DataFrame({"flujo": "x", "h": 1, "fuente": "producción", "periodo": ["2026-01", "2026-02", "2026-03"],
                         "WAPE": [0.32, 0.33, 0.33], "WAPE_media_movil_3": [0.40, 0.31, 0.30], "sesgo": [0.0, 0.0, 0.0]})
    status = monitoring.performance_status(pd.concat([ref, live])).set_index("periodo")
    assert list(status.pierde_vs_media_movil) == ["ok", "advertencia", "alerta"]
    assert status.loc["2026-03", "estado"] == "alerta"
