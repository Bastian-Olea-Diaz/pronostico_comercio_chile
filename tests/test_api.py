"""La API sobre un estado generado por el pipeline con datos sintéticos."""
import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pronostico import pipeline


@pytest.fixture(scope="module")
def client(local_data, tmp_path_factory, monkeypatch_module):
    folder = tmp_path_factory.mktemp("artefactos")
    pipeline.bootstrap("2026-04", state=folder, local_data=local_data)
    monkeypatch_module.setenv("ARTIFACTS_DIR", str(folder))
    import api.main

    importlib.reload(api.main)
    return TestClient(api.main.app), folder


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health_and_status(client):
    c, _ = client
    assert c.get("/salud").json()["datos_hasta"] == "2026-04"
    assert c.get("/estado").json()["proximo_mes"] == "2026-05"


def test_series_forecast(client):
    c, folder = client
    fc = pd.read_parquet(folder / "pronosticos.parquet")
    row = fc[fc.flujo == "exportaciones"].iloc[0]
    body = c.get("/pronosticos", params={"flujo": "exportaciones", "hs6": row.hs6, "pais": row.pais}).json()
    assert [p["h"] for p in body["pronosticos"]] == [1, 2, 3]
    assert body["pronosticos"][0]["objetivo"] == "2026-05"


def test_unknown_series_and_bad_input(client):
    c, _ = client
    assert c.get("/pronosticos", params={"flujo": "exportaciones", "hs6": "999999", "pais": "1"}).status_code == 404
    assert c.get("/pronosticos", params={"flujo": "otro", "hs6": "080610", "pais": "225"}).status_code == 422
    assert c.get("/pronosticos", params={"flujo": "exportaciones", "hs6": "uva", "pais": "225"}).status_code == 422


def test_totals_and_performance(client):
    c, _ = client
    totals = c.get("/pronosticos/totales", params={"flujo": "importaciones"}).json()
    assert {t["h"] for t in totals} == {1, 2, 3}
    perf = c.get("/monitoreo/desempeno", params={"flujo": "exportaciones"}).json()
    assert perf and all(p["fuente"] == "backtest" for p in perf)
