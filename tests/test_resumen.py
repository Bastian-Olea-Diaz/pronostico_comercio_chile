"""resumen.json es un contrato con el portafolio: si cambia su forma, el portafolio dibuja mal."""
import json

import pytest

from pronostico import pipeline, resumen, sources


@pytest.fixture(scope="module")
def site(local_data, tmp_path_factory):
    state = tmp_path_factory.mktemp("estado")
    pipeline.bootstrap("2026-04", state=state, local_data=local_data)
    first = json.loads(resumen.write_site(state, tmp_path_factory.mktemp("site0")).read_text(encoding="utf-8"))
    pipeline.run(state, fetch=sources.local_fetch(local_data), published=lambda m: True)
    out = tmp_path_factory.mktemp("site")
    path = resumen.write_site(state, out, {"codigo": "https://github.com/x/y", "tablero": None})
    return first, json.loads(path.read_text(encoding="utf-8")), out


def test_schema(site):
    _, data, out = site
    assert data["esquema"] == resumen.ESQUEMA
    assert data["datos_hasta"] == "2026-05" and data["proximo_dato"] == "2026-06" and data["proxima_actualizacion"] == "2026-07"
    assert data["estado"] in {"normal", "aviso", "alerta"}
    assert data["enlaces"] == {"codigo": "https://github.com/x/y"}   # los enlaces vacíos no se publican
    for flow in ("exportaciones", "importaciones"):
        f = data["flujos"][flow]
        assert len(f["historia"]) == 24 and f["historia"][-1]["mes"] == "2026-05"
        assert [p["h"] for p in f["pronostico"]] == [1, 2, 3]
        assert 0 < f["error_tipico"] < 1
        assert len(f["destacados"]) == resumen.TOP_PRODUCTS
        assert f["resumen"]["meses_evaluados"] == len(f["aciertos"]) == 1
        assert all(p["inferior"] < p["valor"] < p["superior"] for p in f["pronostico"])
        assert f["intervalo"]["nivel"] == 0.8 and f["intervalo"]["metodo"] == "conformal"
        assert isinstance(f["aciertos"][0]["dentro"], bool)
        assert f["resumen"]["cobertura"]["evaluados"] == 1
    assert (out / "index.html").exists() and (out / "reporte.md").exists()
    assert (out / "resumen.json").stat().st_size < 50_000


def test_first_publication_has_no_live_track_record(site):
    """Antes del primer mes en producción no hay aciertos en vivo: el portafolio muestra el backtest."""
    first, _, _ = site
    f = first["flujos"]["exportaciones"]
    assert f["aciertos"] == [] and f["resumen"]["meses_evaluados"] == 0
    assert f["resumen"]["error_modelo"] is None and f["resumen"]["error_modelo_backtest"] > 0
    assert f["resumen"]["cobertura"] == {"evaluados": 0, "dentro": 0}


def test_labels_can_be_refreshed(local_data, tmp_path):
    """Publicar resumen puede usar los nombres vigentes del proyecto de análisis sin reprocesar el mes."""
    state = tmp_path / "estado"
    pipeline.bootstrap("2026-04", state=state, local_data=local_data)
    data = resumen.build(state)
    top = data["flujos"]["exportaciones"]["destacados"][0]["hs6"]
    names = tmp_path / "hs6_nombres.csv"
    import pandas as pd
    pd.DataFrame({"hs6": [top], "nombre": ["Nombre corregido"]}).to_csv(names, index=False)
    fresh = resumen.build(state, labels_csv=names)
    assert fresh["flujos"]["exportaciones"]["destacados"][0]["producto"] == "Nombre corregido"
