"""Datos sintéticos con la misma estructura que los de Aduanas, para probar el pipeline sin red."""
import numpy as np
import pandas as pd
import pytest

MONTHS = pd.period_range("2024-01", "2026-06", freq="M")
FLOWS = ("exportaciones", "importaciones")


def synthetic_panel(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hs6 = [f"{c:02d}{i:04d}" for c in (8, 22, 30, 74, 84) for i in range(12)]
    countries = ["220", "225", "336", "331", "517"]
    rows = []
    for h in hs6:
        for c in countries:
            level = rng.lognormal(12, 1.5)
            season = 1 + 0.3 * np.sin(2 * np.pi * (np.arange(len(MONTHS)) + rng.integers(12)) / 12)
            active = rng.random(len(MONTHS)) > 0.05
            for t, p in enumerate(MONTHS):
                if active[t]:
                    v = level * season[t] * rng.lognormal(0, 0.3)
                    rows.append((p, "bienes", h, c, v, v / rng.uniform(2, 4), 3, 2, 1, v / 3, False, h[:2]))
    cols = ["periodo", "segmento", "hs6", "pais", "valor_usd", "cantidad", "n_items", "n_decl", "n_empresas",
            "peso_kg", "pseudo_pais", "capitulo"]
    return pd.DataFrame(rows, columns=cols)


def ingest_log(months) -> pd.DataFrame:
    return pd.DataFrame([{"kind": f, "year": p.year, "month": p.month, "source": f"{f}-{p}.rar", "rows": 10_000,
                          "lines_joined": 0, "lines_padded": 0, "invalid_rows": 0, "numeric_coerced": 0,
                          "date_coerced": 0, "rows_outside_month": 0} for f in FLOWS for p in months])


@pytest.fixture(scope="session")
def local_data(tmp_path_factory):
    """Carpeta con la estructura de data/ del proyecto de análisis: paneles procesados y log de ingesta."""
    root = tmp_path_factory.mktemp("datos")
    (root / "processed").mkdir()
    (root / "interim").mkdir()
    for i, flow in enumerate(FLOWS):
        synthetic_panel(i).to_parquet(root / "processed" / f"panel_{flow}.parquet", index=False)
    ingest_log(MONTHS).to_csv(root / "interim" / "_ingest_log.csv", index=False)
    return root
