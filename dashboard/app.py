"""Tablero de pronósticos del comercio exterior de Chile (Streamlit).

Lee los artefactos de la última release de GitHub (o de ARTIFACTS_DIR, para desarrollo local):
    streamlit run dashboard/app.py
"""
import os
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pronostico import artifacts as A  # noqa: E402

COLORS = {"exportaciones": "#2a78d6", "importaciones": "#eb6834"}   # paleta del proyecto (validada para daltonismo)
NEUTRAL = "#8a8984"                                                # media móvil: referencia, siempre punteada
REPO_URL = f"https://github.com/{A.REPO}"
ANALYSIS_URL = "https://github.com/Bastian-Olea-Diaz/analisis_comercio_chile"

st.set_page_config(page_title="Pronóstico del comercio exterior de Chile", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Cargando la última versión de los pronósticos…")
def load() -> dict:
    local = os.environ.get("ARTIFACTS_DIR")
    folder = Path(local) if local else Path("/tmp/artefactos_tablero")
    release = "local" if local else A.download_release(folder)
    data = {name: A.read(folder, name) for name in (A.FORECASTS, A.TOTALS, A.PERFORMANCE, A.DRIFT, A.LABELS, A.COUNTRIES)}
    data["status"], data["release"] = A.read_status(folder), release
    for flow in COLORS:
        p = pd.read_parquet(folder / A.PANEL.format(flow=flow))
        p = p[(p.segmento == "bienes") & ~p.pseudo_pais & (p.hs6 != "000000")]
        data[flow] = p.assign(periodo=p.periodo.astype(str))[["periodo", "hs6", "pais", "valor_usd"]]
    return data


SERIES = ["Observado", "Pronóstico", "Media móvil 3 meses"]
DASHES = [[1, 0], [2, 3], [6, 4]]   # continua, punteada, segmentada: la identidad no depende solo del color
# Números y meses en español (d3-format / d3-time-format)
LOCALE = {
    "number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["US$ ", ""]},
    "time": {"dateTime": "%A, %e de %B de %Y, %X", "date": "%d/%m/%Y", "time": "%H:%M:%S", "periods": ["AM", "PM"],
             "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
             "shortDays": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
             "months": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                        "octubre", "noviembre", "diciembre"],
             "shortMonths": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]},
}


def es(value: float, decimals: int = 0) -> str:
    """Número con separadores en español: 8.872,5"""
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def history_and_forecast(history: pd.DataFrame, forecast: pd.DataFrame, color: str, title: str) -> alt.Chart:
    """Valor observado, pronóstico (unido al último observado) y media móvil, en US$. Un solo eje, en millones
    o en miles según el tamaño de la serie."""
    last = history.iloc[[-1]]
    df = pd.concat([history.assign(serie="Observado"),
                    last.assign(serie="Pronóstico"), forecast.assign(serie="Pronóstico"),
                    last.assign(serie="Media móvil 3 meses"),
                    forecast.assign(valor=forecast.media_movil, serie="Media móvil 3 meses")])
    divisor, unit = (1e6, "US$ millones") if df.valor.max() >= 1e6 else (1e3, "US$ miles")
    df = df.assign(valor=df.valor / divisor)
    enc = dict(
        x=alt.X("periodo:T", title=None, axis=alt.Axis(format="%b %Y", grid=False)),
        y=alt.Y("valor:Q", title=unit, axis=alt.Axis(format=",.0f"), scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", scale=alt.Scale(domain=SERIES, range=[color, color, NEUTRAL]),
                        legend=alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=2.5)),
        strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=SERIES, range=DASHES), legend=None),
        tooltip=[alt.Tooltip("periodo:T", format="%B %Y", title="Mes"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", format=",.1f", title=unit)],
    )
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(**enc)
    points = alt.Chart(df[(df.serie == "Pronóstico") & df.periodo.isin(forecast.periodo)]).mark_point(
        size=70, filled=True, color=color).encode(x="periodo:T", y="valor:Q", tooltip=enc["tooltip"])
    layers = [lines, points]
    if {"inferior", "superior"} <= set(forecast) and forecast.inferior.notna().all():
        band = forecast.assign(inferior=forecast.inferior / divisor, superior=forecast.superior / divisor)
        layers.insert(0, alt.Chart(band).mark_area(opacity=0.18, color=color).encode(
            x="periodo:T", y="inferior:Q", y2="superior:Q",
            tooltip=[alt.Tooltip("periodo:T", format="%B %Y", title="Mes"),
                     alt.Tooltip("inferior:Q", format=",.1f", title=f"Rango 80%: desde ({unit})"),
                     alt.Tooltip("superior:Q", format=",.1f", title=f"hasta ({unit})")]))
    return alt.layer(*layers).properties(height=340, title=title).configure(locale=LOCALE)


def title_es(text: str) -> str:
    """Mayúscula inicial sin capitalizar artículos: 'Estados Unidos de América'. Los nombres de productos ya
    vienen en formato de oración ('Cátodo de cobre') y se dejan como están."""
    if not text.isupper():
        return text
    return re.sub(r" (De|Del|La|Las|Los|Y|E|En|Con|Para|Sin) ", lambda m: m.group(0).lower(), text.title())


data = load()
status = data["status"]
labels = data[A.LABELS].set_index("hs6")["etiqueta"].map(title_es)
countries = data[A.COUNTRIES].set_index("codigo")["glosa"].map(title_es)

st.title("Pronóstico del comercio exterior de Chile")
st.caption(f"Datos de Aduanas hasta **{status['ultimo_mes']}** · se actualiza solo cuando Aduanas publica el mes "
           f"siguiente (cerca del día 30) · versión `{data['release']}` · [código]({REPO_URL}) · "
           f"[proyecto de análisis]({ANALYSIS_URL})")
for alert in status["alertas"]:
    st.error(f"Alerta: {alert}", icon="⚠️")
for notice in status.get("avisos", []):
    st.info(f"Aviso: {notice}", icon="ℹ️")

flow = st.radio("Flujo", list(COLORS), horizontal=True, format_func=str.capitalize)
color = COLORS[flow]
fc_all = data[A.FORECASTS].query("flujo == @flow")
panel = data[flow]

tab_total, tab_series, tab_monitor = st.tabs(["Total", "Producto y país", "Monitoreo del modelo"])

with tab_total:
    keys = fc_all[["hs6", "pais"]].drop_duplicates()
    hist = panel.merge(keys).groupby("periodo", as_index=False).valor_usd.sum().tail(24).rename(columns={"valor_usd": "valor"})
    tot = data[A.TOTALS].query("flujo == @flow").rename(columns={"objetivo": "periodo", "total": "valor",
                                                                    "total_media_movil_3": "media_movil"})
    has_band = {"inferior", "superior"} <= set(tot) and tot.inferior.notna().all()
    for col, r in zip(st.columns(len(tot)), tot.itertuples()):
        delta = r.valor / hist.valor.iloc[-1] - 1
        col.metric(f"Pronóstico {r.periodo}", f"US$ {es(r.valor / 1e6)} M",
                   f"{es(100 * delta, 1)}% vs {hist.periodo.iloc[-1]}", delta_color="off")
        if has_band:
            col.caption(f"Rango 80%: {es(r.inferior / 1e6)}–{es(r.superior / 1e6)} M")
    cols = ["periodo", "valor", "media_movil"] + (["inferior", "superior"] if has_band else [])
    st.altair_chart(history_and_forecast(hist, tot[cols], color,
                                         f"{flow.capitalize()}: series regulares ({es(len(keys))} pares producto–país)"),
                    use_container_width=True)
    st.caption("El total suma las series con comercio en al menos 10 de los últimos 12 meses (la mayor parte del valor). "
               "Usa el modelo de media (Tweedie), porque sumar medianas subestima el total."
               + (" La banda es un intervalo conformal del 80%: su ancho se fija con los errores de los últimos 12 "
                  "meses evaluados para que el valor real caiga dentro 8 de cada 10 veces." if has_band else ""))

with tab_series:
    h1 = fc_all[fc_all.h == 1]
    products = h1.groupby("hs6").pronostico_media.sum().sort_values(ascending=False).index   # los más grandes primero
    c1, c2 = st.columns(2)
    hs6 = c1.selectbox("Producto", products, format_func=lambda h: f"{labels.get(h, 'sin descripción')} ({h})")
    options = h1[h1.hs6 == hs6].sort_values("pronostico_media", ascending=False).pais
    pais = c2.selectbox("País", options, format_func=lambda c: countries.get(c, c))
    sel = fc_all[(fc_all.hs6 == hs6) & (fc_all.pais == pais)].rename(
        columns={"objetivo": "periodo", "pronostico": "valor", "media_movil_3": "media_movil"})
    hist = panel[(panel.hs6 == hs6) & (panel.pais == pais)].groupby("periodo", as_index=False).valor_usd.sum()
    hist = hist.rename(columns={"valor_usd": "valor"}).tail(24)
    st.altair_chart(history_and_forecast(hist, sel[["periodo", "valor", "media_movil"]], color,
                                         f"{labels.get(hs6, hs6)} → {countries.get(pais, pais)}"), use_container_width=True)
    table = sel[["periodo", "h", "valor", "media_movil"]].assign(
        valor=lambda d: d.valor.map(lambda v: f"US$ {es(v)}"), media_movil=lambda d: d.media_movil.map(lambda v: f"US$ {es(v)}"))
    st.dataframe(table.rename(columns={"periodo": "Mes", "h": "Meses adelante", "valor": "Pronóstico",
                                       "media_movil": "Media móvil 3 meses"}), hide_index=True, use_container_width=True)

with tab_monitor:
    perf = data[A.PERFORMANCE].query("flujo == @flow and h == 1").copy()
    long = perf.melt(id_vars=["periodo", "fuente"], value_vars=["WAPE", "WAPE_media_movil_3"], var_name="metodo", value_name="error")
    long["metodo"] = long.metodo.map({"WAPE": "Modelo", "WAPE_media_movil_3": "Media móvil 3 meses"})
    scale = alt.Scale(domain=["Modelo", "Media móvil 3 meses"], range=[color, NEUTRAL])
    chart = alt.Chart(long).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=alt.X("periodo:T", title=None, axis=alt.Axis(format="%b %Y", grid=False)),
        y=alt.Y("error:Q", title="Error ponderado (WAPE)", axis=alt.Axis(format=".0%"), scale=alt.Scale(zero=False)),
        color=alt.Color("metodo:N", scale=scale, legend=alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=2.5)),
        strokeDash=alt.StrokeDash("metodo:N", scale=alt.Scale(domain=["Modelo", "Media móvil 3 meses"], range=[[1, 0], [6, 4]]), legend=None),
        tooltip=[alt.Tooltip("periodo:T", format="%B %Y", title="Mes"), alt.Tooltip("metodo:N", title="Método"),
                 alt.Tooltip("error:Q", format=".1%", title="WAPE"), alt.Tooltip("fuente:N", title="Origen")],
    )
    start = perf.loc[perf.fuente == "producción", "periodo"].min()
    layers = [chart]
    if pd.notna(start):
        rule = alt.Chart(pd.DataFrame({"periodo": [start]})).mark_rule(color="#52514e", strokeDash=[2, 2])
        layers.append(rule.encode(x="periodo:T"))
    st.altair_chart(alt.layer(*layers).properties(height=320, title="Error del pronóstico a 1 mes (menor es mejor)")
                    .configure(locale=LOCALE),
                    use_container_width=True)
    st.caption("Antes de la línea vertical: backtest (reentrenando cada mes sobre datos históricos), que define lo normal "
               "para el gráfico de control. Después: pronósticos publicados en producción, evaluados al llegar el dato real.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calidad de los datos del último mes")
        quality = [{"Chequeo": k, "Valor": v["valor"], "Estado": v["estado"]}
                   for q in status["calidad"] if q["flujo"] == flow for k, v in q["chequeos"].items()]
        if quality:
            st.dataframe(pd.DataFrame(quality), hide_index=True, use_container_width=True)
        else:
            st.write("Se completa con el primer mes procesado en producción.")
    with c2:
        st.subheader("Drift del último mes")
        drift = data[A.DRIFT].query("flujo == @flow")
        drift = drift[drift.periodo == drift.periodo.max()]
        drift = drift[["feature", "tipo", "PSI", "nivel_actual", "estado"]].round(3).astype(object)
        st.dataframe(drift.where(drift.notna(), "").rename(columns={
            "feature": "Variable", "tipo": "Tipo", "nivel_actual": "Nivel", "estado": "Estado"}),
            hide_index=True, use_container_width=True)
