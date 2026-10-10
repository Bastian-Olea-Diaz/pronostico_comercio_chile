# Pronóstico del comercio exterior de Chile en producción

En mi proyecto [analisis_comercio_chile](https://github.com/Bastian-Olea-Diaz/analisis_comercio_chile) construí
un modelo que pronostica las exportaciones e importaciones de Chile por producto y país, y diseñé cómo
monitorearlo. Pero todo vivía en notebooks: el modelo solo funcionaba cuando yo los ejecutaba. Este repositorio
es la segunda parte: el mismo modelo, operando solo.

Cada mes, cuando Aduanas publica un archivo nuevo (cerca del día 30 del mes siguiente), el sistema:

1. lo detecta y lo descarga desde datos.gob.cl;
2. valida su calidad y se detiene si viene mal, sin tocar nada;
3. compara los pronósticos que hizo antes con lo que realmente pasó;
4. revisa si los datos cambiaron de comportamiento (drift);
5. reentrena los modelos y solo publica los nuevos si pasan los controles;
6. publica una versión con modelos, pronósticos y reportes, y avisa con un issue si algo requiere revisión.

Los resultados se ven en un tablero, se consultan con una API y se muestran en vivo en mi
[portafolio](https://bastian-olea-diaz.github.io/comercio-ext-chile).

## Arquitectura

```
GitHub Actions (revisa a diario si hay un mes nuevo)
   ├─ descarga e ingesta del mes (paquete comercio_chile)
   ├─ calidad de datos ──────────────► si falla: se detiene y abre un issue
   ├─ desempeño del pronóstico anterior y drift
   ├─ reentrenamiento y controles ───► si el modelo nuevo falla: se reutiliza el anterior
   ├─ release de GitHub (registro de modelos y datos)
   └─ resumen.json en GitHub Pages
          │
          ├── API (FastAPI + Docker, Hugging Face Spaces)
          ├── tablero (Streamlit Community Cloud)
          └── portafolio (lee resumen.json)
```

Algunas decisiones que tomé y por qué:

- **Pronóstico por lotes.** Los datos cambian una vez al mes, así que calcular en cada consulta solo agregaría
  latencia y dependencias. La API sirve pronósticos ya calculados y no necesita LightGBM.
- **Releases como registro de modelos.** Cada mes queda una versión identificada (`datos-AAAA-MM`) con sus
  modelos, pronósticos, paneles y reportes. Es trazable, gratis y no depende de otra plataforma.
- **Alertas y avisos separados.** En el proyecto de análisis vi que el drift de precios de 2026 fue persistente
  sin degradar el modelo; si eso abriera un issue cada mes, terminaría ignorando las alertas. Solo abren issue
  las alertas que piden acción (calidad, desempeño, modelo descartado); el drift queda como aviso informativo.
- **Un mes a la vez, sin saltos.** Reprocesar un mes ya procesado no cambia nada, y saltarse uno es un error.
- **Honestidad del historial.** El historial "en vivo" empieza con el primer mes posterior al despliegue. Los
  meses anteriores se muestran como backtest, separados, porque esos datos ya existían cuando se evaluaron.

## Monitoreo

| Capa | Qué revisa | Qué pasa si falla |
|---|---|---|
| Calidad de datos | Filas descartadas, valores no convertibles, volumen vs el mismo mes del año anterior, códigos nunca vistos | El pipeline se detiene sin modificar el estado y abre un issue |
| Desempeño | Error (WAPE) y sesgo del pronóstico contra lo observado, con un gráfico de control calibrado en el backtest; además, si pierde contra la media móvil dos meses seguidos | Alerta e issue |
| Drift | PSI de las variables del modelo, mix de productos e índice de precios implícito | Aviso informativo |
| Modelo nuevo | Valores negativos o faltantes, totales desproporcionados | Se reutiliza el modelo anterior y se alerta |

Los umbrales son los que calibré en el proyecto de análisis, donde no generaron falsas alarmas en un período de
control.

## Estructura

```
src/pronostico/
  pipeline.py     orquestación: bootstrap (estado inicial) y run (mes nuevo)
  sources.py      detección y descarga del mes en datos.gob.cl
  modeling.py     entrenamiento, guardado y pronóstico (LightGBM: mediana por serie y media para totales)
  monitoring.py   calidad de datos, desempeño, gráfico de control y drift
  resumen.py      resumen.json para el portafolio
  artifacts.py    lectura de releases (lo usan también la API y el tablero)
api/              FastAPI y Dockerfile
dashboard/        tablero en Streamlit
tests/            pruebas con datos sintéticos (sin red)
.github/          tests, actualización mensual, estado inicial y despliegue de la API
```

## Uso local

Requiere una copia de `analisis_comercio_chile` al lado de esta carpeta.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ../analisis_comercio_chile -e ".[api,dashboard,dev]"
python -m pytest tests

# simular el pipeline con los datos ya procesados del proyecto de análisis, sin descargar nada
python -m pronostico.pipeline bootstrap --hasta 2026-06 --datos-locales ../analisis_comercio_chile/data
python -m pronostico.pipeline run --datos-locales ../analisis_comercio_chile/data

ARTIFACTS_DIR=state uvicorn api.main:app        # API en http://localhost:8000/docs
ARTIFACTS_DIR=state streamlit run dashboard/app.py
```

## Tecnologías

Python · pandas · LightGBM · FastAPI · Docker · Streamlit · Altair · GitHub Actions · GitHub Releases ·
GitHub Pages · Hugging Face Spaces · pytest

## Licencia de los datos

Los datos son publicados por el Servicio Nacional de Aduanas de Chile en el Portal de Datos Abiertos del
Gobierno de Chile, bajo licencia Creative Commons Atribución.
