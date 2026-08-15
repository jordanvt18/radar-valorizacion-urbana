# 🏙️ Radar de Valorización Urbana

Sistema de predicción de valorización inmobiliaria para Quito y Guayaquil, Ecuador. Combina datos transaccionales, movilidad urbana, imágenes satelitales, servicios y indicadores socioeconómicos mediante modelos de machine learning multimodal para proyectar tendencias de valorización con intervalos de confianza y explicabilidad.

---

## ✨ Características

- 🗂️ **Pipeline ETL completo** con 5 fuentes de datos integradas
- 🗺️ **Rejilla hexagonal H3** (resolución 8) para análisis espacial uniforme
- 🌳 **Modelo tabular LightGBM** con Quantile Regression para intervalos de predicción
- 🖼️ **Modelo multimodal CNN+MLP** que combina datos tabulares con imágenes satelitales
- 🎲 **Calibración de incertidumbre** mediante MC Dropout y Conformal Prediction
- 🔍 **Explicabilidad SHAP** global y por celda individual
- 🎛️ **Simulador de escenarios** para evaluar impacto de cambios urbanos
- 🚀 **API REST** con FastAPI y documentación OpenAPI automática
- 🗺️ **Frontend interactivo** con mapa de calor Leaflet y panel SHAP en Plotly
- 🐳 **Despliegue con Docker** y docker-compose

---

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.10 o superior
- pip o conda
- Docker y docker-compose (opcional, para despliegue containerizado)

### Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd radar-valorizacion-urbana

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Copiar archivo de configuración de entorno
cp .env.example .env
```

### Ejecutar ETL

Genera los datos sintéticos (transacciones, movilidad, satélite, servicios, socioeconómico) y construye features:

```bash
python scripts/run_etl.py
```

Los datos se guardan en `data/processed/` en formato CSV y Parquet.

### Entrenar Modelos

```bash
python -m src.models.train
```

Los artefactos del modelo se guardan en `models/`.

### Iniciar la API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

La documentación interactiva está disponible en:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Iniciar el Frontend

El frontend es estático (HTML/CSS/JS). Servir con cualquier servidor HTTP:

```bash
# Opción 1: Python
python -m http.server 3000 --directory web

# Opción 2: Node (si tiene npx)
npx serve web -p 3000
```

Abrir http://localhost:3000 en el navegador.

---

## 📁 Estructura del Proyecto

```
radar-valorizacion-urbana/
├── config/
│   └── config.yaml                # Configuración centralizada
├── data/
│   ├── raw/                       # Datos crudos (con .gitkeep)
│   └── processed/                 # Datos procesados (con .gitkeep)
├── models/                        # Artefactos de modelos entrenados
├── notebooks/
│   └── model_analysis.ipynb       # Análisis de modelos
├── outputs/                       # Salidas de predicciones y gráficos
├── scripts/
│   └── run_etl.py                 # Orquestador del pipeline ETL
├── src/
│   ├── __init__.py
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── etl_transactions.py    # Transacciones inmobiliarias
│   │   ├── etl_mobility.py        # Movilidad urbana
│   │   ├── etl_satellite.py       # Imágenes satelitales
│   │   ├── etl_services.py        # Servicios urbanos
│   │   └── etl_socioeconomic.py   # Indicadores socioeconómicos
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py      # Feature engineering sobre H3
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tabular_model.py       # LightGBM con Quantile Regression
│   │   ├── multimodal_model.py    # CNN + MLP
│   │   ├── calibration.py         # MC Dropout y conformal
│   │   ├── explain.py             # SHAP global y local
│   │   └── train.py               # Pipeline de entrenamiento
│   └── api/
│       ├── __init__.py
│       ├── main.py                # App FastAPI
│       ├── routes.py              # Endpoints REST
│       ├── schemas.py             # Modelos Pydantic
│       └── scenario_simulator.py  # Simulador de escenarios
├── tests/
│   ├── __init__.py
│   ├── test_etl.py                # Tests del pipeline ETL
│   ├── test_models.py             # Tests de modelos
│   └── test_api.py                # Tests de la API
├── web/
│   ├── index.html                 # Página principal
│   ├── main.js                    # Lógica del mapa y UI
│   ├── styles.css                 # Estilos
│   └── plotly_panel.js            # Panel SHAP con Plotly
├── .env.example                   # Variables de entorno de ejemplo
├── .gitignore
├── docker-compose.yml             # Orquestación de contenedores
├── Dockerfile                     # Imagen de la API
├── METHODOLOGY.md                 # Documentación metodológica completa
├── pyproject.toml                 # Configuración del proyecto
├── requirements.txt               # Dependencias de Python
└── README.md                      # Este archivo
```

---

## 🔌 Endpoints de la API

| Endpoint | Método | Descripción | Parámetros |
|---|---|---|---|
| `/health` | GET | Estado del servicio | — |
| `/predict` | POST | Predicción de valorización | `h3_index`, `model` (opcional), `horizon_years` (opcional) |
| `/explain` | POST | Explicación SHAP | `h3_index`, `model` (opcional) |
| `/simulate` | POST | Simulación de escenario | `h3_index`, `modifications` (dict) |
| `/cells` | GET | Celdas H3 disponibles | `city` (opcional) |

### Ejemplos de Uso

```bash
# Verificar estado
curl http://localhost:8000/health

# Obtener celdas disponibles
curl http://localhost:8000/cells

# Predicción para una celda
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"h3_index": "88e6e3099bfffff", "model": "tabular"}'

# Explicación SHAP
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"h3_index": "88e6e3099bfffff"}'

# Simular escenario
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "h3_index": "88e6e3099bfffff",
    "modifications": {
      "avg_travel_time_cbd_min": 20,
      "schools_count": 5
    }
  }'
```

---

## ⚙️ Configuración

La configuración se centraliza en `config/config.yaml`. Los parámetros principales son:

| Sección | Parámetro | Valor por defecto | Descripción |
|---|---|---|---|
| `h3.resolution` | 8 | Resolución de rejilla hexagonal |
| `cities.quito` | bbox, num_cells | 30 celdas | Configuración de Quito |
| `cities.guayaquil` | bbox, num_cells | 25 celdas | Configuración de Guayaquil |
| `transactions.total_count` | 2500 | Número de transacciones a generar |
| `transactions.annual_price_growth` | 0.065 | Crecimiento anual de precios (6.5%) |
| `satellite.patch_size` | 128 | Tamaño de parche satelital |
| `random_seed` | 42 | Semilla para reproducibilidad |

Las variables de entorno (`.env`) controlan parámetros de ejecución:

```env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
```

---

## 🐳 Docker

### Construcción y despliegue con docker-compose

```bash
# Construir y levantar todos los servicios
docker-compose up --build

# En background
docker-compose up -d --build

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

### Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `api` | 8000 | API FastAPI con los modelos |
| `web` | 3000 | Frontend estático servido con Nginx |

### Volúmenes

- `./data:/app/data` — datos del proyecto
- `./models:/app/models` — artefactos de modelos
- `./outputs:/app/outputs` — salidas de predicciones
- `./config:/app/config` — configuración
- `./web:/usr/share/nginx/html:ro` — archivos del frontend

### Construir solo la imagen de la API

```bash
docker build -t radar-valorizacion-api .
docker run -p 8000:8000 radar-valorizacion-api
```

---

## 🧪 Desarrollo

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Con cobertura
pytest --cov=src --cov-report=html

# Tests específicos
pytest tests/test_etl.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v
```

### Configuración de Desarrollo

El archivo `pyproject.toml` contiene la configuración para:

- **pytest**: configuración de tests, marcadores
- **ruff**: linter y formateador
- **mypy**: verificación de tipos (opcional)
- **black**: formateador (compatible con ruff)

### Flujo de Trabajo Recomendado

1. Ejecutar ETL: `python scripts/run_etl.py`
2. Entrenar modelos: `python -m src.models.train`
3. Iniciar API: `uvicorn src.api.main:app --reload`
4. Iniciar frontend: `python -m http.server 3000 --directory web`
5. Ejecutar tests: `pytest -v`

### Añadir una Nueva Ciudad

1. Agregar la configuración de la ciudad en `config/config.yaml` bajo `cities`
2. Ejecutar `python scripts/run_etl.py` (detectará automáticamente la nueva ciudad)
3. Re-entrenar el modelo: `python -m src.models.train`

---

## 📄 Licencia

Este proyecto es de uso interno. Todos los derechos reservados.

---

*Para detalles técnicos completos, consultar [METHODOLOGY.md](METHODOLOGY.md).*
