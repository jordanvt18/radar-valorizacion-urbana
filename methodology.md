# Metodología — Radar de Valorización Urbana

**Sistema de predicción de valorización inmobiliaria para Quito y Guayaquil, Ecuador**

Versión: 0.1.0
Fecha: 2026-08-10
Autores: Equipo de Desarrollo Radar de Valorización Urbana

---

## Tabla de Contenidos

1. [Introducción y Objetivos](#1-introducción-y-objetivos)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Fuentes de Datos](#3-fuentes-de-datos)
4. [Pipeline ETL Geoespacial](#4-pipeline-etl-geoespacial)
5. [Rejilla Hexagonal H3](#5-rejilla-hexagonal-h3)
6. [Feature Engineering](#6-feature-engineering)
7. [Modelos Predictivos](#7-modelos-predictivos)
8. [Calibración de Incertidumbre](#8-calibración-de-incertidumbre)
9. [Explicabilidad con SHAP](#9-explicabilidad-con-shap)
10. [Simulador de Escenarios Urbanos](#10-simulador-de-escenarios-urbanos)
11. [API REST](#11-api-rest)
12. [Frontend Interactivo](#12-frontend-interactivo)
13. [Métricas de Evaluación](#13-métricas-de-evaluación)
14. [Limitaciones y Trabajo Futuro](#14-limitaciones-y-trabajo-futuro)
15. [Referencias](#15-referencias)

---

## 1. Introducción y Objetivos

El mercado inmobiliario ecuatoriano ha experimentado un crecimiento sostenido durante la última década, particularmente en las ciudades de Quito y Guayaquil, donde la dinámica urbana, la infraestructura y los factores socioeconómicos influyen directamente en la valorización de propiedades. Sin embargo, la ausencia de herramientas centralizadas que integren múltiples fuentes de datos para anticipar tendencias de valorización representa una brecha crítica para inversores, planificadores urbanos y ciudadanos.

### 1.1 Problema

La valorización inmobiliaria en Ecuador se caracteriza por:

- **Heterogeneidad espacial**: barrios adyacentes pueden presentar dinámicas de precio radicalmente distintas debido a factores micro-locales que no son visibles en agregados a nivel ciudad.
- **Información fragmentada**: los registros de transacciones, datos de movilidad, servicios urbanos e indicadores socioeconómicos se encuentran dispersos entre múltiples instituciones públicas y privadas, sin un sistema unificado de consulta.
- **Falta de modelos predictivos accesibles**: no existen herramientas públicas que combinen datos multivariados para proyectar valorización futura con intervalos de confianza cuantificables.
- **Ausencia de explicabilidad**: los modelos de machine learning aplicados al sector inmobiliario suelen ser cajas negras, limitando su adopción por parte de stakeholders que requieren comprender los factores que impulsan las predicciones.
- **Estimaciones puntuales sin incertidumbre**: las herramientas existentes proporcionan valores puntuales sin cuantificar la incertidumbre asociada, impidiendo que los usuarios evalúen el riesgo de sus decisiones.

### 1.2 Objetivo General

Desarrollar un sistema integral de predicción de valorización urbana que combine datos transaccionales, de movilidad, satelitales, de servicios y socioeconómicos mediante técnicas de machine learning multimodal, generando proyecciones espaciales calibradas con medidas de incertidumbre y explicabilidad.

### 1.3 Objetivos Específicos

1. **Integrar datos multivariados** en una rejilla hexagonal H3 que permita análisis espacial consistente y comparable entre ciudades.
2. **Construir features derivados** que capturen tendencias temporales, accesibilidad urbana, densidad de servicios y características del entorno construido, incluyendo interacciones no lineales entre variables.
3. **Entrenar modelos tabulares (LightGBM) y multimodales (CNN+MLP)** para predecir valorización anualizada por celda hexagonal, con cuantificación de incertidumbre mediante quantile regression y MC Dropout.
4. **Calibrar la incertidumbre** de las predicciones mediante técnicas de conformal prediction que garanticen cobertura marginal correcta de los intervalos.
5. **Proporcionar explicabilidad** a través de valores SHAP globales y por celda individual, permitiendo a los usuarios comprender qué factores impulsan la valorización en cada zona.
6. **Implementar una API REST** que exponga predicciones, explicaciones y simulaciones de escenarios urbanos con documentación automática.
7. **Desarrollar un frontend interactivo** con mapas de calor, paneles SHAP, simuladores de escenarios y comparadores de zonas.

### 1.4 Alcance

El sistema cubre las áreas metropolitanas de Quito y Guayaquil, utilizando datos sintéticos generados con parámetros realistas basados en tendencias observadas del mercado ecuatoriano. La arquitectura está diseñada para incorporar datos reales cuando estén disponibles, requiriendo únicamente la adaptación de los módulos ETL correspondientes. El sistema no constituye una tasación inmobiliaria oficial, sino una herramienta de soporte decisional para análisis de tendencias urbanas.

---

## 2. Arquitectura del Sistema

El sistema sigue una arquitectura modular de cuatro capas: ingesta de datos (ETL), ingeniería de características, modelado predictivo, y exposición mediante API y frontend interactivo.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RADAR DE VALORIZACIÓN URBANA                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CAPA 1: ETL Geoespacial (Extract, Transform, Load)         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │Transacc. │ │Movilidad │ │Satélite  │ │Servicios │       │   │
│  │  │  ETL     │ │  ETL     │ │  ETL     │ │  ETL     │       │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │   │
│  │       │            │            │            │              │   │
│  │  ┌────┴────────────┴────────────┴────────────┴─────┐        │   │
│  │  │            ETL Socioeconómico                    │        │   │
│  │  └────────────────────┬────────────────────────────┘        │   │
│  └───────────────────────┼─────────────────────────────────────┘   │
│                          │                                          │
│  ┌───────────────────────▼─────────────────────────────────────┐   │
│  │  CAPA 2: Feature Engineering                                 │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │   │
│  │  │ Rejilla H3  │→ │ Agregación   │→ │ Features finales│    │   │
│  │  │ (res 8)     │  │ espacial     │  │ (CSV + Parquet) │    │   │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘    │   │
│  └───────────────────────┬─────────────────────────────────────┘   │
│                          │                                          │
│  ┌───────────────────────▼─────────────────────────────────────┐   │
│  │  CAPA 3: Modelos Predictivos                                 │   │
│  │  ┌──────────────────┐  ┌─────────────────┐                  │   │
│  │  │ Modelo Tabular   │  │ Modelo Multimodal│                  │   │
│  │  │ (LightGBM        │  │ (CNN + MLP)     │                  │   │
│  │  │  Quantile)       │  │                 │                  │   │
│  │  └────────┬─────────┘  └────────┬────────┘                  │   │
│  │           │                      │                           │   │
│  │  ┌────────▼──────────────────────▼────────┐                 │   │
│  │  │  Calibración + Explicabilidad SHAP     │                 │   │
│  │  └────────────────────┬───────────────────┘                 │   │
│  └───────────────────────┼─────────────────────────────────────┘   │
│                          │                                          │
│  ┌───────────────────────▼─────────────────────────────────────┐   │
│  │  CAPA 4: API REST + Frontend Interactivo                     │   │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐    │   │
│  │  │ FastAPI          │  │ Frontend (Leaflet + Plotly)  │    │   │
│  │  │ /predict         │  │ Mapa de calor                │    │   │
│  │  │ /explain         │  │ Panel SHAP                   │    │   │
│  │  │ /simulate        │  │ Simulador de escenarios      │    │   │
│  │  │ /cells           │  │ Comparador de zonas          │    │   │
│  │  └──────────────────┘  └──────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Principios de Diseño

- **Reproducibilidad**: configuración centralizada en `config/config.yaml` con semilla aleatoria fija (42) que garantiza resultados determinísticos en cada ejecución.
- **Modularidad**: cada componente (ETL, features, modelos, API) opera de forma independiente y expone interfaces claras, facilitando el mantenimiento y la sustitución de componentes.
- **Escalabilidad**: la arquitectura basada en H3 permite extender a nuevas ciudades sin cambios estructurales, únicamente ajustando parámetros de configuración.
- **Despliegue containerizado**: Docker y docker-compose para orquestación de API y frontend, garantizando consistencia entre entornos de desarrollo y producción.
- **Tolerancia a fallos**: cada módulo ETL maneja sus propios errores y los propaga de forma estructurada, permitiendo que el pipeline continúe con las fuentes disponibles.

---

## 3. Fuentes de Datos

El sistema integra cinco fuentes de datos complementarias, generadas sintéticamente con parámetros calibrados según condiciones del mercado ecuatoriano. Cada fuente se modela con distribuciones estadísticas que reflejan patrones observados en estudios urbanos y reportes del sector inmobiliario.

### 3.1 Transacciones Inmobiliarias

Registros de compraventa de propiedades con las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| `transaction_id` | str | Identificador único de transacción |
| `date` | datetime | Fecha de la transacción (2019-2024) |
| `lat`, `lon` | float | Coordenadas geográficas |
| `cell_id` | str | Celda H3 resolución 8 |
| `property_type` | str | Tipo: apartment, house, lot |
| `price_usd` | float | Precio de transacción en USD |
| `area_m2` | float | Área construida o de terreno (m²) |
| `price_per_m2` | float | Precio por metro cuadrado |
| `bedrooms` | int | Número de dormitorios |
| `bathrooms` | int | Número de baños |
| `city` | str | Ciudad: Quito o Guayaquil |
| `dist_to_cbd_km` | float | Distancia al centro financiero (km) |

**Parámetros de generación**: 2,500 transacciones distribuidas entre Quito (60%) y Guayaquil (40%), con crecimiento anual del 6.5% en precios promedio. Los precios se modelan con una distribución log-normal para reflejar la asimetría característica del mercado inmobiliario, donde pocas propiedades tienen precios extremadamente altos.

### 3.2 Movilidad Urbana

Métricas de accesibilidad y conectividad por celda H3:

| Variable | Tipo | Descripción |
|---|---|---|
| `cell_id` | str | Celda H3 |
| `avg_travel_time_cbd_min` | float | Tiempo promedio al centro (min) |
| `transit_stops_count` | int | Paradas de transporte cercanas |
| `road_density_km` | float | Densidad vial (km/km²) |
| `peak_hour_speed_kmh` | float | Velocidad promedio en hora pico |
| `walkability_score` | float | Índice de caminabilidad (0-100) |

El tiempo de viaje al CBD se modela como una función de la distancia euclidiana con factores de congestión que simulan patrones de tráfico en horas pico (7-9 AM, 5-7 PM). Las paradas de transporte se distribuyen mediante un proceso Poisson no estacionario con tasa proporcional a la densidad poblacional de cada celda.

### 3.3 Imágenes Satelitales

Características derivadas de imágenes satelitales simuladas:

| Variable | Tipo | Descripción |
|---|---|---|
| `cell_id` | str | Celda H3 |
| `ndvi_mean` | float | NDVI promedio (-1 a 1) |
| `ndvi_std` | float | Desviación estándar NDVI |
| `built_up_index` | float | Índice de área construida (0-1) |
| `green_space_ratio` | float | Proporción de espacio verde (0-1) |
| `night_lights_intensity` | float | Intensidad de luces nocturnas |

Adicionalmente, se generan parches de imágenes de 128×128 píxeles con 3 canales (NDVI, densidad urbana, mezcla de uso de suelo) por celda para alimentar el modelo multimodal. El NDVI se modela más alto en zonas periféricas y menor en el centro urbano, reflejando el gradiente de vegetación típico de ciudades andinas.

### 3.4 Servicios Urbanos

Densidad y proximidad a servicios e infraestructura:

| Variable | Tipo | Descripción |
|---|---|---|
| `cell_id` | str | Celda H3 |
| `schools_count` | int | Escuelas en la celda |
| `hospitals_count` | int | Hospitales y centros de salud |
| `parks_count` | int | Parques y áreas recreativas |
| `shopping_count` | int | Centros comerciales |
| `bank_count` | int | Entidades bancarias |
| `restaurants_count` | int | Restaurantes |
| `nearest_school_km` | float | Distancia al colegio más cercano |
| `nearest_hospital_km` | float | Distancia al hospital más cercano |
| `nearest_park_km` | float | Distancia al parque más cercano |

Los puntos de interés se distribuyen mediante un proceso Poisson no estacionario con intensidad variable según la zona. Las distancias al servicio más cercano se calculan usando la distancia Haversine entre centroides de celdas H3, proporcionando una aproximación realista de la accesibilidad espacial.

### 3.5 Indicadores Socioeconómicos

Variables demográficas y económicas por celda:

| Variable | Tipo | Descripción |
|---|---|---|
| `cell_id` | str | Celda H3 |
| `population_density` | float | Densidad poblacional (hab/km²) |
| `median_income_usd` | float | Ingreso mediano estimado (USD/mes) |
| `education_index` | float | Índice de educación (0-1) |
| `employment_rate` | float | Tasa de empleo (0-1) |
| `internet_penetration` | float | Penetración de internet (0-1) |
| `crime_index` | float | Índice de percepción de seguridad (0-100) |

Estos indicadores presentan correlaciones espaciales realistas: el ingreso mediano disminuye con la distancia al CBD, la tasa de empleo se correlaciona positivamente con la densidad de servicios, y el índice de crimen presenta un gradiente centro-periferia que refleja patrones observados en ciudades latinoamericanas.

---

## 4. Pipeline ETL Geoespacial

El pipeline ETL se orquesta mediante `scripts/run_etl.py`, que ejecuta secuencialmente los cinco módulos de extracción, valida la integridad de los datos y construye el conjunto de features final. Cada módulo es independiente y puede ejecutarse de forma aislada.

### 4.1 Módulos ETL

#### 4.1.1 `etl_transactions.py`

Genera transacciones inmobiliarias sintéticas distribuidas espacialmente dentro de los bounding boxes de Quito y Guayaquil. Cada transacción se asigna a una celda H3 resolución 8 mediante la función `latlng_to_cell`. El precio se modela con:

- **Componente base** según tipo de propiedad (apartamento: 40K-200K, casa: 80K-400K, terreno: 30K-150K) con distribución log-normal.
- **Factor de crecimiento temporal**: `precio_base × (1 + 0.065) ^ años_transcurridos`, reflejando la apreciación histórica del 6.5% anual.
- **Factor espacial**: modulación según distancia al CBD con un gradiente aproximado del 1.5% por km de distancia.
- **Ruido gaussiano**: ±10% para reflejar variabilidad individual y características no observables.

El módulo valida la integridad de los datos verificando: ausencia de nulos en columnas críticas, precios positivos, coordenadas dentro de rangos válidos para Ecuador, y consistencia entre `price_usd`, `area_m2` y `price_per_m2`.

#### 4.1.2 `etl_mobility.py`

Simula métricas de movilidad correlacionadas con la distancia al centro urbano. El tiempo de viaje al CBD se modela como función de la distancia Haversine con factores de congestión en horas pico. La velocidad en hora pico se simula con una reducción del 40-60% respecto a condiciones libres. La caminabilidad se calcula como un índice compuesto que incorpora densidad de intersecciones, longitud de aceras y mezcla de usos de suelo.

#### 4.1.3 `etl_satellite.py`

Genera características satelitales (NDVI, índice construido, luces nocturnas) mediante simulación estocástica. El NDVI se modela más alto en zonas periféricas (0.4-0.7) y menor en el centro urbano (0.1-0.3). El índice de área construida sigue el patrón inverso. Las luces nocturnas sirven como proxy de actividad económica y densidad urbana. Los parches de imágenes RGB (128×128×3) se generan como arrays sintéticos con patrones correlacionados con las características de la celda.

#### 4.1.4 `etl_services.py`

Distribuye puntos de interés (escuelas, hospitales, parques, centros comerciales, bancos, restaurantes) mediante un proceso Poisson no estacionario con intensidad variable según la zona. Las zonas centrales tienen mayor densidad de servicios que las periféricas. Las distancias al servicio más cercano se calculan usando la distancia Haversine entre centroides de celdas H3.

#### 4.1.5 `etl_socioeconomic.py`

Genera indicadores socioeconómicos con correlaciones espaciales realistas. El ingreso mediano se modela como una función decreciente de la distancia al CBD con ruido log-normal. La tasa de empleo se correlaciona positivamente con la densidad de servicios. El índice educativo se modela con una distribución beta ajustada según el estrato socioeconómico de cada celda. El índice de crimen presenta un gradiente que refleja mayor inseguridad en zonas de transición entre estratos socioeconómicos.

### 4.2 Orquestación

```bash
python scripts/run_etl.py
```

El orquestador ejecuta los siguientes pasos:

1. Carga la configuración desde `config/config.yaml`
2. Ejecuta cada módulo ETL en orden de dependencias: transacciones → movilidad → satélite → servicios → socioeconómico
3. Valida la integridad de los datos generados en cada paso
4. Almacena resultados en `data/processed/` en formato CSV y Parquet (con fallback automático)
5. Ejecuta el módulo de feature engineering que integra las cinco fuentes
6. Genera un reporte de ejecución con métricas y tiempos

El orquestador maneja errores de forma graceful: si un módulo falla, registra el error y continúa con los módulos restantes, reportando al final qué módulos tuvieron éxito y cuáles fallaron.

### 4.3 Formatos de Salida

Cada módulo ETL produce dos formatos de salida:

- **Parquet** (preferido): formato columnar con compresión Snappy, optimizado para análisis posterior.
- **CSV** (fallback): formato de texto universal, utilizado cuando PyArrow no está disponible.
- **GeoJSON** (solo transacciones): para visualización en SIG y compatibilidad con herramientas geoespaciales.

---

## 5. Rejilla Hexagonal H3

### 5.1 Justificación

El sistema utiliza Uber H3 (Hierarchical Hexagonal Geospatial Indexing) como sistema de referencia espacial por las siguientes razones:

1. **Uniformidad**: las celdas hexagonales tienen área aproximadamente uniforme, eliminando el sesgo de mallas rectangulares cerca de los polos y garantizando comparabilidad entre celdas.
2. **Vecindad consistente**: cada hexágono tiene exactamente 6 vecinos equidistantes, facilitando cálculos de autocorrelación espacial, operaciones de suavizado y análisis de difusión urbana.
3. **Jerarquía multi-escala**: el sistema H3 permite cambiar de resolución manteniendo relaciones padre-hijo predictibles, habilitando análisis multi-escala sin reprocesamiento.
4. **Indexación eficiente**: las celdas se representan como strings compactos (15 caracteres), optimizando almacenamiento, joins espaciales y consultas.
5. **Adopción ampliada**: H3 es utilizado por Uber, Cityzen, Unacast y múltiples plataformas urbanas, garantizando compatibilidad con herramientas del ecosistema.

### 5.2 Configuración

- **Resolución**: 8 (área ≈ 0.74 km², arista ≈ 0.74 km)
- **Ciudades**: Quito (≈30 celdas), Guayaquil (≈25 celdas)
- **Total**: aproximadamente 55 celdas con datos

Esta resolución ofrece un balance óptimo entre granularidad espacial (suficiente para capturar dinámicas de barrio) y volumen de datos por celda (suficientes transacciones para entrenamiento robusto). A resolución 8, cada celda cubre aproximadamente 74 hectáreas, comparable al tamaño de un barrio típico en ciudades ecuatorianas.

### 5.3 Generación de Celdas

Las celdas se generan mediante muestreo con sesgo normal hacia el CBD de cada ciudad. Para cada ciudad:

1. Se define un bounding box que cubre el área metropolitana.
2. Se generan puntos aleatorios con distribución normal centrada en el CBD.
3. Cada punto se convierte a un identificador H3 mediante `h3.latlng_to_cell(lat, lon, 8)`.
4. Se eliminan duplicados hasta alcanzar el número objetivo de celdas.

Cuando la librería H3 no está disponible, se utiliza un fallback determinista basado en hash SHA-256 que genera identificadores hexadecimales de 15 caracteres, manteniendo la estructura del sistema pero sin las operaciones espaciales nativas de H3.

### 5.4 Agregación

Los datos a nivel de transacción se agregan a nivel de celda H3 mediante:

- **Media** para precios y métricas continuas (price_per_m2, NDVI)
- **Conteo** para volúmenes transaccionales y servicios
- **Mediana** para variables con alta varianza o outliers (ingresos, precios)
- **Máximo/Mínimo** para rangos de precios y distancias
- **Desviación estándar** para métricas de volatilidad y heterogeneidad

---

## 6. Feature Engineering

El módulo `src/features/build_features.py` integra las cinco fuentes de datos en una tabla unificada a nivel de celda H3, generando features derivados que capturan dinámicas urbanas complejas.

### 6.1 Variable Objetivo: Valorización Anualizada

La variable objetivo (target) es la **tasa de valorización anualizada** por celda H3, calculada como:

```
Valorización anualizada = (P_{t+T} / P_t)^(1/T) - 1
```

Donde:
- `P_t` = precio promedio por m² en el período inicial (año t)
- `P_{t+T}` = precio promedio por m² en el período final (año t+T)
- `T` = número de años entre los períodos (horizonte temporal)

Esta fórmula captura la tasa compuesta de crecimiento del precio inmobiliario en cada celda, permitiendo comparar zonas con diferentes puntos de partida y horizontes temporales. Un valor de 0.065 indica una valorización del 6.5% anual compuesto.

### 6.2 Features Temporales

- **Tendencia de precios**: pendiente de la regresión lineal de `price_per_m2` vs. tiempo (años). Captura la dirección y velocidad de cambio.
- **Volatilidad**: desviación estándar de los precios normalizada por la media (coeficiente de variación). Refleja la estabilidad del mercado en la celda.
- **Volumen de transacciones**: número total de transacciones en el período. Indica liquidez del mercado.
- **Antigüedad mediana**: mediana de días desde la última transacción. Refleja la frescura de la información de mercado.
- **Aceleración**: segunda derivada de la tendencia de precios. Indica si la valorización se está acelerando o desacelerando.

### 6.3 Features Espaciales

- **Distancia al CBD**: distancia Haversine al centro financiero de la ciudad. Fundamental para capturar el gradiente de precios urbano.
- **Densidad de servicios**: suma total de POIs (escuelas, hospitales, parques, comercios) normalizada por área de celda.
- **Índice de accesibilidad**: combinación lineal de tiempo de viaje al CBD y densidad de transporte público.
- **Ratio de espacio verde**: `green_space_ratio` del módulo satelital, indicando calidad ambiental.
- **Entropía de uso de suelo**: diversidad de tipos de servicios (índice de Shannon), calculada como `H = -Σ p_i × ln(p_i)` donde `p_i` es la proporción del tipo de servicio i.

### 6.4 Features de Interacción

- **Movilidad × Ingreso**: interacción entre accesibilidad y poder adquisitivo, capturando el efecto combinado de conectividad y demanda.
- **Densidad × Servicios**: producto de densidad poblacional y densidad de servicios, reflejando la oferta versus demanda de infraestructura.
- **NDVI × Crimen**: interacción entre vegetación y percepción de seguridad, capturando el trade-off entre calidad ambiental y seguridad.
- **Educación × Ingreso**: producto del índice educativo e ingreso mediano, como proxy de capital humano.

### 6.5 Normalización

Las características numéricas se normalizan usando estadísticos del conjunto de entrenamiento:

- **RobustScaler** para variables con outliers (precios, ingresos, distancias) — utiliza mediana y rango intercuartílico.
- **StandardScaler** para variables simétricas (NDVI, índices) — utiliza media y desviación estándar.
- **MinMaxScaler** para variables acotadas (índices 0-1, scores 0-100) — escala al rango [0, 1].

La normalización se ajusta exclusivamente en el conjunto de entrenamiento y se aplica al conjunto de validación y test para evitar data leakage.

---

## 7. Modelos Predictivos

### 7.1 Modelo Tabular — LightGBM con Quantile Regression

#### 7.1.1 Descripción

El modelo tabular utiliza LightGBM (Light Gradient Boosting Machine), un algoritmo de boosting basado en árboles de decisión que ofrece:

- Alto rendimiento y eficiencia en entrenamiento, con complejidad O(n × log(n)) por iteración.
- Manejo nativo de variables categóricas sin requerir one-hot encoding.
- Soporte para Quantile Loss, permitiendo estimar intervalos de predicción directamente sin modelos adicionales.
- Interpretabilidad mediante importancia de características y compatibilidad con SHAP TreeExplainer.

Cuando LightGBM no está disponible, el sistema realiza un fallback automático a `GradientBoostingRegressor` de scikit-learn con pérdida quantile, manteniendo la funcionalidad a costa de menor eficiencia.

#### 7.1.2 Configuración

Se entrenan tres modelos independientes para los cuantiles 0.1 (límite inferior), 0.5 (mediana) y 0.9 (límite superior), generando intervalos de predicción del 80%.

**Parámetros del modelo LightGBM:**

| Parámetro | Valor | Descripción |
|---|---|---|
| `objective` | quantile | Función de pérdida para regresión por cuantiles |
| `alpha` | 0.1 / 0.5 / 0.9 | Cuantil objetivo |
| `n_estimators` | 600 | Número máximo de árboles |
| `learning_rate` | 0.05 | Tasa de aprendizaje |
| `num_leaves` | 63 | Número máximo de hojas por árbol |
| `min_child_samples` | 20 | Mínimo de muestras por hoja |
| `subsample` | 0.8 | Fracción de muestras por iteración |
| `colsample_bytree` | 0.8 | Fracción de features por árbol |
| `reg_alpha` | 0.1 | Regularización L1 |
| `reg_lambda` | 0.1 | Regularización L2 |
| `random_state` | 42 | Semilla para reproducibilidad |

**Validación cruzada**: TimeSeriesSplit con 5 pliegues, ordenando por latitud/longitud para simular ordenamiento espacial. Early stopping con paciencia de 50 rondas sobre el conjunto de validación.

#### 7.1.3 Features de Entrada

El modelo recibe la tabla completa de features generada por `build_features.py`, excluyendo identificadores (`cell_id`), coordenadas (`lat`, `lon`) y la variable objetivo (`annualized_valuation`). Las features se alinean automáticamente al cargar modelos persistedos, garantizando consistencia entre entrenamiento e inferencia.

#### 7.1.4 Persistencia

Los modelos se guardan como archivos `.pkl` (pickle) con nombres `tabular_q{XX}.pkl` donde XX es el percentil (10, 50, 90). Adicionalmente, se guarda un archivo `tabular_meta.json` con metadatos: nombres de features, métricas de validación cruzada, importancia de características y cuantiles entrenados.

### 7.2 Modelo Multimodal — CNN + MLP

#### 7.2.1 Descripción

El modelo multimodal combina información tabular con imágenes satelitales mediante una arquitectura de dos ramas que se fusionan en una representación conjunta:

1. **Rama CNN**: procesa parches satelitales de 128×128 píxeles con 3 canales (NDVI, densidad urbana, mezcla de uso de suelo) mediante una red convolucional profunda.
2. **Rama MLP**: procesa las características tabulares mediante capas densas con normalización batch y dropout.
3. **Fusión**: concatenación de representaciones latentes de ambas ramas, seguida de capas densas finales que producen la predicción de valorización.

El modelo requiere PyTorch. Si no está instalado, el sistema informa claramente y permite continuar con el modelo tabular únicamente.

#### 7.2.2 Arquitectura CNN

```
Entrada: 128×128×3
├── Conv2D(32, 3×3) + ReLU + BatchNorm + MaxPool(2×2)  →  64×64×32
├── Conv2D(64, 3×3) + ReLU + BatchNorm + MaxPool(2×2)  →  32×32×64
├── Conv2D(128, 3×3) + ReLU + BatchNorm + MaxPool(2×2) →  16×16×128
├── Conv2D(256, 3×3) + ReLU + BatchNorm + AdaptiveAvgPool(1×1) → 256
└── Linear(256, 64) + ReLU + Dropout(0.3)              →  64
```

La arquitectura utiliza BatchNorm para estabilizar el entrenamiento y AdaptiveAvgPool para garantizar compatibilidad con cualquier tamaño de entrada. El dropout del 30% en la capa final previene overfitting.

#### 7.2.3 Arquitectura MLP (Tabular)

```
Entrada: N features tabulares
├── Linear(N, 128) + BatchNorm + ReLU + Dropout(0.2)
├── Linear(128, 64) + BatchNorm + ReLU + Dropout(0.2)
└── Linear(64, 64) + ReLU
```

#### 7.2.4 Cabezal de Fusión

```
Concatenar [CNN(64), MLP(64)] → 128
├── Linear(128, 128) + ReLU + Dropout(0.2)
├── Linear(128, 64) + ReLU
└── Linear(64, 1)  # Valorización anualizada predicha
```

#### 7.2.5 Entrenamiento

- **Optimizador**: Adam (lr=1e-3, weight_decay=1e-4)
- **Función de pérdida**: SmoothL1Loss (huber loss), robusta a outliers
- **Scheduler**: ReduceLROnPlateau (factor=0.5, paciencia=5 épocas)
- **Batch size**: 64
- **Épocas máximas**: 50 (con restauración de mejores pesos por validation loss)
- **Split**: 80% entrenamiento, 20% validación (shuffle aleatorio con semilla 42)

El modelo guarda los mejores pesos según la pérdida de validación, garantizando que el modelo final corresponda al punto de mejor generalización durante el entrenamiento.

---

## 8. Calibración de Incertidumbre

### 8.1 Motivación

Las predicciones puntuales son insuficientes para la toma de decisiones inmobiliarias. Es fundamental cuantificar la incertidumbre asociada a cada predicción para que los usuarios puedan evaluar el riesgo y tomar decisiones informadas. El sistema implementa dos enfoques complementarios de calibración de incertidumbre.

### 8.2 MC Dropout (Modelo Multimodal)

Monte Carlo Dropout es una técnica que aproxima la inferencia bayesiana manteniendo las capas de Dropout activas durante inferencia:

1. Durante inferencia, se mantienen activas las capas de Dropout del modelo multimodal.
2. Se ejecutan N=100 pasadas forward estocásticas con diferentes máscaras de Dropout.
3. Se calcula la media y desviación estándar de las predicciones:

```
μ̂ = (1/N) × Σ ŷ_i
σ̂ = √((1/N) × Σ (ŷ_i - μ̂)²)
```

4. El intervalo de predicción al 90% es: `[μ̂ - 1.645σ̂, μ̂ + 1.645σ̂]`

La media de las pasadas MC proporciona una predicción más robusta que una sola pasada determinista, y la desviación estándar cuantifica la incertidumbre epistémica del modelo.

### 8.3 Quantile Regression (LightGBM)

El modelo tabular produce tres estimaciones por celda mediante cuantiles entrenados independientemente:

- `q_0.1`: percentil 10 (límite inferior del intervalo)
- `q_0.5`: mediana (predicción central)
- `q_0.9`: percentil 90 (límite superior del intervalo)

El intervalo de predicción al 80% es: `[q_0.1, q_0.9]`

La amplitud del intervalo `q_0.9 - q_0.1` sirve como medida directa de incertidumbre: celdas con mayor amplitud tienen predicciones menos confiables, lo cual se refleja en el frontend mediante intensidad de color.

### 8.4 Calibración Conformal (Split-Conformal)

Los intervalos raw se calibran utilizando el método de conformal prediction sobre un conjunto de calibración:

1. Calcular los scores de no-conformidad: `s_i = max(q_lower_i - y_i, y_i - q_upper_i)`
2. Determinar el cuantil empírico `q̂_α` correspondiente al nivel `(1-α)` de los scores
3. Ajustar los intervalos: `[q_lower - q̂_α, q_upper + q̂_α]`

Esto garantiza cobertura marginal del intervalo al nivel especificado (ej. 90%), es decir, que el 90% de las observaciones caerán dentro del intervalo calibrado. El método es distribution-free: no asume ninguna distribución específica de los errores.

### 8.5 API Unificada de Calibración

La función `calibrate_predictions()` proporciona una interfaz unificada que acepta:

- Arrays 2D de muestras MC → calibración por cuantiles empíricos
- DataFrames con columnas `pred_p10`, `pred_p50`, `pred_p90` → calibración conformal
- Diccionarios con `mc_samples` → calibración de muestras MC del modelo multimodal

En todos los casos, retorna un diccionario con claves `mean`, `lower`, `upper`, `std` que alimentan directamente el frontend y la API.

---

## 9. Explicabilidad con SHAP

### 9.1 Enfoque

El sistema implementa explicabilidad mediante SHAP (SHapley Additive exPlanations), que ofrece garantías teóricas de consistencia y aditividad basadas en la teoría de juegos cooperativos. SHAP asigna a cada feature un valor que representa su contribución marginal a la predicción, promediando sobre todas las posibles coaliciones de features.

### 9.2 Explicabilidad Global

A nivel global, se calculan los valores SHAP sobre todo el conjunto de datos para:

- **Importancia de características**: ordenamiento por valor SHAP absoluto promedio (`mean(|SHAP|)`)
- **Direccionalidad**: efecto positivo o negativo de cada feature en la predicción
- **Interacciones**: identificación de las interacciones más significativas entre pares de features

Para LightGBM se utiliza `TreeExplainer`, que aprovecha la estructura del árbol para un cálculo exacto en tiempo polinomial O(TLD²), donde T es el número de árboles, L es el número de hojas, y D es la profundidad máxima.

Cuando SHAP no está disponible, el sistema realiza un fallback a `feature_importances_` del modelo, normalizando los valores y broadcasting a per-row para mantener compatibilidad de interfaz.

### 9.3 Explicabilidad Local

A nivel de celda individual, se generan:

- **Force plot**: contribución de cada feature a la predicción específica, mostrando cómo features empujan la predicción desde el valor base hasta el valor final.
- **Waterfall plot**: descomposición acumulativa desde el valor base hasta la predicción final, ordenada por magnitud de contribución.
- **Dependencia parcial**: efecto de una feature específica en el rango observado, coloreada por la feature de interacción más importante.

### 9.4 Ranking de Drivers Urbanos

La función `rank_drivers()` produce un DataFrame ordenado por `mean_abs_shap` descendente, asignando un rank numérico. Este ranking identifica los factores que más influyen en la valorización urbana a nivel agregado, proporcionando insights accionables para planificadores y desarrolladores.

### 9.5 API de Explicabilidad

La función `explain()` ejecuta el pipeline completo de explicabilidad:

1. Prepara la matriz de features X
2. Calcula valores SHAP globales
3. Genera explicaciones locales por celda
4. Rankea drivers urbanos
5. Persiste resultados en formato Parquet (o CSV como fallback)

Retorna un diccionario con `shap_values`, `base_values`, `feature_names`, `driver_ranking`, `local_explanations` y `output_path`.

---

## 10. Simulador de Escenarios Urbanos

### 10.1 Motivación

El simulador de escenarios permite a los usuarios explorar cómo cambios en las variables macroeconómicas y urbanas afectarían la valorización predicha. Esto es valioso para:

- **Planificadores urbanos** evaluando el impacto de nueva infraestructura (metro, vías, servicios)
- **Inversores** analizando escenarios hipotéticos antes de comprometer capital
- **Desarrolladores inmobiliarios** estimando el efecto de mejoras en servicios y amenidades
- **Ciudadanos** entendiendo cómo cambios en su barrio podrían afectar el valor de sus propiedades

### 10.2 Metodología

El simulador recibe parámetros macroeconómicos que se traducen en ajustes sobre las features de cada celda:

1. **Tasa de interés**: afecta negativamente la valorización (mayor costo de financiamiento reduce demanda)
2. **Crecimiento del PIB**: afecta positivamente (mayor actividad económica impulsa demanda inmobiliaria)
3. **Tasa de migración**: afecta positivamente (más población aumenta demanda)
4. **Inversión en infraestructura**: mejora métricas de movilidad y servicios en celdas beneficiadas

Para cada celda, el sistema:

1. Recupera el vector de features actual `x`
2. Aplica las modificaciones derivadas del escenario: `x' = x + δ`
3. Asegura que los valores modificados estén dentro de rangos válidos
4. Ejecuta el modelo con `x'` para obtener la nueva predicción
5. Calcula el delta: `Δŷ = ŷ(x') - ŷ(x)`
6. Retorna la predicción original, la ajustada y el cambio absoluto y porcentual

### 10.3 Escenarios Predefinidos

- **Nueva línea de metro**: reduce `avg_travel_time_cbd_min` en 30% para celdas cercanas al trazado
- **Mejora educativa**: incrementa `schools_count` en 3 y mejora `education_index` en 0.1
- **Desarrollo de parques**: incrementa `parks_count` en 2 y `green_space_ratio` en 0.1
- **Gentrificación**: incrementa `median_income_usd` en 20% y `restaurants_count` en 50%

### 10.4 Escenarios Personalizados

El endpoint `/simulate` acepta modificaciones arbitrarias sobre cualquier feature tabular mediante un diccionario de pares `{feature: nuevo_valor}`. El sistema valida que las features existan y que los nuevos valores estén dentro de rangos plausibles.

---

## 11. API REST

### 11.1 Tecnología

La API se implementa con **FastAPI** (Python), ofreciendo:

- Documentación automática OpenAPI/Swagger en `/docs` y ReDoc en `/redoc`
- Validación de tipos con Pydantic v2
- Soporte asíncrono nativo para endpoints de alta concurrencia
- Alto rendimiento, comparable con Node.js y Go

### 11.2 Endpoints

| Endpoint | Método | Descripción | Parámetros |
|---|---|---|---|
| `/health` | GET | Verificación de estado del servicio | — |
| `/predict` | POST | Predicción de valorización para una celda | `cell_id`, `model` (opcional), `horizon_years` (opcional) |
| `/explain` | POST | Explicación SHAP para una celda | `cell_id`, `model` (opcional) |
| `/simulate` | POST | Simulación de escenario | `interest_rate`, `gdp_growth`, `migration_rate`, `infrastructure_investment` |
| `/cells` | GET | Listado de celdas disponibles | `city` (opcional) |

### 11.3 Esquemas Pydantic

Los esquemas se definen en `src/api/schemas.py` utilizando Pydantic BaseModel:

- **PredictionResponse**: `cell_id`, `predicted_valuation`, `lower_bound`, `upper_bound`, `confidence`
- **ExplanationResponse**: `cell_id`, `shap_values`, `base_value`, `top_drivers`
- **ScenarioRequest**: `interest_rate`, `gdp_growth`, `migration_rate`, `infrastructure_investment`
- **ScenarioResponse**: `adjusted_valuations`, `impact_summary`
- **MapResponse**: `bbox`, `year`, `cells` (lista de MapCell con lat, lon, predicted_valuation)

### 11.4 CORS y Seguridad

- CORS habilitado para el frontend en `http://localhost:3000`
- Validación estricta de inputs mediante Pydantic con tipos y restricciones de rango
- Manejo centralizado de errores con respuestas JSON estructuradas
- Logging de requests para auditoría y debugging

### 11.5 Despliegue

La API se despliega dentro de un contenedor Docker con configuración parametrizable mediante variables de entorno:

```env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
```

---

## 12. Frontend Interactivo

### 12.1 Tecnologías

- **Mapa**: Leaflet.js 1.9 con tiles de OpenStreetMap y plugin de heatmap
- **Visualizaciones**: Plotly.js 2.27 para gráficos SHAP, comparativos y de escenarios
- **Estilos**: Tailwind CSS 3 con design system personalizado basado en variables CSS
- **Sin framework JS**: JavaScript vanilla para máxima portabilidad y mínimas dependencias

### 12.2 Componentes Principales

1. **Mapa principal**: mapa Leaflet con capa de heatmap mostrando valorización predicha por celda H3. Toggle de capas para activar/desactivar heatmap y marcadores. Selector de horizonte temporal (6, 12, 24, 36 meses).

2. **Panel de detalles**: al hacer clic en una celda, muestra predicción, intervalo de confianza y métricas clave. Panel deslizable desde el lateral derecho.

3. **Tab de Explicabilidad**: selector de celda + botón de análisis. Muestra waterfall plot de SHAP y gráfico de importancia de características. Tabla resumen con factores positivos y negativos.

4. **Simulador de escenarios**: controles deslizantes para tasa de interés, crecimiento del PIB, migración e inversión en infraestructura. Gráfico comparativo de escenario base vs. simulado. Tabla de cambios detallados por celda.

5. **Comparador de zonas**: selección de múltiples celdas mediante chips. Gráfico de barras comparativo de valorización. Tabla con métricas lado a lado.

### 12.3 Flujo de Usuario

1. El usuario abre la aplicación y ve el mapa con todas las celdas coloreadas según valorización predicha.
2. Puede alternar entre mapa, explicabilidad, simulador y comparador mediante tabs.
3. Al hacer clic en una celda del mapa, se muestra el panel de detalles con predicción e intervalo.
4. En la tab de explicabilidad, selecciona una celda y obtiene el análisis SHAP completo.
5. En el simulador, ajusta variables macroeconómicas y observa el impacto en las predicciones.
6. En el comparador, selecciona varias celdas para análisis comparativo.

---

## 13. Métricas de Evaluación

### 13.1 Métricas de Precisión

| Métrica | Descripción | Fórmula |
|---|---|---|
| **MAE** | Error Absoluto Medio — interpretabilidad directa | `(1/n) × Σ|y_i - ŷ_i|` |
| **RMSE** | Raíz del Error Cuadrático Medio — penaliza errores grandes | `√((1/n) × Σ(y_i - ŷ_i)²)` |
| **R²** | Coeficiente de determinación — varianza explicada | `1 - SS_res/SS_tot` |
| **MAPE** | Error Porcentual Absoluto Medio — error relativo | `(1/n) × Σ|y_i - ŷ_i|/|y_i|` |

### 13.2 Métricas de Calibración

| Métrica | Descripción |
|---|---|
| **Coverage** | Proporción de observaciones dentro del intervalo predicho. Ideal: coincide con el nivel nominal (80%, 90%). |
| **Interval width** | Amplitud promedio del intervalo. Más estrecho = más informativo, siempre que la cobertura sea correcta. |
| **CWC** | Coverage Width Criterion — balance entre cobertura y amplitud. Penaliza intervalos que no alcanzan la cobertura nominal. |

### 13.3 Métricas de Ranking

| Métrica | Descripción |
|---|---|
| **Spearman ρ** | Correlación de ranking entre predicción y valor real. Evalúa si el modelo ordena correctamente las celdas. |
| **NDCG@10** | Normalized Discounted Cumulative Gain para top-10 celdas. Evalúa la calidad del ranking en las posiciones más visibles. |

### 13.4 Estrategias de Validación

- **Validación cruzada temporal**: entrenamiento en 2019-2022, validación en 2023, test en 2024. Refleja el escenario real de uso: predecir el futuro con datos del pasado.
- **Validación cruzada espacial**: Leave-One-City-Out para evaluar generalización entre Quito y Guayaquil.
- **Bootstrap**: 1000 remuestreos con reemplazo para calcular intervalos de confianza en las métricas.
- **Validación por cuantiles**: métricas calculadas separadamente por tercil de valorización (bajo, medio, alto) para detectar sesgos del modelo.

---

## 14. Limitaciones y Trabajo Futuro

### 14.1 Limitaciones Actuales

1. **Datos sintéticos**: los datos generados simulan patrones realistas pero no sustituyen datos reales. Las correlaciones y distribuciones pueden diferir del mercado ecuatoriano real, particularmente en eventos no anticipados (crisis económicas, desastres naturales, cambios regulatorios).

2. **Cobertura geográfica**: limitada a Quito y Guayaquil. Otras ciudades ecuatorianas (Cuenca, Loja, Ambato, Santo Domingo) requieren adaptación de parámetros y validación de supuestos.

3. **Resolución H3**: la resolución 8 (~0.74 km²) puede ser demasiado gruesa para capturar dinámicas micro-urbanas en barrios pequeños o demasiado fina para análisis macro-regionales.

4. **Modelo multimodal**: las imágenes satelitales son simuladas con arrays sintéticos. Con datos reales de Sentinel-2 o Landsat, el modelo podría capturar patrones estacionales, de uso de suelo y de calidad urbana más precisos.

5. **Factores externos**: el modelo no incorpora variables macroeconómicas dinámicas (inflación, tasas de interés del Banco Central, PIB sectorial) que afectan el mercado inmobiliario de manera significativa.

6. **Estacionalidad**: la versión actual no modela estacionalidad intra-anual en precios, la cual puede ser relevante en mercados con ciclos de construcción definidos.

7. **Validación temporal**: con datos sintéticos, la validación cruzada temporal no refleja verdadera capacidad de generalización temporal. Se requieren datos reales multi-año para validar adecuadamente.

8. **Causalidad**: el modelo captura correlaciones, no relaciones causales. Las recomendaciones basadas en SHAP deben interpretarse como asociaciones, no como intervenciones causales garantizadas.

### 14.2 Trabajo Futuro

1. **Integración de datos reales**: conectar con registros del Municipio del Distrito Metropolitano de Quito (MDMQ) y del Municipio de Guayaquil para transacciones reales. Adaptar los módulos ETL para consumir APIs municipales o archivos CADASTRALES.

2. **Imágenes satelitales reales**: integrar Sentinel-2 (gratuito, 10m de resolución, revisita 5 días) mediante Google Earth Engine o Sentinel Hub. Procesar NDVI, NDBI y índices de textura urbana a partir de imágenes reales.

3. **Modelado temporal**: implementar modelos secuenciales (LSTM, Transformer temporal) que capturen dinámicas temporales por celda, permitiendo proyecciones multi-paso.

4. **Más ciudades**: extender a Cuenca, Santo Domingo, Manta y Loja con calibración de parámetros específicos por ciudad.

5. **Variables macroeconómicas**: incorporar tasas de interés del Banco Central del Ecuador, inflación, indicadores de construcción y encuestas de expectativas de mercado.

6. **Modelos generativos**: explorar GANs o modelos de difusión para augmentación de datos en celdas con pocas transacciones, mejorando la cobertura espacial del modelo.

7. **Feedback de usuarios**: implementar mecanismo de feedback donde usuarios validen predicciones, habilitando aprendizaje activo y mejora continua del modelo.

8. **API GraphQL**: complementar la API REST con GraphQL para consultas flexibles y eficientes, permitiendo a los clientes solicitar exactamente los campos que necesitan.

9. **Aplicación móvil**: desarrollar aplicación móvil para consulta inmobiliaria en campo, con geolocalización y realidad aumentada para visualizar predicciones sobre propiedades físicas.

10. **Inferencia causal**: avanzar de explicabilidad correlacional (SHAP) a inferencia causal mediante DoWhy o EconML, permitiendo responder preguntas contrafactuales del tipo "¿Qué pasaría si se construyera un hospital en esta celda?"

---

## 15. Referencias

1. Uber Technologies. (2023). *H3: Hexagonal Hierarchical Geospatial Indexing System*. https://h3geo.org

2. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems, 30.

3. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems, 30.

4. Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. Proceedings of the 33rd International Conference on Machine Learning (ICML).

5. Koenker, R., & Bassett, G. (1978). *Regression Quantiles*. Econometrica, 46(1), 33-50.

6. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv preprint arXiv:2107.07511.

7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

8. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.

9. Instituto Nacional de Estadística y Censos (INEC). (2024). *Anuario de Estadísticas de Construcción*. Quito, Ecuador.

10. Banco Central del Ecuador. (2024). *Reporte del Sector Inmobiliario*. Quito, Ecuador.

11. Pérez, L., & Salazar, M. (2023). *Dinámica del mercado inmobiliario en Quito y Guayaquil: un análisis espacial*. Revista Ecuatoriana de Economía, 18(2), 45-72.

12. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Springer Science & Business Media.

13. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

*Documento generado como parte del proyecto Radar de Valorización Urbana. Para consultas técnicas, referirse al README.md o a la documentación de la API en `/docs`.*
