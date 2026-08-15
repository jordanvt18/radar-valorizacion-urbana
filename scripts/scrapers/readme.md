# Scrapers Inmobiliarios — Radar de Valorización Urbana

Recopilación **ética** de avisos de venta de inmuebles en Ecuador desde los
principales portales inmobiliarios, para alimentar el componente de precios
de oferta del proyecto.

| Fuente | Sitio | Qué extrae |
|---|---|---|
| Plusvalía | [plusvalia.com](https://www.plusvalia.com) | título, precio USD, área, habitaciones, baños, ubicación, URL |
| Properati | [properati.com.ec](https://www.properati.com.ec) | título, precio USD, área, habitaciones, ubicación, URL |
| OLX | [olx.com.ec](https://www.olx.com.ec) | título, precio USD, ubicación, URL |
| RE/MAX | [remax.com.ec](https://www.remax.com.ec) | título, precio USD, área, ubicación, URL |

---

## Requisitos

- Python 3.10+
- `requests` y `beautifulsoup4` (no están en `requirements.txt` del repo;
  instálelos por separado):

```bash
pip install requests beautifulsoup4
```

Si falta alguna dependencia, los scripts terminan con un mensaje claro.

## Cómo ejecutar

Ejecutar siempre desde la raíz del repositorio (`repo/`).

### Scraper individual

```bash
# Plusvalía — Quito, 3 páginas
python scripts/scrapers/scrape_plusvalia.py --city quito --pages 3

# Properati — Guayaquil, 5 páginas, salida explícita
python scripts/scrapers/scrape_properati.py --city guayaquil --pages 5 --output data/raw/scraped

# OLX — Cuenca, 2 páginas
python scripts/scrapers/scrape_olx.py --city cuenca --pages 2

# RE/MAX — Quito, 3 páginas
python scripts/scrapers/scrape_remax.py --city quito --pages 3
```

Ciudades soportadas: `quito`, `guayaquil`, `cuenca`.

Opciones comunes:

| Opción | Default | Descripción |
|---|---|---|
| `--city` | `quito` | Ciudad a scrapear |
| `--pages` | `2–3` | Número de páginas de resultados |
| `--output` | `data/raw/scraped` | Directorio de salida |
| `--delay-min` / `--delay-max` | `2.0` / `5.0` | Retraso aleatorio entre peticiones (segundos) |
| `--max-retries` | `3` | Reintentos ante errores de red / HTTP 429 / 5xx |
| `--verbose` | — | Logging detallado |

> `--delay-min` no puede ser menor a 1 segundo: es la política de cortesía
> mínima del proyecto.

### Todos los scrapers (orquestador)

```bash
python scripts/scrapers/run_all.py
python scripts/scrapers/run_all.py --cities quito,guayaquil,cuenca --pages 2
python scripts/scrapers/run_all.py --skip olx            # omitir una fuente
python scripts/scrapers/run_all.py --pages 3 --verbose
```

El orquestador:

1. Ejecuta las fuentes en secuencia por ciudad.
2. Si una fuente falla (p. ej. OLX bloquea el acceso), registra el error y
   continúa con las demás.
3. Agrega todo en `data/raw/scraped/all_listings.csv` (+ `.json`).
4. Imprime un resumen: total de avisos, conteo por fuente y rangos de precio.

---

## Política de scraping ético

Este proyecto recopila datos de forma responsable. Reglas obligatorias
(implementadas en `common.py` y en cada scraper):

1. **robots.txt**: antes de cada petición se consulta `robots.txt`
   (`urllib.robotparser`, con caché de 1 hora por sitio). Si el sitio
   prohíbe el acceso a una ruta, esa ruta se omite.
2. **User-Agent claro**: cada petición identifica al proyecto
   (`RadarValorizacionUrbana/0.1 (+repositorio; contacto)`). No se suplanta
   a un navegador.
3. **Retrasos corteses**: entre peticiones se espera un tiempo aleatorio de
   2–5 segundos (configurable, mínimo 1 s).
4. **Límites de tasa**: ante HTTP 429 se respeta la cabecera `Retry-After`
   (o 30 s de respaldo) y se reintenta; ante 5xx se reintenta con espera.
5. **Sin evasión de anti-bot**: si un sitio bloquea el acceso (403, captcha,
   etc.) el scraper **aborta** con un mensaje claro. Prohibido: resolver
   captchas, rotar IPs/proxies, headless browsers, o cualquier técnica de
   evasión.
6. **Volumen moderado**: se scrapean solo las primeras páginas de resultados
   (por defecto 2–3). Para datasets grandes, ejecutar en varias sesiones
   espaciadas en el tiempo.

### OLX (caso especial)

OLX Ecuador implementa mecanismos anti-bot agresivos. El scraper de OLX
detecta el bloqueo y termina de forma controlada con un mensaje explicativo
(sin intentar evadirlo). Si OLX bloquea, considere:

- Ejecutar en otro horario o red.
- Usar los datos manualmente (sin automatización).
- Contactar al portal para un canal oficial de datos.

---

## Formato de salida

Cada fuente genera dos archivos en `data/raw/scraped/`:

- `{fuente}.csv` — UTF-8 con BOM (compatible con Excel en español).
- `{fuente}.json` — lista de objetos.

Columnas estandarizadas:

| Campo | Tipo | Descripción |
|---|---|---|
| `title` | text | Título del aviso |
| `price_usd` | float | Precio en dólares (parsed de formatos `"$ 120.000"`, `"USD 120,000"`, `"120000"`) |
| `price_per_m2` | float | Precio por m² (calculado si faltaba) |
| `area_m2` | float | Superficie en m² (`"156 m²"`, `"505.53 m2"`) |
| `bedrooms` | int/float | Habitaciones |
| `bathrooms` | int/float | Baños |
| `location` | text | Ubicación/sector |
| `lat` / `lon` | float | Coordenadas (None si el listado no las expone) |
| `url` | text | URL del aviso (clave de deduplicación) |
| `source` | text | `plusvalia`, `properati`, `olx`, `remax` |
| `scraped_at` | datetime | Marca de tiempo UTC de recolección |
| `property_type` | text | `casa`, `apartamento`, `terreno`, `oficina`, … (inferido) |

Deduplicación: los re-ejecuciones se fusionan con lo ya guardado usando la
URL normalizada como clave (los avisos sin URL usan `source:título`).

### Agregado

`run_all.py` genera además:

- `data/raw/scraped/all_listings.csv` — todos los avisos únicos de todas las
  fuentes.
- `data/raw/scraped/all_listings.json` — lo mismo en JSON.

---

## Estructura del código

```
scripts/scrapers/
├── __init__.py
├── common.py              # utilidades compartidas (sesión, robots.txt, retrasos, parsing, persistencia)
├── scrape_plusvalia.py    # plusvalia.com
├── scrape_properati.py    # properati.com.ec
├── scrape_olx.py          # olx.com.ec (fallo controlado ante anti-bot)
├── scrape_remax.py        # remax.com.ec
├── run_all.py             # orquestador + agregación + resumen
└── README.md              # este documento
```

Notas de mantenimiento:

- **Plusvalía** pagina con `?pagina=N` (respaldo) y expone enlaces
  "siguiente" (`find_next_page`).
- **Properati** pagina con `?page=N`.
- **OLX** pagina con `?page=N`.
- **RE/MAX** usa `?page=N` con base 0 (`pageSize=24`); los códigos de
  ubicación son códigos cantonales INEC (Quito=1701, Guayaquil=0901,
  Cuenca=0101).

---

## Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `Falta una dependencia` | `requests`/`beautifulsoup4` no instalados | `pip install requests beautifulsoup4` |
| `bloqueó el acceso (…)` | Anti-bot del sitio | No evadirlo; ejecutar más tarde o usar canal oficial |
| `no se encontraron tarjetas` | El sitio cambió su HTML | Actualizar `CARD_SELECTORS` / `_TITLE_SELECTORS` del scraper |
| `robots.txt impide acceder` | El sitio prohíbe la ruta | El scraper omite esa ruta (comportamiento correcto) |
| `HTTP 429` | Demasiadas peticiones | El scraper espera `Retry-After` y reintenta; aumentar `--delay-min/max` |

---

## Aviso legal

- **Términos de servicio**: revise y respete los términos de servicio de cada
  portal. Este código es una herramienta de investigación; el operador es
  responsable de usarla conforme a la ley y a las condiciones de cada sitio.
- **Datos personales**: los avisos son información pública de oferta
  inmobiliaria; no recopile ni publique datos personales de contactos.
- **Uso académico**: los datos recopilados deben usarse para fines de
  investigación y análisis urbano, no para re-publicación comercial de los
  contenidos de los portales.

## Nota sobre cambios de HTML

Los portales cambian su estructura HTML con frecuencia. Estos scrapers usan
múltiples selectores y respaldos, pero **pueden dejar de funcionar** si un
sitio rediseña sus páginas. El comportamiento esperado ante eso es un
registro de advertencia y una terminación limpia (nunca un crash), y la
actualización de los selectores en `CARD_SELECTORS`, `_TITLE_SELECTORS` y
`_DETAIL_LINK_SELECTORS`.

---

*Radar de Valorización Urbana — proyecto de predicción de valorización
inmobiliaria para Quito y Guayaquil, Ecuador.*
