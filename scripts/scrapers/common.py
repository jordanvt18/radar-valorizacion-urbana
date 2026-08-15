"""
Utilidades compartidas para los scrapers del Radar de Valorización Urbana.

Recopilación ÉTICA de datos: respeto a robots.txt, retrasos corteses
(2-5 s) entre peticiones, manejo de límites de tasa (HTTP 429), User-Agent
identificable y persistencia estandarizada en CSV/JSON.

Funciones principales:
    - get_session()           Sesión requests con User-Agent realista.
    - polite_sleep()          Retraso aleatorio entre peticiones.
    - respect_robots()        Verifica robots.txt antes de cada fetch.
    - fetch_page()            GET con reintentos y manejo de 429.
    - is_blocked_response()   Detecta respuestas anti-bot (sin evadirlas).
    - parse_price()           Parsea formatos de precio ecuatorianos.
    - parse_area()            Parsea superficies ("156 m²", "505.53 m2").
    - normalize_record()      Estandariza campos de un aviso.
    - save_results()          Guarda en data/raw/scraped/ (CSV + JSON).
    - load_existing()         Carga avisos previos para deduplicar.
    - find_next_page()        Detecta el enlace "siguiente" de una página.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
    BS4_AVAILABLE = True
except ImportError:  # pragma: no cover - se valida en cada scraper
    BS4_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, name: str = "radar_scrapers") -> logging.Logger:
    """Configura el logging raíz y devuelve el logger del paquete de scrapers."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


logger = setup_logging()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PROJECT_NAME = "Radar de Valorización Urbana"
PROJECT_REPO = "https://github.com/jordanvt18/radar-valorizacion-urbana"

# User-Agent identificable: un bot ético se presenta, no se hace pasar por navegador.
USER_AGENT = (
    f"RadarValorizacionUrbana/0.1 (+{PROJECT_REPO}; "
    "scraping ético para investigación urbana; respeta robots.txt; "
    "contacto: radar-valorizacion@example.com)"
)

DEFAULT_TIMEOUT = 30.0

# Campos estandarizados de salida (orden del CSV).
FIELDS = [
    "title",
    "price_usd",
    "price_per_m2",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "location",
    "lat",
    "lon",
    "url",
    "source",
    "scraped_at",
    "property_type",
]

# Alias aceptados al normalizar registros (inglés/español).
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "title": ("title", "titulo", "name", "heading", "aviso"),
    "price_usd": ("price_usd", "price", "precio", "usd", "price_usd_raw"),
    "price_per_m2": ("price_per_m2", "precio_m2", "price_m2", "usd_per_m2", "usd_m2", "precio_por_m2"),
    "area_m2": (
        "area_m2", "area", "surface", "superficie", "m2", "total_area", "area_total",
        "area_terreno", "area_construida", "built_area", "covered_area", "sizes",
    ),
    "bedrooms": ("bedrooms", "rooms", "habitaciones", "dormitorios", "bedroom", "bed", "num_rooms"),
    "bathrooms": ("bathrooms", "bathroom", "banos", "banos_", "baths", "bath", "num_baths"),
    "location": ("location", "ubicacion", "place", "zona", "sector", "city", "address", "direccion", "sublocation"),
    "lat": ("lat", "latitude", "latitud"),
    "lon": ("lon", "lng", "longitude", "long", "longitud"),
    "url": ("url", "link", "href", "listing_url", "detail_url", "permalink"),
    "source": ("source", "source_name", "portal", "site", "fuente"),
    "scraped_at": ("scraped_at", "scraped_at_utc", "timestamp", "date", "fecha", "collected_at"),
    "property_type": ("property_type", "tipo", "type", "property_type_es", "tipo_inmueble", "listing_type"),
}

# Marcadores típicos de páginas anti-bot (se detectan para ABORTAR con un
# mensaje claro; nunca se intenta evadirlos).
BLOCK_MARKERS = (
    "captcha",
    "access denied",
    "pardon our interruption",
    "verify you are human",
    "you have been blocked",
    "robot check",
    "cf-challenge",
    "cf-chl-",
    "cf-error-details",
    "perimeterx",
    "datadome",
    "arkoselabs",
    "blocked",
    "prove you are not a robot",
)

_ROBOTS_CACHE: Dict[str, Tuple[float, Optional[robotparser.RobotFileParser]]] = {}
_ROBOTS_CACHE_TTL = 3600.0  # 1 hora

DEFAULT_DELAY_MIN = 2.0
DEFAULT_DELAY_MAX = 5.0


# ---------------------------------------------------------------------------
# Excepciones
# ---------------------------------------------------------------------------

class SiteBlockedError(RuntimeError):
    """El sitio bloqueó el acceso (anti-bot / HTTP 403 / captcha).

    Se lanza para que el operador lo sepa con claridad; NUNCA se intenta
    evadir la protección del sitio.
    """


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Devuelve la raíz del repositorio (subiendo desde scripts/scrapers)."""
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists() or (parent / "readme.md").exists() or (parent / "data").is_dir():
            return parent
    return current.parent.parent.parent


def default_output_dir() -> Path:
    """Directorio por defecto de salida: <raíz>/data/raw/scraped."""
    return project_root() / "data" / "raw" / "scraped"


# ---------------------------------------------------------------------------
# Sesión y retrasos
# ---------------------------------------------------------------------------

def get_session(user_agent: str = USER_AGENT) -> requests.Session:
    """Crea una sesión requests con cabeceras realistas y User-Agent claro.

    El User-Agent identifica el proyecto de investigación; no se suplanta a
    un navegador.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-EC,es;q=0.9,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    )
    return session


def polite_sleep(min_seconds: float = DEFAULT_DELAY_MIN, max_seconds: float = DEFAULT_DELAY_MAX) -> float:
    """Espera un tiempo aleatorio entre peticiones (cortesía con el servidor).

    Devuelve los segundos realmente esperados.
    """
    if max_seconds < min_seconds:
        min_seconds, max_seconds = max_seconds, min_seconds
    delay = random.uniform(min_seconds, max_seconds)
    logger.debug("Retraso cortés de %.1f s", delay)
    time.sleep(delay)
    return delay


# ---------------------------------------------------------------------------
# robots.txt
# ---------------------------------------------------------------------------

def _robots_parser_for(
    scheme_netloc: str, user_agent: str, timeout: float
) -> Tuple[Optional[robotparser.RobotFileParser], bool]:
    """Obtiene (parser, ok) para un sitio, con caché de 1 hora.

    - parser == None y ok == True  -> el sitio no publica robots.txt (permitido).
    - parser == None y ok == False -> no se pudo verificar robots.txt.
    - parser != None y ok == True  -> usar parser.can_fetch().
    """
    now = time.time()
    cached = _ROBOTS_CACHE.get(scheme_netloc)
    if cached is not None and now - cached[0] < _ROBOTS_CACHE_TTL:
        return cached[1], True

    robots_url = f"{scheme_netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, timeout=timeout, headers={"User-Agent": user_agent})
    except requests.RequestException as exc:
        logger.warning("No se pudo leer robots.txt (%s): %s", robots_url, exc)
        return None, False

    if resp.status_code == 404:
        _ROBOTS_CACHE[scheme_netloc] = (now, None)
        return None, True
    if resp.status_code in (401, 403):
        logger.warning("robots.txt devolvió HTTP %d en %s", resp.status_code, robots_url)
        return None, False
    if resp.status_code != 200:
        logger.warning("robots.txt devolvió HTTP %d en %s", resp.status_code, robots_url)
        return None, False

    parser = robotparser.RobotFileParser()
    parser.parse(resp.text.splitlines())
    _ROBOTS_CACHE[scheme_netloc] = (now, parser)
    return parser, True


def respect_robots(url: str, user_agent: str = USER_AGENT, timeout: float = 10.0, fail_closed: bool = False) -> bool:
    """Verifica si robots.txt permite acceder a ``url``.

    Política por defecto (fail-open con advertencia): si robots.txt no se
    puede leer, se permite el acceso pero se registra una advertencia.
    Con ``fail_closed=True`` el acceso se deniega cuando no se puede verificar.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return True
    scheme_netloc = f"{parsed.scheme}://{parsed.netloc}"
    try:
        parser, ok = _robots_parser_for(scheme_netloc, user_agent, timeout)
    except Exception as exc:  # noqa: BLE001 - nunca debe romper el flujo
        logger.warning("Error al verificar robots.txt para %s: %s", url, exc)
        return not fail_closed
    if not ok:
        logger.warning("No se pudo verificar robots.txt para %s; continuando (fail-open).", url)
        return not fail_closed
    if parser is None:
        return True  # sin robots.txt -> permitido
    allowed = parser.can_fetch(user_agent, url)
    if not allowed:
        logger.info("robots.txt impide acceder a %s", url)
    return allowed


# ---------------------------------------------------------------------------
# Fetch con reintentos
# ---------------------------------------------------------------------------

def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
    """Lee la cabecera Retry-After (segundos o fecha HTTP) con respaldo."""
    value = response.headers.get("Retry-After")
    if not value:
        return fallback
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        # Formato fecha HTTP: Sat, 01 Jan 2026 00:00:00 GMT
        parsed = datetime.strptime(value, "%a, %d %b %Y %H:%M:%S %Z")
        wait = (parsed - datetime.utcnow()).total_seconds()
        return max(wait, 1.0)
    except ValueError:
        return fallback


def fetch_page(
    session: requests.Session,
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 3,
    retry_backoff: float = 30.0,
    check_robots: bool = True,
    user_agent: str = USER_AGENT,
) -> Optional[requests.Response]:
    """Obtiene una página con manejo de errores y límites de tasa.

    - Verifica robots.txt antes de cada petición (por defecto).
    - Reintenta con espera en HTTP 429 (respeta Retry-After) y 5xx.
    - Devuelve None si no se pudo obtener la página (no lanza).
    """
    if check_robots and not respect_robots(url, user_agent=user_agent):
        logger.warning("Omitido por robots.txt: %s", url)
        return None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
        except requests.Timeout:
            logger.warning("Timeout al obtener %s (intento %d/%d)", url, attempt, max_retries)
        except requests.RequestException as exc:
            logger.warning("Error de red al obtener %s: %s (intento %d/%d)", url, exc, attempt, max_retries)
        else:
            if response.status_code == 429:
                wait = _retry_after_seconds(response, retry_backoff)
                logger.warning(
                    "HTTP 429 (rate limit) en %s — esperando %.0f s (intento %d/%d)",
                    url, wait, attempt, max_retries,
                )
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                logger.warning(
                    "HTTP %d en %s (intento %d/%d)", response.status_code, url, attempt, max_retries
                )
                if attempt < max_retries:
                    time.sleep(retry_backoff)
                continue
            response.raise_for_status()
            return response

        if attempt < max_retries:
            time.sleep(retry_backoff)
    logger.error("No se pudo obtener %s tras %d intentos", url, max_retries)
    return None


def is_blocked_response(response: Optional[requests.Response], html_text: Optional[str] = None) -> Optional[str]:
    """Detecta una respuesta bloqueada por anti-bot. Devuelve la razón o None.

    Si devuelve algo distinto de None, el operador debe detenerse: NO se
    intenta evadir la protección (captchas, rotación de IP, etc.).
    """
    if response is None:
        return "sin respuesta del servidor"
    if response.status_code in (401, 403):
        return f"HTTP {response.status_code}"
    if not response.ok:
        return f"HTTP {response.status_code}"
    sample = html_text if html_text is not None else response.text[:50000]
    low = sample.lower()
    for marker in BLOCK_MARKERS:
        if marker in low:
            return f"página anti-bot detectada ('{marker}')"
    return None


# ---------------------------------------------------------------------------
# Parsing de números en formato ecuatoriano
# ---------------------------------------------------------------------------

def _to_float(raw: str) -> Optional[float]:
    """Convierte un número con separadores ('1.200.000', '120,5', '120000') a float.

    Heurística para formatos mixtos:
    - Punto y coma presentes -> el último de los dos es el separador decimal.
    - Solo comas: una coma con 1-2 decimales es decimal; si no, es miles.
    - Solo puntos: grupos de 3 dígitos tras el punto son miles (formato
      español de Ecuador: "120.000" = 120 mil); 1-2 dígitos son decimales.
    """
    s = raw.strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "")
        elif len(parts) == 2 and len(parts[1]) in (1, 2):
            pass  # punto decimal
        else:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_price(price_str: Any) -> Optional[float]:
    """Parsea precios en formatos ecuatorianos a USD (float).

    Acepta: "USD 120.000", "$ 120,000", "120000", "120.000", "USD 1.450",
    "87.000 USD", rangos "120.000 - 150.000" (toma el primer valor).
    """
    if price_str is None:
        return None
    if isinstance(price_str, bool):
        return None
    if isinstance(price_str, (int, float)):
        return float(price_str)
    s = str(price_str).strip()
    if not s:
        return None
    # Quitar códigos de moneda y símbolos
    s = re.sub(r"(?i)\b(usd|d[oó]lares?|d[oó]lar)\b", " ", s)
    s = s.replace("$", " ").replace("€", " ").replace("¢", " ")
    # Conservar solo dígitos, separadores y guiones
    s = re.sub(r"[^\d.,\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    # Rangos: tomar el primer valor
    if "-" in s:
        s = s.split("-")[0].strip()
        if not s:
            return None
    return _to_float(s)


def parse_area(area_str: Any) -> Optional[float]:
    """Parsea superficies: "156 m²", "505.53 m2", "1.200 m² tot.", "120,5 m2"."""
    if area_str is None:
        return None
    if isinstance(area_str, bool):
        return None
    if isinstance(area_str, (int, float)):
        return float(area_str)
    s = str(area_str).strip()
    if not s:
        return None
    match = re.search(r"(\d[\d.,]*)\s*(?:m\s*[²2]|metros?\s*(?:cuadrados?|2)?)", s, re.IGNORECASE)
    if match:
        return _to_float(match.group(1))
    match = re.search(r"\d[\d.,]*", s)
    if match:
        return _to_float(match.group(0))
    return None


def parse_int(value: Any) -> Optional[int]:
    """Parsea un entero ("4", "3 hab.", 5, "5.0") o devuelve None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _clean_number(value: Any) -> Any:
    """Convierte a int si es entero, a float si no, o None si no es numérico."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number) if number.is_integer() else number
    s = str(value).strip()
    if not s:
        return None
    number = _to_float(s)
    if number is None:
        return None
    return int(number) if number.is_integer() else number


# ---------------------------------------------------------------------------
# Extracción de características desde texto de tarjeta
# ---------------------------------------------------------------------------

def text_of(element: Any) -> Optional[str]:
    """Texto plano normalizado de un elemento BeautifulSoup (o None)."""
    if element is None:
        return None
    try:
        text = element.get_text(" ", strip=True)
    except AttributeError:
        text = str(element)
    text = " ".join(text.split())
    return text or None


def extract_features(text: Any) -> Dict[str, Optional[float]]:
    """Extrae área, habitaciones y baños de un texto de tarjeta de aviso.

    Ejemplo: "156 m² tot. 4 hab. 2 baños" -> {area_m2: 156, bedrooms: 4, bathrooms: 2}
    """
    result: Dict[str, Optional[float]] = {"area_m2": None, "bedrooms": None, "bathrooms": None}
    if not text:
        return result
    sample = " ".join(str(text).split())

    match = re.search(r"(\d[\d.,]*)\s*m\s*[²2]", sample, re.IGNORECASE)
    if match:
        result["area_m2"] = _to_float(match.group(1))

    match = re.search(r"(\d+)\s*(?:hab|habitaciones?|dormitorios?)", sample, re.IGNORECASE)
    if match:
        result["bedrooms"] = float(match.group(1))

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:baños?|banos?|bathrooms?|baths?)", sample, re.IGNORECASE)
    if match:
        result["bathrooms"] = _to_float(match.group(1))

    return result


# ---------------------------------------------------------------------------
# Normalización de registros
# ---------------------------------------------------------------------------

def infer_property_type(title: str, url: str = "") -> str:
    """Infere el tipo de inmueble a partir del título y la URL."""
    sample = f"{title} {url}".lower()
    if any(word in sample for word in ("departamento", "apartamento", "depto", "dept.")):
        return "apartamento"
    if "casa" in sample and "caseta" not in sample:
        return "casa"
    if any(word in sample for word in ("terreno", "lote", "solar")):
        return "terreno"
    if "oficina" in sample:
        return "oficina"
    if "local" in sample and "localidad" not in sample:
        return "local"
    if "bodega" in sample:
        return "bodega"
    if "quinta" in sample:
        return "quinta"
    if "suite" in sample:
        return "suite"
    if "villa" in sample:
        return "villa"
    return "unknown"


def normalize_url(url: Optional[str]) -> str:
    """Normaliza una URL para usarla como clave de deduplicación."""
    if not url:
        return ""
    value = str(url).strip().split("#")[0].rstrip("/")
    return value.lower()


def record_key(record: Dict[str, Any]) -> str:
    """Clave única de un registro: URL normalizada o 'source:título'."""
    url = normalize_url(record.get("url"))
    if url:
        return url
    title = (record.get("title") or "").strip().lower()
    source = (record.get("source") or "unknown").lower()
    return f"{source}:{title}"


def normalize_record(
    record: Dict[str, Any],
    source: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Estandariza un aviso al esquema común del proyecto.

    Campos de salida: title, price_usd, price_per_m2, area_m2, bedrooms,
    bathrooms, location, lat, lon, url, source, scraped_at, property_type.
    Acepta alias en español e inglés (ver FIELD_ALIASES).
    """
    normalized: Dict[str, Any] = {field: None for field in FIELDS}
    lowered = {str(key).strip().lower(): value for key, value in record.items()}

    # Resolver alias campo por campo
    for field in FIELDS:
        value = record.get(field)
        if value in (None, ""):
            for alias in FIELD_ALIASES.get(field, ()):
                candidate = lowered.get(alias)
                if candidate not in (None, ""):
                    value = candidate
                    break
        normalized[field] = value

    # URL absoluta
    url = normalized.get("url")
    if url:
        url = str(url).strip()
        if base_url:
            url = urljoin(base_url, url)
        normalized["url"] = url

    # Precio y área
    normalized["price_usd"] = parse_price(normalized.get("price_usd"))
    normalized["area_m2"] = parse_area(normalized.get("area_m2"))

    # Precio por m² (calculado si falta)
    price_per_m2 = parse_price(normalized.get("price_per_m2"))
    if price_per_m2 is None and normalized["price_usd"] and normalized["area_m2"]:
        price_per_m2 = round(normalized["price_usd"] / normalized["area_m2"], 2)
    normalized["price_per_m2"] = price_per_m2

    # Conteos
    normalized["bedrooms"] = _clean_number(normalized.get("bedrooms"))
    normalized["bathrooms"] = _clean_number(normalized.get("bathrooms"))

    # Coordenadas
    normalized["lat"] = _clean_number(normalized.get("lat"))
    normalized["lon"] = _clean_number(normalized.get("lon"))

    # Textos
    normalized["title"] = text_of(normalized.get("title"))
    normalized["location"] = text_of(normalized.get("location"))

    # Fuente y tipo de propiedad
    normalized["source"] = source or normalized.get("source") or "unknown"
    if not normalized.get("property_type"):
        normalized["property_type"] = infer_property_type(
            normalized["title"] or "", normalized["url"] or ""
        )

    # Marca de tiempo
    if not normalized.get("scraped_at"):
        normalized["scraped_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return normalized


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------

def load_existing(output_dir: Any, source_name: str) -> Dict[str, Dict[str, Any]]:
    """Carga avisos ya guardados (CSV o JSON) para deduplicar.

    Devuelve {clave: registro} donde la clave es ``record_key`` (URL o
    source:título).
    """
    out = Path(output_dir)
    existing: Dict[str, Dict[str, Any]] = {}
    csv_path = out / f"{source_name}.csv"
    json_path = out / f"{source_name}.json"

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row and any(row.values()):
                        existing[record_key(row)] = row
            logger.debug("Cargados %d registros previos desde %s", len(existing), csv_path)
        except (csv.Error, OSError) as exc:
            logger.warning("No se pudo leer %s: %s", csv_path, exc)
    elif json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        existing[record_key(row)] = row
            logger.debug("Cargados %d registros previos desde %s", len(existing), json_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("No se pudo leer %s: %s", json_path, exc)
    return existing


def save_results(records: List[Dict[str, Any]], output_dir: Any, source_name: str) -> List[Dict[str, Any]]:
    """Guarda avisos en data/raw/scraped/ (CSV UTF-8-BOM + JSON), deduplicando.

    Fusiona con lo ya guardado (por URL/título) y devuelve la lista final.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged = load_existing(out, source_name)
    new_count = 0
    for raw in records:
        rec = normalize_record(raw, source=source_name)
        key = record_key(rec)
        if key and key in merged:
            continue
        merged[key] = rec
        new_count += 1

    rows = list(merged.values())

    # CSV (utf-8-sig para compatibilidad con Excel en América Latina)
    csv_path = out / f"{source_name}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})

    # JSON
    json_path = out / f"{source_name}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    logger.info(
        "Guardados %d avisos de %s (%d nuevos) en %s", len(rows), source_name, new_count, csv_path
    )
    return rows


# ---------------------------------------------------------------------------
# Navegación entre páginas
# ---------------------------------------------------------------------------

def find_next_page(soup: Any, current_url: str, patterns: Tuple[str, ...] = ("pagina", "page", "p=")) -> Optional[str]:
    """Busca la URL de la siguiente página en el HTML. Devuelve None si no hay.

    Revisa: <link rel="next">, <a rel="next">, enlaces con texto
    "siguiente/next/»" y enlaces cuyo href contenga tokens de paginación.
    """
    if soup is None:
        return None
    if not BS4_AVAILABLE:
        logger.warning("BeautifulSoup no está disponible; no se puede buscar la página siguiente.")
        return None

    for selector in ('link[rel~="next"]', 'a[rel~="next"]'):
        element = soup.select_one(selector)
        href = element.get("href") if element else None
        if href:
            return urljoin(current_url, str(href))

    for anchor in soup.find_all("a", href=True):
        text = (text_of(anchor) or "").lower()
        if any(token in text for token in ("siguiente", "next", "»", "siguiente »")):
            return urljoin(current_url, str(anchor["href"]))

    current_path = urlparse(current_url).path
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if not any(token in href.lower() for token in patterns):
            continue
        resolved = urljoin(current_url, href)
        if resolved == current_url:
            continue
        if urlparse(resolved).path and urlparse(resolved).path != current_path:
            return resolved
    return None


# ---------------------------------------------------------------------------
# Utilidad de consola
# ---------------------------------------------------------------------------

def require_dependencies() -> None:
    """Valida requests/BeautifulSoup y termina con un mensaje claro si faltan."""
    missing = []
    try:
        import requests  # noqa: F401
    except ImportError:
        missing.append("requests")
    try:
        import bs4  # noqa: F401
    except ImportError:
        missing.append("beautifulsoup4")
    if missing:
        print(
            f"[ERROR] Faltan dependencias: {', '.join(missing)}.\n"
            "Instálelas con:\n"
            "    pip install requests beautifulsoup4\n"
        )
        sys.exit(1)
