"""
Scraper de RE/MAX Ecuador (remax.com.ec) — Radar de Valorización Urbana.

Extrae de las páginas de resultados de compra: título, precio (USD), área,
ubicación y URL del aviso.

Política ÉTICA: respeta robots.txt, retrasos aleatorios de 2-5 s entre
peticiones, manejo de HTTP 429 y NUNCA evasión de anti-bot.

Uso:
    python scripts/scrapers/scrape_remax.py --city quito --pages 3
    python scripts/scrapers/scrape_remax.py --city guayaquil --pages 5 --output data/raw/scraped
    python scripts/scrapers/scrape_remax.py --city cuenca --pages 2

Nota: la estructura HTML del sitio puede cambiar (RE/MAX usa una SPA de
Next.js); si no se encuentran tarjetas se registra una advertencia y se
continúa sin fallar.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permitir ejecución directa: python scripts/scrapers/scrape_remax.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from bs4 import BeautifulSoup

    from common import (
        DEFAULT_DELAY_MAX,
        DEFAULT_DELAY_MIN,
        SiteBlockedError,
        extract_features,
        fetch_page,
        get_session,
        is_blocked_response,
        polite_sleep,
        save_results,
        setup_logging,
        text_of,
        urljoin,
    )
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] Falta una dependencia: {exc}")
    print("Instale con: pip install requests beautifulsoup4")
    sys.exit(1)

logger = setup_logging(name="scrape_remax")

SOURCE_NAME = "remax"

# URLs de búsqueda verificadas (agosto 2026). La plataforma de RE/MAX Ecuador
# pagina con ?page=N (base 0) y pageSize. Los códigos de ubicación son los
# códigos cantonales INEC: Quito=1701, Guayaquil=0901, Cuenca=0101.
_CITY_CODE = {
    "quito": "1701",
    "guayaquil": "0901",
    "cuenca": "0101",
}

CITIES: Dict[str, str] = {
    city: (
        "https://www.remax.com.ec/listings/buy"
        f"?page={{page}}&pageSize=24&sort=-createdAt&in:operationId=1"
        f"&locations=in:::{code}@{city}"
    )
    for city, code in _CITY_CODE.items()
}

# Anclas de detalle de aviso en el sitio de RE/MAX.
_DETAIL_LINK_SELECTORS = (
    "a[href*='/listing/']",
    "a[href*='/propiedades/']",
    "a[href*='/property/']",
)

_CONTAINER_HINTS = ("card", "listing", "property", "item", "result", "search")

_PRICE_RE = re.compile(r"(?i)(usd\s*[\d.,]+|\$\s*[\d.,]+|[\d.,]+\s*(?:usd|d[oó]lares?))")


def _find_container(anchor: Any) -> Any:
    """Sube desde el ancla hasta el contenedor de la tarjeta (máx. 5 niveles)."""
    container = anchor
    for _ in range(5):
        parent = container.parent
        if parent is None:
            break
        classes = " ".join(parent.get("class", [])) if parent.get("class") else ""
        if any(hint in classes.lower() for hint in _CONTAINER_HINTS):
            container = parent
            break
        container = parent
    return container


def parse_listing_card(card: Any, base_url: str) -> Optional[Dict[str, Any]]:
    """Extrae un aviso de una tarjeta de RE/MAX. None si no es parseable."""
    link = None
    for selector in _DETAIL_LINK_SELECTORS:
        element = card.select_one(selector)
        if element is not None and element.get("href"):
            link = element
            break
    if link is None:
        return None

    url = urljoin(base_url, str(link["href"]))

    title = (
        text_of(card.select_one("h2, h3, [class*='title'], [class*='heading']"))
        or text_of(link)
    )

    price_element = card.select_one(
        "[class*='price'], [class*='precio'], [data-testid*='price']"
    )
    price_text = text_of(price_element) if price_element is not None else None
    if not price_text:
        match = _PRICE_RE.search(card.get_text(" ", strip=True))
        price_text = match.group(0) if match else None

    location_element = card.select_one(
        "[class*='address'], [class*='location'], [class*='place'], "
        "[class*='ubicacion'], [data-testid*='location']"
    )
    location = text_of(location_element) if location_element is not None else None

    features = extract_features(card.get_text(" ", strip=True))

    return {
        "title": title,
        "url": url,
        "price_usd": price_text,
        "location": location,
        "area_m2": features.get("area_m2"),
        "bedrooms": features.get("bedrooms"),
        "bathrooms": features.get("bathrooms"),
        "source": SOURCE_NAME,
        "property_type": None,  # lo infiere normalize_record
    }


def parse_search_page(html: str, base_url: str) -> List[Dict[str, Any]]:
    """Parsea todas las tarjetas de aviso de una página de resultados de RE/MAX."""
    soup = BeautifulSoup(html, "html.parser")

    anchors: List[Any] = []
    for selector in _DETAIL_LINK_SELECTORS:
        found = soup.select(selector)
        if found:
            anchors = found
            logger.debug("Selectores de enlace de detalle usado: %s (%d)", selector, len(found))
            break

    if not anchors:
        # Respaldo: contenedores genéricos de tarjeta
        for selector in ("[class*='listing-card']", "[class*='property-card']", "article"):
            found = soup.select(selector)
            if found:
                logger.debug("Selector de tarjetas de respaldo: %s (%d)", selector, len(found))
                cards = found
                break
        else:
            logger.warning(
                "RE/MAX: no se encontraron tarjetas ni enlaces de detalle con %s. "
                "La estructura HTML del sitio puede haber cambiado.",
                _DETAIL_LINK_SELECTORS,
            )
            return []
    else:
        seen_containers = set()
        cards: List[Any] = []
        for anchor in anchors:
            container = _find_container(anchor)
            container_key = id(container)
            if container_key in seen_containers:
                continue
            seen_containers.add(container_key)
            cards.append(container)

    records: List[Dict[str, Any]] = []
    seen_urls: set = set()
    for card in cards:
        record = parse_listing_card(card, base_url)
        if record is None:
            continue
        key = record["url"]
        if key in seen_urls:
            continue
        seen_urls.add(key)
        records.append(record)
    return records


def scrape_city(
    city: str,
    pages: int = 3,
    output_dir: Any = "data/raw/scraped",
    delay_min: float = DEFAULT_DELAY_MIN,
    delay_max: float = DEFAULT_DELAY_MAX,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Scrapea ``pages`` páginas de venta en ``city`` (RE/MAX Ecuador).

    Devuelve los registros normalizados guardados. Lanza SiteBlockedError si
    el sitio bloquea el acceso.
    """
    if city not in CITIES:
        raise ValueError(
            f"Ciudad no soportada: '{city}'. Válidas: {', '.join(sorted(CITIES))}"
        )
    pattern = CITIES[city]
    session = get_session()

    records: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for page_no in range(1, pages + 1):
        # La API de listados de RE/MAX pagina desde 0
        current_url = pattern.format(page=page_no - 1)
        logger.info("RE/MAX [%s] — página %d/%d: %s", city, page_no, pages, current_url)

        response = fetch_page(session, current_url, max_retries=max_retries)
        if response is None:
            logger.warning("No se pudo obtener %s; se detiene la paginación.", current_url)
            break

        blocked = is_blocked_response(response)
        if blocked:
            raise SiteBlockedError(
                f"RE/MAX bloqueó el acceso ({blocked}). "
                "No se intentará evadir la protección. Ejecute más tarde o "
                "use el canal oficial del portal."
            )

        page_records = parse_search_page(response.text, current_url)
        page_urls = {r["url"] for r in page_records}

        if page_no > 1 and page_urls and page_urls.issubset(seen_urls):
            logger.info("La página %d no aporta avisos nuevos; se detiene.", page_no)
            break
        if not page_urls and page_no > 1:
            logger.warning("Página %d sin avisos; se detiene la paginación.", page_no)
            break

        records.extend(page_records)
        seen_urls |= page_urls
        logger.info("RE/MAX [%s] — página %d: %d avisos (acumulado: %d)", city, page_no, len(page_records), len(records))

        if page_no < pages:
            polite_sleep(delay_min, delay_max)

    return save_results(records, output_dir, SOURCE_NAME)


def main() -> None:
    """CLI del scraper de RE/MAX Ecuador."""
    parser = argparse.ArgumentParser(
        description="Scraper de RE/MAX Ecuador (remax.com.ec) — Radar de Valorización Urbana. "
        "Recopilación ética: respeta robots.txt y usa retrasos de 2-5 s."
    )
    parser.add_argument("--city", default="quito", choices=sorted(CITIES),
                        help="Ciudad a scrapear (default: quito)")
    parser.add_argument("--pages", type=int, default=3,
                        help="Número de páginas de resultados (default: 3)")
    parser.add_argument("--output", default="data/raw/scraped",
                        help="Directorio de salida (default: data/raw/scraped)")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_DELAY_MIN,
                        help="Retraso mínimo entre peticiones en segundos (default: 2.0)")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_DELAY_MAX,
                        help="Retraso máximo entre peticiones en segundos (default: 5.0)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Reintentos por página ante errores/429 (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Logging detallado (DEBUG)")
    args = parser.parse_args()

    if args.pages < 1:
        parser.error("--pages debe ser >= 1")
    if args.delay_min < 1.0:
        parser.error("--delay-min debe ser >= 1.0 segundo (política de cortesía)")
    if args.delay_max < args.delay_min:
        parser.error("--delay-max debe ser >= --delay-min")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start = time.time()
    try:
        records = scrape_city(
            city=args.city,
            pages=args.pages,
            output_dir=args.output,
            delay_min=args.delay_min,
            delay_max=args.delay_max,
            max_retries=args.max_retries,
        )
    except SiteBlockedError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ValueError as exc:
        parser.error(str(exc))

    elapsed = time.time() - start
    logger.info("RE/MAX [%s]: %d avisos guardados en %s (%.1f s)",
                args.city, len(records), Path(args.output), elapsed)


if __name__ == "__main__":
    main()
