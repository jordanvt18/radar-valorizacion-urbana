"""
Scraper de Plusvalía (plusvalia.com) — Radar de Valorización Urbana.

Plusvalía es el portal inmobiliario más grande de Ecuador. Este scraper
recorre las páginas de resultados de venta de inmuebles por ciudad y extrae:
título, precio (USD), área, habitaciones, baños, ubicación y URL del aviso.

Política ÉTICA: respeta robots.txt, usa retrasos aleatorios de 2-5 s entre
peticiones, maneja HTTP 429 y NUNCA intenta evadir mecanismos anti-bot.

Uso:
    python scripts/scrapers/scrape_plusvalia.py --city quito --pages 3
    python scripts/scrapers/scrape_plusvalia.py --city guayaquil --pages 5 --output data/raw/scraped
    python scripts/scrapers/scrape_plusvalia.py --city cuenca --pages 2 --delay-min 3 --delay-max 6

Nota: la estructura HTML del sitio puede cambiar; si no se encuentran
tarjetas se registra una advertencia y se continúa sin fallar.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permitir ejecución directa: python scripts/scrapers/scrape_plusvalia.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from bs4 import BeautifulSoup

    from common import (
        DEFAULT_DELAY_MAX,
        DEFAULT_DELAY_MIN,
        SiteBlockedError,
        extract_features,
        fetch_page,
        find_next_page,
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

logger = setup_logging(name="scrape_plusvalia")

SOURCE_NAME = "plusvalia"

# URLs de búsqueda verificadas (agosto 2026):
#   https://www.plusvalia.com/venta/inmuebles/pichincha/quito
#   https://www.plusvalia.com/venta/inmuebles/guayas/guayaquil
#   https://www.plusvalia.com/venta/inmuebles/azuay/cuenca
CITIES: Dict[str, str] = {
    "quito": "https://www.plusvalia.com/venta/inmuebles/pichincha/quito",
    "guayaquil": "https://www.plusvalia.com/venta/inmuebles/guayas/guayaquil",
    "cuenca": "https://www.plusvalia.com/venta/inmuebles/azuay/cuenca",
}

# Selectores de tarjetas de aviso (se prueba en orden; se usa el que más
# resultados devuelva). Actualizar si Plusvalía cambia su HTML.
CARD_SELECTORS: List[str] = [
    "article.post",
    "div.post",
    "[class*='post-card']",
    "[class*='listing-card']",
    "[class*='property-card']",
    "[data-testid*='listing']",
    "[class*='result']",
]

_TITLE_SELECTORS = (
    "h2.post-title a",
    "h3.post-title a",
    "h2 a",
    "h3 a",
    "a.post-title",
    "a[class*='title']",
    "a[href*='/venta/']",
)

_PRICE_RE = re.compile(r"(?i)(usd\s*[\d.,]+|\$\s*[\d.,]+|[\d.,]+\s*(?:usd|d[oó]lares?))")


def parse_listing_card(card: Any, base_url: str) -> Optional[Dict[str, Any]]:
    """Extrae un aviso de una tarjeta de Plusvalía. None si no es parseable."""
    # --- Título y URL -----------------------------------------------------
    link = None
    for selector in _TITLE_SELECTORS:
        element = card.select_one(selector)
        if element is not None and element.get("href"):
            link = element
            break
    if link is None:
        # Último recurso: cualquier ancla absoluta del mismo dominio
        for anchor in card.find_all("a", href=True):
            href = str(anchor["href"])
            if href.startswith("http") and "plusvalia.com" in href:
                link = anchor
                break
    if link is None:
        return None

    url = urljoin(base_url, str(link["href"]))
    title = text_of(link) or text_of(card.select_one("h2, h3"))

    # --- Precio -----------------------------------------------------------
    price_element = card.select_one(
        "[class*='price'], [class*='precio'], [class*='valor']"
    )
    price_text = text_of(price_element) if price_element is not None else None
    if not price_text:
        match = _PRICE_RE.search(card.get_text(" ", strip=True))
        price_text = match.group(0) if match else None

    # --- Ubicación --------------------------------------------------------
    location_element = card.select_one(
        "[class*='location'], [class*='address'], [class*='ubicacion'], "
        "span.post-location, p.post-location"
    )
    location = text_of(location_element) if location_element is not None else None

    # --- Características (área, hab., baños) ------------------------------
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
    """Parsea todas las tarjetas de aviso de una página de resultados."""
    soup = BeautifulSoup(html, "html.parser")

    cards: List[Any] = []
    best_selector: Optional[str] = None
    for selector in CARD_SELECTORS:
        found = soup.select(selector)
        if len(found) > len(cards):
            cards = found
            best_selector = selector
    if best_selector:
        logger.debug("Selector de tarjetas usado: %s (%d encontradas)", best_selector, len(cards))
    else:
        logger.warning(
            "Plusvalía: no se encontraron tarjetas con %s. "
            "La estructura HTML del sitio puede haber cambiado.",
            CARD_SELECTORS,
        )
        return []

    records: List[Dict[str, Any]] = []
    seen: set = set()
    for card in cards:
        record = parse_listing_card(card, base_url)
        if record is None:
            continue
        key = record["url"]
        if key in seen:
            continue
        seen.add(key)
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
    """Scrapea ``pages`` páginas de venta de inmuebles en ``city``.

    Devuelve los registros normalizados guardados. Lanza SiteBlockedError si
    el sitio bloquea el acceso (no se intenta evadir).
    """
    if city not in CITIES:
        raise ValueError(
            f"Ciudad no soportada: '{city}'. Válidas: {', '.join(sorted(CITIES))}"
        )
    base_url = CITIES[city]
    session = get_session()

    records: List[Dict[str, Any]] = []
    seen_urls: set = set()
    current_url: Optional[str] = base_url

    for page_no in range(1, pages + 1):
        if current_url is None:
            break
        logger.info("Plusvalía [%s] — página %d/%d: %s", city, page_no, pages, current_url)

        response = fetch_page(session, current_url, max_retries=max_retries)
        if response is None:
            logger.warning("No se pudo obtener %s; se detiene la paginación.", current_url)
            break

        blocked = is_blocked_response(response)
        if blocked:
            raise SiteBlockedError(
                f"Plusvalía bloqueó el acceso ({blocked}). "
                "No se intentará evadir la protección. Ejecute más tarde o "
                "use el canal oficial del portal."
            )

        page_records = parse_search_page(response.text, current_url)
        page_urls = {r["url"] for r in page_records}

        # Detener si una página posterior no aporta avisos nuevos
        if page_no > 1 and page_urls and page_urls.issubset(seen_urls):
            logger.info("La página %d no aporta avisos nuevos; se detiene.", page_no)
            break
        if not page_urls and page_no > 1:
            logger.warning("Página %d sin avisos; se detiene la paginación.", page_no)
            break

        records.extend(page_records)
        seen_urls |= page_urls
        logger.info("Plusvalía [%s] — página %d: %d avisos (acumulado: %d)", city, page_no, len(page_records), len(records))

        if page_no < pages:
            # Cortesía con el servidor entre peticiones
            polite_sleep(delay_min, delay_max)
            next_url = find_next_page(BeautifulSoup(response.text, "html.parser"), current_url)
            if next_url is None or next_url == current_url:
                # Respaldo determinista (Plusvalía usa ?pagina=N)
                current_url = f"{base_url}?pagina={page_no + 1}"
            else:
                current_url = next_url

    return save_results(records, output_dir, SOURCE_NAME)


def main() -> None:
    """CLI del scraper de Plusvalía."""
    parser = argparse.ArgumentParser(
        description="Scraper de Plusvalía (plusvalia.com) — Radar de Valorización Urbana. "
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
    logger.info("Plusvalía [%s]: %d avisos guardados en %s (%.1f s)",
                args.city, len(records), Path(args.output), elapsed)


if __name__ == "__main__":
    main()
