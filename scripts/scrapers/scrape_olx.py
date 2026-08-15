"""
Scraper de OLX Ecuador (olx.com.ec) — Radar de Valorización Urbana.

Sección inmobiliaria: extrae título, precio (USD), ubicación y URL del aviso.

⚠️ AVISO IMPORTANTE: OLX implementa mecanismos anti-bot agresivos (captchas,
bloqueos por IP, etc.). Este scraper DETECTA el bloqueo y aborta con un
mensaje claro. NO intenta evadir la protección (ni captchas, ni rotación de
IP, ni headless browsers). Si OLX bloquea el acceso, la ejecución termina
de forma controlada y se recomienda usar los canales oficiales del sitio.

Política ÉTICA: respeta robots.txt, retrasos aleatorios de 2-5 s, manejo de
HTTP 429 y nunca evasión de anti-bot.

Uso:
    python scripts/scrapers/scrape_olx.py --city quito --pages 2
    python scripts/scrapers/scrape_olx.py --city guayaquil --pages 3 --output data/raw/scraped

Nota: la estructura HTML del sitio puede cambiar; si no se encuentran
tarjetas se registra una advertencia y se continúa sin fallar.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permitir ejecución directa: python scripts/scrapers/scrape_olx.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from bs4 import BeautifulSoup

    from common import (
        DEFAULT_DELAY_MAX,
        DEFAULT_DELAY_MIN,
        SiteBlockedError,
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

logger = setup_logging(name="scrape_olx")

SOURCE_NAME = "olx"

# Búsquedas de OLX (patrón verificable: https://www.olx.com.ec/items/q-{query}/)
CITY_QUERIES: Dict[str, str] = {
    "quito": "casas-en-venta-en-quito",
    "guayaquil": "casas-en-venta-en-guayaquil",
    "cuenca": "casas-en-venta-en-cuenca",
}

CITIES: Dict[str, str] = {
    city: f"https://www.olx.com.ec/items/q-{query}/"
    for city, query in CITY_QUERIES.items()
}

# OLX marca sus tarjetas con atributos data-aut-id (patrón histórico del sitio).
CARD_SELECTORS: List[str] = [
    "li[data-aut-id='itemBox']",
    "[data-aut-id*='itemBox']",
    "[data-aut-id*='item']",
    "div[class*='listing']",
]

_TITLE_SELECTORS = (
    "a[data-aut-id='itemTitle']",
    "span[data-aut-id='itemTitle']",
    "a[data-aut-id*='title']",
    "h2 a",
    "h3 a",
    "a[href*='/item/']",
)


def parse_listing_card(card: Any, base_url: str) -> Optional[Dict[str, Any]]:
    """Extrae un aviso de una tarjeta de OLX. None si no es parseable."""
    link = None
    for selector in _TITLE_SELECTORS:
        element = card.select_one(selector)
        if element is not None and element.get("href"):
            link = element
            break
    if link is None:
        for anchor in card.find_all("a", href=True):
            href = str(anchor["href"])
            if href and not href.startswith("#") and "olx.com.ec" in urljoin(base_url, href):
                link = anchor
                break
    if link is None:
        return None

    url = urljoin(base_url, str(link["href"]))
    title = text_of(link) or text_of(card.select_one("[data-aut-id='itemTitle']")) or None

    price_element = card.select_one("[data-aut-id='itemPrice'], [data-aut-id*='price']")
    price_text = text_of(price_element) if price_element is not None else None

    location_element = card.select_one("[data-aut-id='itemLocation'], [data-aut-id*='location']")
    location = text_of(location_element) if location_element is not None else None

    return {
        "title": title,
        "url": url,
        "price_usd": price_text,
        "location": location,
        "area_m2": None,
        "bedrooms": None,
        "bathrooms": None,
        "source": SOURCE_NAME,
        "property_type": None,  # lo infiere normalize_record
    }


def parse_search_page(html: str, base_url: str) -> List[Dict[str, Any]]:
    """Parsea todas las tarjetas de aviso de una página de resultados de OLX."""
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
            "OLX: no se encontraron tarjetas con %s. "
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
    pages: int = 2,
    output_dir: Any = "data/raw/scraped",
    delay_min: float = DEFAULT_DELAY_MIN,
    delay_max: float = DEFAULT_DELAY_MAX,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Scrapea ``pages`` páginas de la sección inmobiliaria de OLX en ``city``.

    Devuelve los registros normalizados guardados (puede ser [] si OLX
    bloqueó el acceso; en ese caso el error ya fue registrado).
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
        logger.info("OLX [%s] — página %d/%d: %s", city, page_no, pages, current_url)

        response = fetch_page(session, current_url, max_retries=max_retries)
        if response is None:
            logger.warning("No se pudo obtener %s; se detiene la paginación.", current_url)
            break

        blocked = is_blocked_response(response)
        if blocked:
            # Fallo GRACIOSO y explícito: no se intenta evadir el anti-bot.
            message = (
                f"OLX bloqueó el acceso ({blocked}). OLX utiliza mecanismos "
                "anti-bot que este proyecto no evade por política ética. "
                "Opciones: (1) ejecutar en otro horario/red, (2) usar la "
                "información publicada en la página sin automatización, o "
                "(3) contactar a OLX para un canal oficial de datos."
            )
            logger.error("%s", message)
            print(f"\n[OLX] {message}\n")
            return []

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
        logger.info("OLX [%s] — página %d: %d avisos (acumulado: %d)", city, page_no, len(page_records), len(records))

        if page_no < pages:
            polite_sleep(delay_min, delay_max)
            # OLX pagina con ?page=N; sin enlace "siguiente" confiable en HTML.
            current_url = f"{base_url}?page={page_no + 1}"

    return save_results(records, output_dir, SOURCE_NAME)


def main() -> None:
    """CLI del scraper de OLX Ecuador."""
    parser = argparse.ArgumentParser(
        description="Scraper de OLX Ecuador (olx.com.ec) — Radar de Valorización Urbana. "
        "Recopilación ética: respeta robots.txt, retrasos de 2-5 s y NO evita anti-bot."
    )
    parser.add_argument("--city", default="quito", choices=sorted(CITIES),
                        help="Ciudad a scrapear (default: quito)")
    parser.add_argument("--pages", type=int, default=2,
                        help="Número de páginas de resultados (default: 2)")
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
    except ValueError as exc:
        parser.error(str(exc))

    elapsed = time.time() - start
    logger.info("OLX [%s]: %d avisos guardados en %s (%.1f s)",
                args.city, len(records), Path(args.output), elapsed)


if __name__ == "__main__":
    main()
