"""
Orquestador de scrapers inmobiliarios — Radar de Valorización Urbana.

Ejecuta todos los scrapers en secuencia (Plusvalía, Properati, OLX, RE/MAX),
agrega los resultados individuales en data/raw/scraped/all_listings.csv
(+ .json) e imprime estadísticas resumidas (totales por fuente y rangos de
precio).

Política ÉTICA: cada scraper respeta robots.txt, usa retrasos de 2-5 s entre
peticiones y no evade anti-bot. Si un sitio bloquea el acceso (p. ej. OLX),
se registra el fallo y se continúa con las demás fuentes.

Uso:
    python scripts/scrapers/run_all.py
    python scripts/scrapers/run_all.py --cities quito,guayaquil --pages 2
    python scripts/scrapers/run_all.py --skip olx --pages 3
    python scripts/scrapers/run_all.py --cities quito --pages 2 --output data/raw/scraped
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Permitir ejecución directa: python scripts/scrapers/run_all.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from common import (
        DEFAULT_DELAY_MAX,
        DEFAULT_DELAY_MIN,
        FIELDS,
        SiteBlockedError,
        polite_sleep,
        record_key,
        setup_logging,
    )
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] Falta una dependencia: {exc}")
    print("Instale con: pip install requests beautifulsoup4")
    sys.exit(1)

logger = setup_logging(name="run_all_scrapers")

AGGREGATE_NAME = "all_listings"

# Cada fuente se importa de forma independiente para que una dependencia
# rota no impida ejecutar el resto.
try:
    from scrape_plusvalia import scrape_city as scrape_plusvalia  # type: ignore
    PLUSVALIA_OK = True
except ImportError as exc:  # pragma: no cover
    logger.warning("No se pudo importar scrape_plusvalia: %s", exc)
    PLUSVALIA_OK = False
    scrape_plusvalia = None  # type: ignore[assignment]

try:
    from scrape_properati import scrape_city as scrape_properati  # type: ignore
    PROPERATI_OK = True
except ImportError as exc:  # pragma: no cover
    logger.warning("No se pudo importar scrape_properati: %s", exc)
    PROPERATI_OK = False
    scrape_properati = None  # type: ignore[assignment]

try:
    from scrape_olx import scrape_city as scrape_olx  # type: ignore
    OLX_OK = True
except ImportError as exc:  # pragma: no cover
    logger.warning("No se pudo importar scrape_olx: %s", exc)
    OLX_OK = False
    scrape_olx = None  # type: ignore[assignment]

try:
    from scrape_remax import scrape_city as scrape_remax  # type: ignore
    REMAX_OK = True
except ImportError as exc:  # pragma: no cover
    logger.warning("No se pudo importar scrape_remax: %s", exc)
    REMAX_OK = False
    scrape_remax = None  # type: ignore[assignment]


# (nombre, función scrape_city, ciudades soportadas)
def _available_sources() -> List[Tuple[str, Any, Dict[str, str]]]:
    sources: List[Tuple[str, Any, Dict[str, str]]] = []
    if PLUSVALIA_OK:
        from scrape_plusvalia import CITIES as P_CITIES  # type: ignore
        sources.append(("plusvalia", scrape_plusvalia, P_CITIES))
    if PROPERATI_OK:
        from scrape_properati import CITIES as PR_CITIES  # type: ignore
        sources.append(("properati", scrape_properati, PR_CITIES))
    if OLX_OK:
        from scrape_olx import CITIES as O_CITIES  # type: ignore
        sources.append(("olx", scrape_olx, O_CITIES))
    if REMAX_OK:
        from scrape_remax import CITIES as R_CITIES  # type: ignore
        sources.append(("remax", scrape_remax, R_CITIES))
    return sources


def _load_source_file(output_dir: Path, source_name: str) -> List[Dict[str, Any]]:
    """Carga los avisos ya guardados de una fuente (CSV preferido, JSON respaldo)."""
    csv_path = output_dir / f"{source_name}.csv"
    json_path = output_dir / f"{source_name}.json"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [row for row in csv.DictReader(handle) if row and any(row.values())]
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    return []


def aggregate(output_dir: Path) -> List[Dict[str, Any]]:
    """Fusiona los CSV/JSON de todas las fuentes en all_listings.csv/.json."""
    merged: Dict[str, Dict[str, Any]] = {}
    for source_name, _, _ in _available_sources():
        for row in _load_source_file(output_dir, source_name):
            key = record_key(row)
            if not key:
                continue
            merged[key] = row

    rows = list(merged.values())
    csv_path = output_dir / f"{AGGREGATE_NAME}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})

    json_path = output_dir / f"{AGGREGATE_NAME}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    logger.info("Agregados %d avisos únicos en %s", len(rows), csv_path)
    return rows


def _price_stats(rows: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float, int]]:
    """(min, mediana, max, n) de price_usd válidos o None si no hay datos."""
    prices = []
    for row in rows:
        try:
            price = float(row.get("price_usd"))
        except (TypeError, ValueError):
            continue
        if price > 0:
            prices.append(price)
    if not prices:
        return None
    prices.sort()
    return (
        prices[0],
        statistics.median(prices),
        prices[-1],
        len(prices),
    )


def print_summary(all_rows: List[Dict[str, Any]], failures: List[Tuple[str, str, str]]) -> None:
    """Imprime el resumen de recolección en consola."""
    line = "=" * 68
    print(f"\n{line}\nRESUMEN DE RECOLECCIÓN — RADAR DE VALORIZACIÓN URBANA\n{line}")

    print(f"Total de avisos únicos: {len(all_rows)}")

    print("\nPor fuente:")
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_source.setdefault(row.get("source") or "unknown", []).append(row)
    for source in sorted(by_source):
        rows = by_source[source]
        stats = _price_stats(rows)
        if stats:
            print(
                f"  {source:<12} {len(rows):>6} avisos | precio USD: "
                f"min {stats[0]:,.0f} | mediana {stats[1]:,.0f} | máx {stats[2]:,.0f}"
            )
        else:
            print(f"  {source:<12} {len(rows):>6} avisos | sin precios válidos")

    stats_all = _price_stats(all_rows)
    if stats_all:
        print(
            f"\nRango de precios global (USD): min {stats_all[0]:,.0f} | "
            f"mediana {stats_all[1]:,.0f} | máx {stats_all[2]:,.0f} "
            f"({stats_all[3]} avisos con precio)"
        )

    if failures:
        print("\nFuentes con error (se omitieron):")
        for source, city, reason in failures:
            print(f"  {source:<12} [{city}]: {reason}")

    print(f"\nArchivos: data/raw/scraped/*.csv, data/raw/scraped/{AGGREGATE_NAME}.csv")
    print(line)


def main() -> None:
    """CLI del orquestador de scrapers."""
    parser = argparse.ArgumentParser(
        description="Ejecuta todos los scrapers inmobiliarios y agrega resultados "
        "(Radar de Valorización Urbana). Recopilación ética: robots.txt, retrasos 2-5 s."
    )
    parser.add_argument("--cities", default="quito,guayaquil",
                        help="Ciudades separadas por coma (default: quito,guayaquil)")
    parser.add_argument("--pages", type=int, default=2,
                        help="Páginas por ciudad y fuente (default: 2)")
    parser.add_argument("--output", default="data/raw/scraped",
                        help="Directorio de salida (default: data/raw/scraped)")
    parser.add_argument("--skip", default="",
                        help="Fuentes a omitir separadas por coma (plusvalia,properati,olx,remax)")
    parser.add_argument("--delay-min", type=float, default=DEFAULT_DELAY_MIN,
                        help="Retraso mínimo entre peticiones (default: 2.0)")
    parser.add_argument("--delay-max", type=float, default=DEFAULT_DELAY_MAX,
                        help="Retraso máximo entre peticiones (default: 5.0)")
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

    cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [(name, func, city_map) for name, func, city_map in _available_sources()
               if name not in skip]
    if not sources:
        logger.error("No hay fuentes disponibles para ejecutar. Revise las dependencias.")
        sys.exit(1)

    start = time.time()
    failures: List[Tuple[str, str, str]] = []

    for source_name, scrape_func, city_map in sources:
        for city in cities:
            if city not in city_map:
                logger.warning("La fuente %s no soporta la ciudad '%s'; se omite.", source_name, city)
                continue
            try:
                logger.info("=== %s [%s] ===", source_name.upper(), city)
                records = scrape_func(
                    city=city,
                    pages=args.pages,
                    output_dir=output_dir,
                    delay_min=args.delay_min,
                    delay_max=args.delay_max,
                )
                logger.info("=== %s [%s]: %d avisos ===", source_name.upper(), city, len(records))
            except SiteBlockedError as exc:
                reason = f"bloqueado por anti-bot: {exc}"
                logger.error("=== %s [%s]: %s ===", source_name.upper(), city, reason)
                failures.append((source_name, city, reason))
            except Exception as exc:  # noqa: BLE001 - una fuente no debe tumbar el resto
                reason = f"error inesperado: {exc}"
                logger.exception("=== %s [%s]: %s ===", source_name.upper(), city, reason)
                failures.append((source_name, city, reason))
            # Cortesía adicional entre ciudad/fuente
            polite_sleep(args.delay_min, args.delay_max)

    all_rows = aggregate(output_dir)
    print_summary(all_rows, failures)

    elapsed = time.time() - start
    logger.info("Recolección completa en %.1f s — %d avisos únicos en %s",
                elapsed, len(all_rows), output_dir / f"{AGGREGATE_NAME}.csv")


if __name__ == "__main__":
    main()
