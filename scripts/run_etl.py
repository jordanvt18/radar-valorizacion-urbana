"""
ETL Pipeline Orchestrator — Radar de Valorización Urbana.

Runs all ETL modules in sequence and then feature engineering.

Usage:
    python scripts/run_etl.py
    python -m scripts.run_etl
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_etl")


def main() -> None:
    """Run the full ETL + feature engineering pipeline."""
    start = time.time()
    logger.info("=" * 60)
    logger.info("RADAR DE VALORIZACIÓN URBANA — ETL PIPELINE")
    logger.info("=" * 60)

    # Step 1: Transactions
    logger.info("[1/6] Generating real estate transactions...")
    try:
        from src.etl.etl_transactions import run as run_transactions
        run_transactions()
    except Exception as e:
        logger.error("Failed: etl_transactions — %s", e)
        raise

    # Step 2: Mobility
    logger.info("[2/6] Generating mobility data...")
    try:
        from src.etl.etl_mobility import run as run_mobility
        run_mobility()
    except Exception as e:
        logger.error("Failed: etl_mobility — %s", e)
        raise

    # Step 3: Satellite
    logger.info("[3/6] Generating satellite features...")
    try:
        from src.etl.etl_satellite import run as run_satellite
        run_satellite()
    except Exception as e:
        logger.error("Failed: etl_satellite — %s", e)
        raise

    # Step 4: Services
    logger.info("[4/6] Generating urban services data...")
    try:
        from src.etl.etl_services import run as run_services
        run_services()
    except Exception as e:
        logger.error("Failed: etl_services — %s", e)
        raise

    # Step 5: Socioeconomic
    logger.info("[5/6] Generating socioeconomic indicators...")
    try:
        from src.etl.etl_socioeconomic import run as run_socioeconomic
        run_socioeconomic()
    except Exception as e:
        logger.error("Failed: etl_socioeconomic — %s", e)
        raise

    # Step 6: Feature Engineering
    logger.info("[6/6] Building features...")
    try:
        from src.features.build_features import build_features
        build_features()
    except Exception as e:
        logger.error("Failed: build_features — %s", e)
        raise

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("ETL PIPELINE COMPLETE — %.1f seconds", elapsed)
    logger.info("Output: data/processed/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
