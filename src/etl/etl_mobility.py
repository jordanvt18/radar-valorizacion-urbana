"""
ETL module for synthetic urban mobility data generation.

Generates per-cell mobility metrics for Quito and Guayaquil, including
travel time to CBD, traffic index, peak-hour speeds, transit stops,
walkability, and connectivity indices.

Usage:
    python -m src.etl.etl_mobility
    python src/etl/etl_mobility.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running as script or module
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from etl_transactions
# ---------------------------------------------------------------------------
from src.etl.etl_transactions import (
    _haversine_km,
    _load_config,
    QUITO_CBD,
    GUAYAQUIL_CBD,
    DEFAULT_SEED,
)
from src.etl.grid import load_or_create_grid

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mobility generation
# ---------------------------------------------------------------------------
def _generate_mobility_for_cells(
    cells: List[Tuple[str, float, float]],
    city_name: str,
    city_cfg: Dict[str, Any],
    mobility_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate mobility metrics for a set of grid cells.

    Args:
        cells: List of (cell_id, lat, lon) tuples.
        city_name: Name of the city.
        city_cfg: City configuration with CBD coordinates.
        mobility_cfg: Mobility parameters from config.
        rng: NumPy random generator.

    Returns:
        DataFrame with mobility metrics per cell.
    """
    cbd_lat = city_cfg["cbd_lat"]
    cbd_lon = city_cfg["cbd_lon"]

    min_tt = mobility_cfg.get("min_travel_time_min", 5)
    max_tt = mobility_cfg.get("max_travel_time_min", 90)

    records: List[Dict[str, Any]] = []

    for cell_id, lat, lon in cells:
        dist_to_cbd = _haversine_km(lat, lon, cbd_lat, cbd_lon)

        # Average travel time to CBD: closer = faster
        # Base travel time proportional to distance, with noise
        # Assume average speed of 25 km/h in city traffic
        base_travel_time = (dist_to_cbd / 25.0) * 60  # minutes
        noise = rng.normal(0, 3.0)
        avg_travel_time = np.clip(base_travel_time + noise, min_tt, max_tt)

        # Traffic index: 0-100, higher = more congestion (near center)
        # Decay with distance from CBD
        traffic_base = 80 * np.exp(-dist_to_cbd * 0.15)
        traffic_index = float(np.clip(traffic_base + rng.normal(0, 8), 0, 100))

        # Peak hour speed: inverse of traffic index
        # Range: 10-60 km/h
        peak_speed = float(np.clip(60 - (traffic_index * 0.5) + rng.normal(0, 4), 10, 60))

        # Transit stops count: Poisson, more near center
        # Lambda decreases with distance from CBD
        transit_lambda = max(0.5, 15 * np.exp(-dist_to_cbd * 0.12))
        transit_stops = int(rng.poisson(transit_lambda))

        # Walkability score: 0-100, higher near center
        # Influenced by density, transit, and connectivity
        walk_base = 90 * np.exp(-dist_to_cbd * 0.10)
        walkability = float(np.clip(walk_base + rng.normal(0, 7), 0, 100))

        # Connectivity index: 0-100, measure of street network density
        # Correlated with walkability but with independent noise
        connectivity_base = 85 * np.exp(-dist_to_cbd * 0.08)
        connectivity = float(np.clip(connectivity_base + rng.normal(0, 6), 0, 100))

        records.append({
            "cell_id": cell_id,
            "city": city_name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "avg_travel_time_cbd_min": round(avg_travel_time, 1),
            "traffic_index": round(traffic_index, 1),
            "peak_hour_speed_kmh": round(peak_speed, 1),
            "transit_stops_count": transit_stops,
            "walkability_score": round(walkability, 1),
            "connectivity_index": round(connectivity, 1),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_mobility(df: pd.DataFrame) -> pd.DataFrame:
    """Validate mobility data integrity.

    Args:
        df: Raw mobility DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)
    issues: List[str] = []

    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        for col, cnt in null_counts.items():
            if cnt > 0:
                issues.append(f"Column '{col}' has {cnt} nulls — dropping affected rows.")
        df = df.dropna()

    # Validate ranges
    df.loc[df["avg_travel_time_cbd_min"] < 0, "avg_travel_time_cbd_min"] = 0
    df.loc[df["traffic_index"] < 0, "traffic_index"] = 0
    df.loc[df["traffic_index"] > 100, "traffic_index"] = 100
    df.loc[df["walkability_score"] < 0, "walkability_score"] = 0
    df.loc[df["walkability_score"] > 100, "walkability_score"] = 100
    df.loc[df["connectivity_index"] < 0, "connectivity_index"] = 0
    df.loc[df["connectivity_index"] > 100, "connectivity_index"] = 100
    df.loc[df["transit_stops_count"] < 0, "transit_stops_count"] = 0
    df.loc[df["peak_hour_speed_kmh"] < 0, "peak_hour_speed_kmh"] = 0

    if issues:
        for issue in issues:
            logger.warning(issue)
        logger.warning(
            "Validation removed %d of %d rows.",
            initial_count - len(df),
            initial_count,
        )

    logger.info("Validation complete: %d valid mobility records.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute the mobility ETL pipeline.

    Generates synthetic urban mobility metrics for all H3 cells in
    Quito and Guayaquil, validates data quality, and saves to CSV.

    Args:
        config: Configuration dictionary. If None, loads from config.yaml.

    Returns:
        DataFrame containing all generated mobility metrics.
    """
    if config is None:
        config = _load_config()

    seed = config.get("random_seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    resolution = config["h3"]["resolution"]
    cities = config["cities"]
    mobility_cfg = config.get("mobility", {})
    paths = config["paths"]

    all_mobility: List[pd.DataFrame] = []

    # Load canonical grid (same cells across ALL ETL modules)
    grid = load_or_create_grid(config)

    for city_key, city_cfg in cities.items():
        city_name = city_cfg.get("name", city_key.capitalize())
        logger.info("Loading grid cells for %s...", city_name)
        city_grid = grid[grid["city"] == city_name]
        cells = list(zip(city_grid["cell_id"], city_grid["lat"], city_grid["lon"]))

        logger.info("Generating mobility metrics for %s...", city_name)
        city_mobility = _generate_mobility_for_cells(
            cells, city_name, city_cfg, mobility_cfg, rng
        )
        all_mobility.append(city_mobility)
        logger.info("Generated %d mobility records for %s", len(city_mobility), city_name)

    df = pd.concat(all_mobility, ignore_index=True)
    logger.info("Total mobility records before validation: %d", len(df))

    df = _validate_mobility(df)

    # Save output
    output_path = paths.get("mobility_csv", "data/processed/mobility.csv")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("MOBILITY ETL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cells: %d", len(df))
    logger.info("Cities: %s", df["city"].unique().tolist())
    logger.info(
        "Avg travel time to CBD: %.1f min (range: %.1f-%.1f)",
        df["avg_travel_time_cbd_min"].mean(),
        df["avg_travel_time_cbd_min"].min(),
        df["avg_travel_time_cbd_min"].max(),
    )
    logger.info(
        "Traffic index: %.1f avg (range: %.1f-%.1f)",
        df["traffic_index"].mean(),
        df["traffic_index"].min(),
        df["traffic_index"].max(),
    )
    logger.info(
        "Walkability: %.1f avg (range: %.1f-%.1f)",
        df["walkability_score"].mean(),
        df["walkability_score"].min(),
        df["walkability_score"].max(),
    )
    logger.info(
        "Transit stops: %.1f avg (range: %d-%d)",
        df["transit_stops_count"].mean(),
        df["transit_stops_count"].min(),
        df["transit_stops_count"].max(),
    )
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
