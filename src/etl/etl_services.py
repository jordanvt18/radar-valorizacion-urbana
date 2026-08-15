"""
ETL module for synthetic urban services data generation.

Generates per-cell counts of urban amenities (hospitals, schools,
supermarkets, parks, banks, restaurants) and distances to nearest
facilities for Quito and Guayaquil.

Usage:
    python -m src.etl.etl_services
    python src/etl/etl_services.py
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
    _generate_grid_cells,
    _haversine_km,
    _load_config,
    DEFAULT_SEED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Services generation
# ---------------------------------------------------------------------------
def _generate_services_for_cells(
    cells: List[Tuple[str, float, float]],
    city_name: str,
    city_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate urban services data for a set of grid cells.

    Service counts follow Poisson/Negative Binomial distributions with
    higher density near the CBD. Distances to nearest facilities are
    computed based on service availability and cell location.

    Args:
        cells: List of (cell_id, lat, lon) tuples.
        city_name: Name of the city.
        city_cfg: City configuration with CBD coordinates.
        rng: NumPy random generator.

    Returns:
        DataFrame with urban services data per cell.
    """
    cbd_lat = city_cfg["cbd_lat"]
    cbd_lon = city_cfg["cbd_lon"]

    records: List[Dict[str, Any]] = []

    for cell_id, lat, lon in cells:
        dist_to_cbd = _haversine_km(lat, lon, cbd_lat, cbd_lon)

        # Service lambda (expected count) decays with distance from CBD
        # Each service type has a different decay rate and base density

        # Hospitals: sparse, concentrated near center
        hosp_lambda = max(0.05, 3.0 * np.exp(-dist_to_cbd * 0.15))
        hospitals_count = int(rng.poisson(hosp_lambda))

        # Schools: moderate density, spread more evenly
        schools_lambda = max(0.2, 8.0 * np.exp(-dist_to_cbd * 0.08))
        schools_count = int(rng.poisson(schools_lambda))

        # Supermarkets: moderate, commercial corridors
        market_lambda = max(0.1, 5.0 * np.exp(-dist_to_cbd * 0.10))
        supermarkets_count = int(rng.poisson(market_lambda))

        # Parks: more in suburbs, some in center
        parks_lambda = max(0.1, 2.0 + 3.0 * (1 - np.exp(-dist_to_cbd * 0.05)))
        parks_count = int(rng.poisson(parks_lambda))

        # Banks: concentrated in commercial/financial districts
        banks_lambda = max(0.05, 4.0 * np.exp(-dist_to_cbd * 0.18))
        banks_count = int(rng.poisson(banks_lambda))

        # Restaurants: dense near center, commercial areas
        rest_lambda = max(0.5, 20.0 * np.exp(-dist_to_cbd * 0.10))
        restaurants_count = int(rng.poisson(rest_lambda))

        # Distances to nearest facilities
        # If services exist in cell, distance is small (within cell radius ~0.5km)
        # Otherwise, distance increases based on how far from CBD
        cell_radius = 0.5  # approximate H3 res 8 cell radius in km

        def _nearest_distance(count: int, base_dist: float) -> float:
            """Estimate distance to nearest facility.

            Args:
                count: Number of facilities in the cell.
                base_dist: Distance to CBD (for fallback estimation).

            Returns:
                Estimated distance in km to nearest facility.
            """
            if count > 0:
                # Facility exists in cell — distance is within cell radius
                return float(rng.uniform(0.05, cell_radius))
            else:
                # No facility in cell — distance based on CBD proximity
                # (services cluster near CBD, so nearer cells have closer facilities)
                return float(np.clip(
                    base_dist * 0.3 + rng.uniform(0.3, 2.0),
                    0.5, 15.0,
                ))

        nearest_hospital = _nearest_distance(hospitals_count, dist_to_cbd)
        nearest_school = _nearest_distance(schools_count, dist_to_cbd)
        nearest_park = _nearest_distance(parks_count, dist_to_cbd)

        records.append({
            "cell_id": cell_id,
            "city": city_name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "hospitals_count": hospitals_count,
            "schools_count": schools_count,
            "supermarkets_count": supermarkets_count,
            "parks_count": parks_count,
            "banks_count": banks_count,
            "restaurants_count": restaurants_count,
            "distance_to_cbd_km": round(dist_to_cbd, 3),
            "nearest_hospital_km": round(nearest_hospital, 3),
            "nearest_school_km": round(nearest_school, 3),
            "nearest_park_km": round(nearest_park, 3),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_services(df: pd.DataFrame) -> pd.DataFrame:
    """Validate urban services data integrity.

    Args:
        df: Raw services DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)
    issues: List[str] = []

    null_counts = df.isnull().sum()
    if null_counts.any():
        for col, cnt in null_counts.items():
            if cnt > 0:
                issues.append(f"Column '{col}' has {cnt} nulls — dropping affected rows.")
        df = df.dropna()

    # Ensure non-negative counts
    count_cols = [
        "hospitals_count", "schools_count", "supermarkets_count",
        "parks_count", "banks_count", "restaurants_count",
    ]
    for col in count_cols:
        df.loc[df[col] < 0, col] = 0

    # Ensure non-negative distances
    dist_cols = ["distance_to_cbd_km", "nearest_hospital_km", "nearest_school_km", "nearest_park_km"]
    for col in dist_cols:
        df.loc[df[col] < 0, col] = 0.0

    if issues:
        for issue in issues:
            logger.warning(issue)
        logger.warning(
            "Validation removed %d of %d rows.",
            initial_count - len(df),
            initial_count,
        )

    logger.info("Validation complete: %d valid service records.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute the services ETL pipeline.

    Generates synthetic urban services data (amenity counts and nearest
    facility distances) for all H3 cells in Quito and Guayaquil.

    Args:
        config: Configuration dictionary. If None, loads from config.yaml.

    Returns:
        DataFrame containing all generated services data.
    """
    if config is None:
        config = _load_config()

    seed = config.get("random_seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    resolution = config["h3"]["resolution"]
    cities = config["cities"]
    paths = config["paths"]

    all_services: List[pd.DataFrame] = []

    for city_key, city_cfg in cities.items():
        city_name = city_cfg.get("name", city_key.capitalize())
        logger.info("Generating grid cells for %s...", city_name)
        cells = _generate_grid_cells(city_cfg, resolution, rng)

        logger.info("Generating urban services data for %s...", city_name)
        city_services = _generate_services_for_cells(cells, city_name, city_cfg, rng)
        all_services.append(city_services)
        logger.info("Generated %d service records for %s", len(city_services), city_name)

    df = pd.concat(all_services, ignore_index=True)
    logger.info("Total service records before validation: %d", len(df))

    df = _validate_services(df)

    # Save output
    output_path = paths.get("services_csv", "data/processed/services.csv")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("SERVICES ETL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cells: %d", len(df))
    logger.info("Cities: %s", df["city"].unique().tolist())
    logger.info(
        "Hospitals: %d total (avg %.1f per cell)",
        df["hospitals_count"].sum(),
        df["hospitals_count"].mean(),
    )
    logger.info(
        "Schools: %d total (avg %.1f per cell)",
        df["schools_count"].sum(),
        df["schools_count"].mean(),
    )
    logger.info(
        "Restaurants: %d total (avg %.1f per cell)",
        df["restaurants_count"].sum(),
        df["restaurants_count"].mean(),
    )
    logger.info(
        "Avg distance to CBD: %.2f km (range: %.2f-%.2f)",
        df["distance_to_cbd_km"].mean(),
        df["distance_to_cbd_km"].min(),
        df["distance_to_cbd_km"].max(),
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
