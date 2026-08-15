"""
ETL module for synthetic real estate transaction data generation.

Generates 2500+ transactions across 55 H3 hexagonal cells (resolution 8)
for Quito and Guayaquil, Ecuador. Includes realistic price ranges, temporal
trends, and geographic distribution.

Usage:
    python -m src.etl.etl_transactions
    python src/etl/etl_transactions.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
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
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUITO_CBD = (-0.18, -78.47)
GUAYAQUIL_CBD = (-2.17, -79.92)
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------
def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "config" / "config.yaml")

    if yaml is None:
        logger.warning("PyYAML not installed; using default configuration.")
        return _default_config()

    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _default_config() -> Dict[str, Any]:
    """Return a minimal default configuration when YAML is unavailable."""
    return {
        "h3": {"resolution": 8},
        "cities": {
            "quito": {
                "cbd_lat": QUITO_CBD[0], "cbd_lon": QUITO_CBD[1],
                "bbox": {"min_lat": -0.45, "max_lat": 0.10,
                          "min_lon": -78.70, "max_lon": -78.30},
                "num_cells": 30,
            },
            "guayaquil": {
                "cbd_lat": GUAYAQUIL_CBD[0], "cbd_lon": GUAYAQUIL_CBD[1],
                "bbox": {"min_lat": -2.30, "max_lat": -1.95,
                          "min_lon": -80.10, "max_lon": -79.75},
                "num_cells": 25,
            },
        },
        "paths": {
            "processed_dir": "data/processed",
            "transactions_parquet": "data/processed/transactions.parquet",
            "transactions_csv": "data/processed/transactions.csv",
            "transactions_geojson": "data/processed/transactions.geojson",
        },
        "transactions": {
            "total_count": 2500,
            "start_date": "2019-01-01",
            "end_date": "2024-12-31",
            "annual_price_growth": 0.065,
            "property_types": {
                "apartment": {"weight": 0.55, "price_min": 40000, "price_max": 200000,
                              "area_min": 40, "area_max": 150,
                              "bedrooms_range": [1, 4], "bathrooms_range": [1, 3]},
                "house": {"weight": 0.30, "price_min": 80000, "price_max": 400000,
                          "area_min": 80, "area_max": 350,
                          "bedrooms_range": [2, 6], "bathrooms_range": [1, 4]},
                "lot": {"weight": 0.15, "price_min": 30000, "price_max": 150000,
                        "area_min": 100, "area_max": 1000,
                        "bedrooms_range": [0, 0], "bathrooms_range": [0, 0]},
            },
        },
        "random_seed": 42,
    }


# ---------------------------------------------------------------------------
# H3 helpers
# ---------------------------------------------------------------------------
def _generate_cell_id(lat: float, lon: float, resolution: int) -> str:
    """Generate an H3 cell ID for a coordinate, or a synthetic fallback.

    Args:
        lat: Latitude.
        lon: Longitude.
        resolution: H3 resolution level.

    Returns:
        H3 cell ID string (or synthetic hex-like ID if h3 unavailable).
    """
    if H3_AVAILABLE:
        try:
            return h3.latlng_to_cell(lat, lon, resolution)
        except Exception:
            pass
    # Synthetic fallback — deterministic 15-char hex string
    import hashlib
    h = hashlib.sha256(f"{lat:.6f},{lon:.6f},{resolution}".encode()).hexdigest()[:15]
    return f"8{h}"


def _cell_center(cell_id: str) -> Tuple[float, float]:
    """Return the center coordinate of an H3 cell, or parse synthetic ID.

    Args:
        cell_id: H3 or synthetic cell identifier.

    Returns:
        (latitude, longitude) tuple.
    """
    if H3_AVAILABLE:
        try:
            lat, lon = h3.cell_to_latlng(cell_id)
            return (lat, lon)
        except Exception:
            pass
    # Cannot recover coords from synthetic hash — caller must track separately
    return (0.0, 0.0)


def _generate_grid_cells(
    city_config: Dict[str, Any],
    resolution: int,
    rng: np.random.Generator,
) -> List[Tuple[str, float, float]]:
    """Generate H3 grid cells within a city's bounding box.

    Uses a jittered grid sampling approach to spread cells across the bbox.

    Args:
        city_config: City configuration with bbox and num_cells.
        resolution: H3 resolution.
        rng: NumPy random generator.

    Returns:
        List of (cell_id, lat, lon) tuples.
    """
    bbox = city_config["bbox"]
    min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
    min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]

    num_cells = city_config.get("num_cells", 25)
    cells: List[Tuple[str, float, float]] = []
    seen_cells: set = set()

    # Generate points until we have enough unique cells
    attempts = 0
    max_attempts = num_cells * 20
    while len(cells) < num_cells and attempts < max_attempts:
        attempts += 1
        # Bias toward city center (normal distribution around CBD)
        center_lat = city_config["cbd_lat"]
        center_lon = city_config["cbd_lon"]
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        lat = center_lat + rng.normal(0, lat_range * 0.25)
        lon = center_lon + rng.normal(0, lon_range * 0.25)

        # Clamp to bbox
        lat = np.clip(lat, min_lat, max_lat)
        lon = np.clip(lon, min_lon, max_lon)

        cell_id = _generate_cell_id(float(lat), float(lon), resolution)
        if cell_id in seen_cells:
            continue
        seen_cells.add(cell_id)
        cells.append((cell_id, float(lat), float(lon)))

    logger.info("Generated %d unique cells for %s", len(cells), city_config.get("name", "unknown"))
    return cells


# ---------------------------------------------------------------------------
# Transaction generation
# ---------------------------------------------------------------------------
def _generate_transactions(
    cells: List[Tuple[str, float, float]],
    city_name: str,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate synthetic transactions for a set of cells.

    Args:
        cells: List of (cell_id, lat, lon) tuples.
        city_name: Name of the city.
        config: Transaction configuration section.
        rng: NumPy random generator.

    Returns:
        DataFrame with transaction records.
    """
    tx_config = config.get("transactions", config)
    total_count = tx_config.get("total_count", 2500)
    annual_growth = tx_config.get("annual_price_growth", 0.065)

    start_date = datetime.strptime(tx_config.get("start_date", "2019-01-01"), "%Y-%m-%d")
    end_date = datetime.strptime(tx_config.get("end_date", "2024-12-31"), "%Y-%m-%d")
    date_range_days = (end_date - start_date).days

    # Distribute transactions proportionally across cells
    # Some cells are more active (center cells have more transactions)
    cell_weights = rng.exponential(scale=1.0, size=len(cells))
    cell_weights = cell_weights / cell_weights.sum()
    tx_per_cell = rng.multinomial(total_count, cell_weights)

    property_types_cfg = tx_config["property_types"]
    ptypes = list(property_types_cfg.keys())
    pweights = np.array([property_types_cfg[pt]["weight"] for pt in ptypes])
    pweights = pweights / pweights.sum()

    records: List[Dict[str, Any]] = []

    for idx, (cell_id, cell_lat, cell_lon) in enumerate(cells):
        n_tx = tx_per_cell[idx]
        if n_tx == 0:
            continue

        # Distance to CBD (for price gradient)
        city_cbd_lat = QUITO_CBD[0] if city_name.lower() == "quito" else GUAYAQUIL_CBD[0]
        city_cbd_lon = QUITO_CBD[1] if city_name.lower() == "quito" else GUAYAQUIL_CBD[1]
        dist_to_cbd = _haversine_km(cell_lat, cell_lon, city_cbd_lat, city_cbd_lon)

        # Price modifier: closer to CBD = more expensive
        # Typical gradient: price decreases ~3% per km from center
        price_mod = max(0.3, 1.0 - dist_to_cbd * 0.015)

        for _ in range(n_tx):
            # Random date within range
            days_offset = rng.integers(0, date_range_days)
            tx_date = start_date + timedelta(days=int(days_offset))

            # Temporal price growth factor
            years_since_start = days_offset / 365.25
            temporal_factor = (1 + annual_growth) ** years_since_start

            # Select property type
            ptype = ptypes[int(rng.choice(len(ptypes), p=pweights))]
            pt_cfg = property_types_cfg[ptype]

            # Generate price (log-normal distribution for realistic skew)
            price_min = pt_cfg["price_min"]
            price_max = pt_cfg["price_max"]
            price_mean = (np.log(price_min) + np.log(price_max)) / 2
            price_std = (np.log(price_max) - np.log(price_min)) / 4
            base_price = np.exp(rng.normal(price_mean, price_std))
            base_price = np.clip(base_price, price_min, price_max)

            # Apply location and temporal modifiers
            final_price = base_price * price_mod * temporal_factor
            final_price = float(np.round(final_price, 2))

            # Area
            area = float(rng.integers(pt_cfg["area_min"], pt_cfg["area_max"] + 1))

            # Bedrooms and bathrooms
            bed_range = pt_cfg["bedrooms_range"]
            bath_range = pt_cfg["bathrooms_range"]
            bedrooms = int(rng.integers(bed_range[0], bed_range[1] + 1)) if bed_range[1] > 0 else 0
            bathrooms = int(rng.integers(bath_range[0], bath_range[1] + 1)) if bath_range[1] > 0 else 0

            # Jitter the location slightly within the cell
            lat_jitter = float(rng.normal(0, 0.002))
            lon_jitter = float(rng.normal(0, 0.002))

            records.append({
                "transaction_id": f"TX-{len(records) + 1:06d}",
                "cell_id": cell_id,
                "city": city_name,
                "lat": round(cell_lat + lat_jitter, 6),
                "lon": round(cell_lon + lon_jitter, 6),
                "price_usd": final_price,
                "area_m2": area,
                "price_per_m2": round(final_price / area, 2),
                "property_type": ptype,
                "date": tx_date.strftime("%Y-%m-%d"),
                "year": tx_date.year,
                "month": tx_date.month,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "dist_to_cbd_km": round(dist_to_cbd, 3),
            })

    df = pd.DataFrame(records)
    return df


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance between two points in kilometers.

    Args:
        lat1, lon1: First point coordinates.
        lat2, lon2: Second point coordinates.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth radius in km
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate transaction data integrity.

    Checks for nulls, positive prices, valid coordinates, and reasonable ranges.

    Args:
        df: Raw transactions DataFrame.

    Returns:
        Cleaned DataFrame.

    Raises:
        ValueError: If critical validation fails.
    """
    initial_count = len(df)
    issues: List[str] = []

    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        for col, cnt in null_counts.items():
            if cnt > 0:
                issues.append(f"Column '{col}' has {cnt} nulls — dropping affected rows.")
        df = df.dropna(subset=["price_usd", "lat", "lon", "cell_id"])

    # Positive prices
    bad_prices = df["price_usd"] <= 0
    if bad_prices.any():
        issues.append(f"{bad_prices.sum()} rows with non-positive prices — removing.")
        df = df[~bad_prices]

    # Valid coordinates for Ecuador
    valid_lat = (df["lat"] >= -5) & (df["lat"] <= 2)
    valid_lon = (df["lon"] >= -82) & (df["lon"] <= -75)
    invalid_coords = ~(valid_lat & valid_lon)
    if invalid_coords.any():
        issues.append(f"{invalid_coords.sum()} rows with invalid coordinates — removing.")
        df = df[~invalid_coords]

    # Valid area
    bad_area = df["area_m2"] <= 0
    if bad_area.any():
        issues.append(f"{bad_area.sum()} rows with non-positive area — removing.")
        df = df[~bad_area]

    if issues:
        for issue in issues:
            logger.warning(issue)
        logger.warning("Validation removed %d of %d rows (%.1f%%).",
                       initial_count - len(df), initial_count,
                       (initial_count - len(df)) / max(initial_count, 1) * 100)

    logger.info("Validation complete: %d valid transactions.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _save_parquet_or_csv(df: pd.DataFrame, parquet_path: str, csv_path: str) -> None:
    """Save DataFrame to parquet (preferred) or CSV (fallback).

    Args:
        df: DataFrame to save.
        parquet_path: Preferred parquet file path.
        csv_path: Fallback CSV file path.
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    if PARQUET_AVAILABLE:
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved %d rows to %s", len(df), parquet_path)
    else:
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s (parquet unavailable)", len(df), csv_path)


def _save_geojson(df: pd.DataFrame, geojson_path: str) -> None:
    """Save transactions as a GeoJSON file.

    Args:
        df: Transactions DataFrame with lat/lon columns.
        geojson_path: Output file path.
    """
    Path(geojson_path).parent.mkdir(parents=True, exist_ok=True)

    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["lon"]), float(row["lat"])],
            },
            "properties": {
                k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                for k, v in row.to_dict().items()
                if k not in ("lat", "lon")
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    import json
    with open(geojson_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, ensure_ascii=False)
    logger.info("Saved %d features to %s", len(features), geojson_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute the transactions ETL pipeline.

    Generates synthetic real estate transactions for Quito and Guayaquil,
    validates data quality, and saves to parquet/CSV and GeoJSON.

    Args:
        config: Configuration dictionary. If None, loads from config.yaml.

    Returns:
        DataFrame containing all generated and validated transactions.
    """
    if config is None:
        config = _load_config()

    seed = config.get("random_seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    resolution = config["h3"]["resolution"]
    cities = config["cities"]
    paths = config["paths"]

    all_transactions: List[pd.DataFrame] = []

    for city_key, city_cfg in cities.items():
        city_name = city_cfg.get("name", city_key.capitalize())
        logger.info("Generating grid cells for %s...", city_name)
        cells = _generate_grid_cells(city_cfg, resolution, rng)

        logger.info("Generating transactions for %s...", city_name)
        city_tx = _generate_transactions(cells, city_name, config, rng)
        all_transactions.append(city_tx)
        logger.info("Generated %d transactions for %s", len(city_tx), city_name)

    df = pd.concat(all_transactions, ignore_index=True)
    logger.info("Total transactions before validation: %d", len(df))

    df = _validate_transactions(df)

    # Save outputs
    processed_dir = Path(paths.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = paths.get("transactions_parquet", "data/processed/transactions.parquet")
    csv_path = paths.get("transactions_csv", "data/processed/transactions.csv")
    geojson_path = paths.get("transactions_geojson", "data/processed/transactions.geojson")

    _save_parquet_or_csv(df, parquet_path, csv_path)
    _save_geojson(df, geojson_path)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("TRANSACTION ETL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total transactions: %d", len(df))
    logger.info("Cities: %s", df["city"].unique().tolist())
    logger.info("Unique cells: %d", df["cell_id"].nunique())
    logger.info("Property types: %s", df["property_type"].value_counts().to_dict())
    logger.info("Date range: %s to %s", df["date"].min(), df["date"].max())
    logger.info("Price range: $%.0f - $%.0f", df["price_usd"].min(), df["price_usd"].max())
    logger.info("Mean price: $%.0f", df["price_usd"].mean())
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
