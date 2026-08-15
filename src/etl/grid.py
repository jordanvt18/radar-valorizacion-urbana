"""
Canonical H3 grid for the Radar de Valorización Urbana.

All ETL modules must use the SAME hexagonal grid so that cell_ids are
consistent across transactions, mobility, satellite, services and
socioeconomic datasets. This module generates the canonical grid once,
persists it to ``data/processed/grid.csv``, and every ETL loads it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root on path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
GRID_RELATIVE_PATH = "data/processed/grid.csv"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers."""
    r = 6371.0
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def _generate_cell_id(lat: float, lon: float, resolution: int) -> str:
    """Generate an H3 cell ID for a coordinate, or a synthetic fallback."""
    if H3_AVAILABLE:
        try:
            return h3.latlng_to_cell(lat, lon, resolution)
        except Exception:
            pass
    import hashlib
    h = hashlib.sha256(f"{lat:.6f},{lon:.6f},{resolution}".encode()).hexdigest()[:15]
    return f"8{h}"


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration (or fall back to defaults)."""
    if config_path is None:
        config_path = str(_PROJECT_ROOT / "config" / "config.yaml")
    if yaml is not None and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    # Minimal default config
    return {
        "h3": {"resolution": 8},
        "cities": {
            "quito": {
                "name": "Quito", "cbd_lat": -0.18, "cbd_lon": -78.47,
                "bbox": {"min_lat": -0.45, "max_lat": 0.10,
                         "min_lon": -78.70, "max_lon": -78.30},
                "num_cells": 30,
            },
            "guayaquil": {
                "name": "Guayaquil", "cbd_lat": -2.17, "cbd_lon": -79.92,
                "bbox": {"min_lat": -2.30, "max_lat": -1.95,
                         "min_lon": -80.10, "max_lon": -79.75},
                "num_cells": 25,
            },
        },
        "random_seed": 42,
    }


def generate_grid(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Generate the canonical grid deterministically.

    Args:
        config: Project config. If None, loads from config.yaml.

    Returns:
        DataFrame with columns: cell_id, city, lat, lon, distance_to_cbd_km.
    """
    if config is None:
        config = _load_config()

    seed = config.get("random_seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)
    resolution = config["h3"]["resolution"]
    cities = config["cities"]

    records: List[Dict[str, Any]] = []
    for city_key, city_cfg in cities.items():
        city_name = city_cfg.get("name", city_key.capitalize())
        bbox = city_cfg["bbox"]
        min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
        min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]
        center_lat = city_cfg["cbd_lat"]
        center_lon = city_cfg["cbd_lon"]
        num_cells = city_cfg.get("num_cells", 25)

        cells: List[Tuple[str, float, float]] = []
        seen: set = set()
        attempts = 0
        max_attempts = num_cells * 20
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon

        while len(cells) < num_cells and attempts < max_attempts:
            attempts += 1
            lat = float(np.clip(center_lat + rng.normal(0, lat_range * 0.25), min_lat, max_lat))
            lon = float(np.clip(center_lon + rng.normal(0, lon_range * 0.25), min_lon, max_lon))
            cell_id = _generate_cell_id(lat, lon, resolution)
            if cell_id in seen:
                continue
            seen.add(cell_id)
            cells.append((cell_id, lat, lon))

        for cell_id, lat, lon in cells:
            dist = _haversine_km(lat, lon, center_lat, center_lon)
            records.append({
                "cell_id": cell_id,
                "city": city_name,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "distance_to_cbd_km": round(dist, 3),
            })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="cell_id").reset_index(drop=True)
    logger.info("Canonical grid generated: %d cells (%d cities)", len(df), len(cities))
    return df


def load_or_create_grid(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load the canonical grid from disk, generating it if missing.

    Args:
        config: Project config.

    Returns:
        DataFrame with canonical grid cells.
    """
    grid_path = _PROJECT_ROOT / GRID_RELATIVE_PATH
    if grid_path.exists():
        df = pd.read_csv(grid_path)
        logger.info("Canonical grid loaded from %s (%d cells)", grid_path, len(df))
        return df

    df = generate_grid(config)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(grid_path, index=False)
    logger.info("Canonical grid saved to %s (%d cells)", grid_path, len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = load_or_create_grid()
    print(df.head(10).to_string())
    print(f"\nTotal cells: {len(df)}")
    print(f"Cities: {df['city'].value_counts().to_dict()}")
