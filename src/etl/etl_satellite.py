"""
ETL module for synthetic satellite-derived urban features.

Generates NDVI, built-up index, green space ratio, night lights intensity,
and land use mix for each H3 cell. Also produces 128×128 numpy patches
representing spatial patterns of NDVI, built-up areas, and land use.

Usage:
    python -m src.etl.etl_satellite
    python src/etl/etl_satellite.py
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
    DEFAULT_SEED,
)
from src.etl.grid import load_or_create_grid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Satellite feature generation
# ---------------------------------------------------------------------------
def _generate_satellite_features(
    cells: List[Tuple[str, float, float]],
    city_name: str,
    city_cfg: Dict[str, Any],
    satellite_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, List[Tuple[str, np.ndarray]]]:
    """Generate satellite-derived features and spatial patches for cells.

    Args:
        cells: List of (cell_id, lat, lon) tuples.
        city_name: Name of the city.
        city_cfg: City configuration with CBD coordinates.
        satellite_cfg: Satellite parameters from config.
        rng: NumPy random generator.

    Returns:
        Tuple of (features DataFrame, list of (cell_id, patch_array) tuples).
    """
    cbd_lat = city_cfg["cbd_lat"]
    cbd_lon = city_cfg["cbd_lon"]
    patch_size = satellite_cfg.get("patch_size", PATCH_SIZE)

    records: List[Dict[str, Any]] = []
    patches: List[Tuple[str, np.ndarray]] = []

    for cell_id, lat, lon in cells:
        dist_to_cbd = _haversine_km(lat, lon, cbd_lat, cbd_lon)

        # --- NDVI: higher in suburbs/parks, lower in dense urban ---
        # Base NDVI increases with distance from CBD (less concrete)
        ndvi_mean = float(np.clip(
            0.15 + 0.35 * (1 - np.exp(-dist_to_cbd * 0.10)) + rng.normal(0, 0.05),
            -0.1, 0.9,
        ))
        ndvi_std = float(np.clip(rng.uniform(0.05, 0.20), 0.02, 0.30))

        # --- Built-up index: higher near CBD ---
        built_up_index = float(np.clip(
            0.90 * np.exp(-dist_to_cbd * 0.12) + rng.normal(0, 0.05),
            0.0, 1.0,
        ))

        # --- Green space ratio: inverse of built-up ---
        green_space_ratio = float(np.clip(
            1.0 - built_up_index + rng.normal(0, 0.08),
            0.0, 1.0,
        ))

        # --- Night lights intensity: proxy for economic activity ---
        # Higher near CBD, decays with distance
        night_lights = float(np.clip(
            85 * np.exp(-dist_to_cbd * 0.10) + rng.normal(0, 8),
            0, 100,
        ))

        # --- Land use mix: Shannon entropy-based measure ---
        # Higher near center (diverse commercial/residential/industrial)
        # Drops in pure residential suburbs
        land_use_mix = float(np.clip(
            0.80 * np.exp(-dist_to_cbd * 0.06) + rng.uniform(-0.05, 0.10),
            0.0, 1.0,
        ))

        records.append({
            "cell_id": cell_id,
            "city": city_name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "ndvi_mean": round(ndvi_mean, 4),
            "ndvi_std": round(ndvi_std, 4),
            "built_up_index": round(built_up_index, 4),
            "green_space_ratio": round(green_space_ratio, 4),
            "night_lights_intensity": round(night_lights, 2),
            "land_use_mix": round(land_use_mix, 4),
        })

        # --- Generate 128×128 spatial patches ---
        # Channel 0: NDVI map — spatial pattern of vegetation
        # Use distance gradient + noise to create realistic spatial pattern
        yy, xx = np.mgrid[0:patch_size, 0:patch_size].astype(float)
        center_y, center_x = patch_size / 2, patch_size / 2
        # Distance from patch center (proxy for distance from cell center)
        patch_dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        patch_dist_norm = patch_dist / (patch_size / 2)

        # NDVI patch: base value with spatial variation
        ndvi_patch = np.clip(
            ndvi_mean
            + 0.15 * (patch_dist_norm * rng.uniform(0.5, 1.5))
            + rng.normal(0, 0.08, size=(patch_size, patch_size)),
            -0.2, 1.0,
        ).astype(np.float32)

        # Built-up patch: inverse of NDVI pattern
        built_up_patch = np.clip(
            built_up_index
            - 0.20 * (patch_dist_norm * rng.uniform(0.3, 0.8))
            + rng.normal(0, 0.06, size=(patch_size, patch_size)),
            0.0, 1.0,
        ).astype(np.float32)

        # Land use patch: categorical-like (0=residential, 1=commercial,
        # 2=industrial, 3=green) encoded as normalized values
        land_use_patch = np.zeros((patch_size, patch_size), dtype=np.float32)
        # Create clusters using random seeds
        num_clusters = max(2, int(rng.integers(3, 8)))
        cluster_centers = rng.uniform(
            low=0.2,
            high=0.8,
            size=(num_clusters, 2),
        ) * patch_size
        cluster_values = rng.uniform(0.0, 1.0, size=num_clusters)

        for cy, cx in cluster_centers:
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            influence = np.exp(-d / (patch_size * 0.15))
            land_use_patch += influence * rng.uniform(0, 1)

        land_use_patch = np.clip(
            land_use_patch / max(land_use_patch.max(), 1e-6)
            + rng.normal(0, 0.05, size=(patch_size, patch_size)),
            0.0, 1.0,
        ).astype(np.float32)

        # Stack channels: (128, 128, 3)
        patch = np.stack([ndvi_patch, built_up_patch, land_use_patch], axis=-1)
        patches.append((cell_id, patch))

    return pd.DataFrame(records), patches


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_satellite(df: pd.DataFrame) -> pd.DataFrame:
    """Validate satellite feature data integrity.

    Args:
        df: Raw satellite features DataFrame.

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

    # Clip values to valid ranges
    df["ndvi_mean"] = df["ndvi_mean"].clip(-1.0, 1.0)
    df["ndvi_std"] = df["ndvi_std"].clip(0.0, 1.0)
    df["built_up_index"] = df["built_up_index"].clip(0.0, 1.0)
    df["green_space_ratio"] = df["green_space_ratio"].clip(0.0, 1.0)
    df["night_lights_intensity"] = df["night_lights_intensity"].clip(0.0, 100.0)
    df["land_use_mix"] = df["land_use_mix"].clip(0.0, 1.0)

    if issues:
        for issue in issues:
            logger.warning(issue)
        logger.warning(
            "Validation removed %d of %d rows.",
            initial_count - len(df),
            initial_count,
        )

    logger.info("Validation complete: %d valid satellite records.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute the satellite ETL pipeline.

    Generates synthetic satellite-derived features (NDVI, built-up index,
    night lights, etc.) and 128×128 spatial patches for each H3 cell.
    Saves features to CSV and patches as .npy files.

    Args:
        config: Configuration dictionary. If None, loads from config.yaml.

    Returns:
        DataFrame containing all generated satellite features.
    """
    if config is None:
        config = _load_config()

    seed = config.get("random_seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    resolution = config["h3"]["resolution"]
    cities = config["cities"]
    satellite_cfg = config.get("satellite", {})
    paths = config["paths"]

    all_features: List[pd.DataFrame] = []
    all_patches: List[Tuple[str, np.ndarray]] = []

    # Load canonical grid (same cells across ALL ETL modules)
    grid = load_or_create_grid(config)

    for city_key, city_cfg in cities.items():
        city_name = city_cfg.get("name", city_key.capitalize())
        logger.info("Loading grid cells for %s...", city_name)
        city_grid = grid[grid["city"] == city_name]
        cells = list(zip(city_grid["cell_id"], city_grid["lat"], city_grid["lon"]))

        logger.info("Generating satellite features for %s...", city_name)
        city_features, city_patches = _generate_satellite_features(
            cells, city_name, city_cfg, satellite_cfg, rng
        )
        all_features.append(city_features)
        all_patches.extend(city_patches)
        logger.info(
            "Generated %d satellite feature records and %d patches for %s",
            len(city_features),
            len(city_patches),
            city_name,
        )

    df = pd.concat(all_features, ignore_index=True)
    logger.info("Total satellite records before validation: %d", len(df))

    df = _validate_satellite(df)

    # Save features CSV
    output_path = paths.get("satellite_csv", "data/processed/satellite_features.csv")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    # Save satellite patches as .npy files
    patches_dir = Path(
        paths.get("satellite_patches_dir", "data/processed/satellite_patches")
    )
    patches_dir.mkdir(parents=True, exist_ok=True)

    for cell_id, patch in all_patches:
        patch_path = patches_dir / f"{cell_id}.npy"
        np.save(str(patch_path), patch)

    logger.info("Saved %d satellite patches to %s", len(all_patches), patches_dir)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("SATELLITE ETL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cells: %d", len(df))
    logger.info("Cities: %s", df["city"].unique().tolist())
    logger.info(
        "NDVI mean: %.4f avg (range: %.4f-%.4f)",
        df["ndvi_mean"].mean(),
        df["ndvi_mean"].min(),
        df["ndvi_mean"].max(),
    )
    logger.info(
        "Built-up index: %.4f avg (range: %.4f-%.4f)",
        df["built_up_index"].mean(),
        df["built_up_index"].min(),
        df["built_up_index"].max(),
    )
    logger.info(
        "Night lights: %.1f avg (range: %.1f-%.1f)",
        df["night_lights_intensity"].mean(),
        df["night_lights_intensity"].min(),
        df["night_lights_intensity"].max(),
    )
    logger.info(
        "Green space ratio: %.4f avg (range: %.4f-%.4f)",
        df["green_space_ratio"].mean(),
        df["green_space_ratio"].min(),
        df["green_space_ratio"].max(),
    )
    logger.info("Satellite patches: %d files (%dx%dx%d)", len(all_patches), PATCH_SIZE, PATCH_SIZE, 3)
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
