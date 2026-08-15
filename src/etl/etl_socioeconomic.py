"""
ETL module for synthetic socioeconomic indicators.

Generates population density, median income, education level, employment
rate, internet penetration, crime index, and Gini coefficient per H3 cell
for Quito and Guayaquil. Based on INEC-like structure for Ecuador.

Usage:
    python -m src.etl.etl_socioeconomic
    python src/etl/etl_socioeconomic.py
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
# Constants — INEC-like baseline values for Ecuador
# ---------------------------------------------------------------------------
# Reference: INEC 2022 Census / ECV (Encuesta de Condiciones de Vida)
QUITO_BASE_INCOME = 850.0   # USD median monthly household income
GUAYAQUIL_BASE_INCOME = 750.0
ECUADOR_GINI = 0.465         # World Bank 2022 estimate
ECUADOR_EMPLOYMENT = 0.92    # ~92% employment rate
ECUADOR_INTERNET = 0.72      # ~72% internet penetration


# ---------------------------------------------------------------------------
# Socioeconomic generation
# ---------------------------------------------------------------------------
def _generate_socioeconomic_for_cells(
    cells: List[Tuple[str, float, float]],
    city_name: str,
    city_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate socioeconomic indicators for a set of grid cells.

    Args:
        cells: List of (cell_id, lat, lon) tuples.
        city_name: Name of the city.
        city_cfg: City configuration with CBD coordinates.
        rng: NumPy random generator.

    Returns:
        DataFrame with socioeconomic indicators per cell.
    """
    cbd_lat = city_cfg["cbd_lat"]
    cbd_lon = city_cfg["cbd_lon"]

    # City-specific baselines
    base_income = QUITO_BASE_INCOME if city_name.lower() == "quito" else GUAYAQUIL_BASE_INCOME

    records: List[Dict[str, Any]] = []

    for cell_id, lat, lon in cells:
        dist_to_cbd = _haversine_km(lat, lon, cbd_lat, cbd_lon)

        # --- Population density: higher near CBD ---
        # Quito: ~4,700/km² avg, up to ~15,000 in dense areas
        # Guayaquil: ~3,800/km² avg, up to ~12,000 in dense areas
        pop_base = 12000 if city_name.lower() == "quito" else 10000
        pop_density = float(np.clip(
            pop_base * np.exp(-dist_to_cbd * 0.15) + rng.normal(0, 800),
            200, 20000,
        ))

        # --- Median income: higher near CBD and affluent zones ---
        # Income gradient: center cells earn ~1.5-2x suburb cells
        income_gradient = 1.0 + 0.8 * np.exp(-dist_to_cbd * 0.10)
        # Add random affluent pockets (gentrified suburbs)
        affluent_pocket = 1.0
        if rng.random() < 0.15:  # 15% chance of affluent pocket
            affluent_pocket = rng.uniform(1.3, 1.8)
        median_income = float(np.clip(
            base_income * income_gradient * affluent_pocket
            + rng.normal(0, 100),
            200, 3000,
        ))

        # --- Education level: years of schooling (0-20) ---
        # Higher near center (access to universities, schools)
        edu_base = 12.0  # Average years of schooling in Ecuador
        education_level = float(np.clip(
            edu_base + 3.0 * np.exp(-dist_to_cbd * 0.08) - 1.5 * (1 - np.exp(-dist_to_cbd * 0.05))
            + rng.normal(0, 1.0),
            3.0, 19.0,
        ))

        # --- Employment rate: 0-1 ---
        # Slightly lower near very center (informal economy) and periphery
        employment_rate = float(np.clip(
            ECUADOR_EMPLOYMENT + 0.03 * np.exp(-dist_to_cbd * 0.06)
            - 0.02 * (dist_to_cbd > 10)
            + rng.normal(0, 0.02),
            0.75, 0.98,
        ))

        # --- Internet penetration: 0-1 ---
        # Higher in center, lower in periphery
        internet_penetration = float(np.clip(
            ECUADOR_INTERNET + 0.20 * np.exp(-dist_to_cbd * 0.08)
            + rng.normal(0, 0.05),
            0.30, 0.98,
        ))

        # --- Crime index: 0-100 ---
        # Mixed pattern: some high crime in both dense center and periphery
        # U-shaped: high near center (density) and far periphery (marginality)
        crime_center = 60 * np.exp(-dist_to_cbd * 0.12)
        crime_periphery = 40 * (1 - np.exp(-dist_to_cbd * 0.08))
        crime_index = float(np.clip(
            crime_center + crime_periphery + rng.normal(0, 8),
            5, 95,
        ))

        # --- Gini coefficient: 0-1 (income inequality) ---
        # Higher in areas with mixed income levels (transitional zones)
        gini_base = ECUADOR_GINI
        # Transitional zones (2-6 km from CBD) tend to have higher inequality
        transitional_boost = 0.08 * np.exp(-((dist_to_cbd - 4) ** 2) / 8)
        gini_coefficient = float(np.clip(
            gini_base + transitional_boost + rng.normal(0, 0.03),
            0.25, 0.65,
        ))

        records.append({
            "cell_id": cell_id,
            "city": city_name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "population_density": round(pop_density, 1),
            "median_income_usd": round(median_income, 2),
            "education_level": round(education_level, 2),
            "employment_rate": round(employment_rate, 4),
            "internet_penetration": round(internet_penetration, 4),
            "crime_index": round(crime_index, 1),
            "gini_coefficient": round(gini_coefficient, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_socioeconomic(df: pd.DataFrame) -> pd.DataFrame:
    """Validate socioeconomic data integrity.

    Args:
        df: Raw socioeconomic DataFrame.

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

    # Validate ranges
    df.loc[df["population_density"] < 0, "population_density"] = 0
    df.loc[df["median_income_usd"] < 0, "median_income_usd"] = 0
    df["education_level"] = df["education_level"].clip(0, 25)
    df["employment_rate"] = df["employment_rate"].clip(0, 1)
    df["internet_penetration"] = df["internet_penetration"].clip(0, 1)
    df["crime_index"] = df["crime_index"].clip(0, 100)
    df["gini_coefficient"] = df["gini_coefficient"].clip(0, 1)

    if issues:
        for issue in issues:
            logger.warning(issue)
        logger.warning(
            "Validation removed %d of %d rows.",
            initial_count - len(df),
            initial_count,
        )

    logger.info("Validation complete: %d valid socioeconomic records.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute the socioeconomic ETL pipeline.

    Generates synthetic socioeconomic indicators (population density,
    income, education, employment, internet, crime, Gini) for all H3
    cells in Quito and Guayaquil.

    Args:
        config: Configuration dictionary. If None, loads from config.yaml.

    Returns:
        DataFrame containing all generated socioeconomic data.
    """
    if config is None:
        config = _load_config()

    seed = config.get("random_seed", DEFAULT_SEED)
    rng = np.random.default_rng(seed)

    resolution = config["h3"]["resolution"]
    cities = config["cities"]
    paths = config["paths"]

    all_socio: List[pd.DataFrame] = []

    for city_key, city_cfg in cities.items():
        city_name = city_cfg.get("name", city_key.capitalize())
        logger.info("Generating grid cells for %s...", city_name)
        cells = _generate_grid_cells(city_cfg, resolution, rng)

        logger.info("Generating socioeconomic indicators for %s...", city_name)
        city_socio = _generate_socioeconomic_for_cells(cells, city_name, city_cfg, rng)
        all_socio.append(city_socio)
        logger.info("Generated %d socioeconomic records for %s", len(city_socio), city_name)

    df = pd.concat(all_socio, ignore_index=True)
    logger.info("Total socioeconomic records before validation: %d", len(df))

    df = _validate_socioeconomic(df)

    # Save output
    output_path = paths.get("socioeconomic_csv", "data/processed/socioeconomic.csv")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("SOCIOECONOMIC ETL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cells: %d", len(df))
    logger.info("Cities: %s", df["city"].unique().tolist())
    logger.info(
        "Population density: %.0f avg (range: %.0f-%.0f)",
        df["population_density"].mean(),
        df["population_density"].min(),
        df["population_density"].max(),
    )
    logger.info(
        "Median income: $%.0f avg (range: $%.0f-$%.0f)",
        df["median_income_usd"].mean(),
        df["median_income_usd"].min(),
        df["median_income_usd"].max(),
    )
    logger.info(
        "Education level: %.1f years avg (range: %.1f-%.1f)",
        df["education_level"].mean(),
        df["education_level"].min(),
        df["education_level"].max(),
    )
    logger.info(
        "Employment rate: %.1f%% avg", df["employment_rate"].mean() * 100
    )
    logger.info(
        "Internet penetration: %.1f%% avg", df["internet_penetration"].mean() * 100
    )
    logger.info(
        "Crime index: %.1f avg (range: %.1f-%.1f)",
        df["crime_index"].mean(),
        df["crime_index"].min(),
        df["crime_index"].max(),
    )
    logger.info(
        "Gini coefficient: %.3f avg (range: %.3f-%.3f)",
        df["gini_coefficient"].mean(),
        df["gini_coefficient"].min(),
        df["gini_coefficient"].max(),
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
