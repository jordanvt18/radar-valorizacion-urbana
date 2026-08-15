"""
Feature engineering module for Radar de Valorización Urbana.

Loads all 5 ETL outputs (transactions, mobility, satellite, services,
socioeconomic), aggregates transaction data per cell, computes the
valuation target, merges all data sources, generates derived features,
and saves the final feature matrix.

Usage:
    python -m src.features.build_features
    python src/features/build_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
# Defaults
# ---------------------------------------------------------------------------
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
        "paths": {
            "processed_dir": "data/processed",
            "transactions_csv": "data/processed/transactions.csv",
            "transactions_parquet": "data/processed/transactions.parquet",
            "mobility_csv": "data/processed/mobility.csv",
            "satellite_csv": "data/processed/satellite_features.csv",
            "services_csv": "data/processed/services.csv",
            "socioeconomic_csv": "data/processed/socioeconomic.csv",
            "features_csv": "data/processed/features.csv",
            "features_parquet": "data/processed/features.parquet",
        },
        "random_seed": 42,
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_csv_or_parquet(
    csv_path: str,
    parquet_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load data from parquet (preferred) or CSV (fallback).

    Args:
        csv_path: Path to CSV file.
        parquet_path: Optional path to parquet file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If neither file exists.
    """
    if parquet_path and PARQUET_AVAILABLE and Path(parquet_path).exists():
        logger.info("Loading parquet: %s", parquet_path)
        return pd.read_parquet(parquet_path)

    if Path(csv_path).exists():
        logger.info("Loading CSV: %s", csv_path)
        return pd.read_csv(csv_path)

    if parquet_path and Path(parquet_path).exists():
        logger.info("Loading parquet (fallback): %s", parquet_path)
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(f"Could not find data file at {csv_path} or {parquet_path}")


# ---------------------------------------------------------------------------
# Transaction aggregation
# ---------------------------------------------------------------------------
def _aggregate_transactions(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data per cell_id.

    Computes average price, price trend, transaction count, and average
    price per square meter for each cell.

    Args:
        df_tx: Raw transactions DataFrame.

    Returns:
        Aggregated DataFrame with one row per cell_id.
    """
    logger.info("Aggregating transactions for %d total records...", len(df_tx))

    # Ensure year column exists
    if "year" not in df_tx.columns:
        df_tx["year"] = pd.to_datetime(df_tx["date"]).dt.year

    # Per-cell aggregation
    agg = df_tx.groupby("cell_id").agg(
        avg_price=("price_usd", "mean"),
        price_std=("price_usd", "std"),
        transaction_count=("price_usd", "count"),
        avg_price_per_m2=("price_per_m2", "mean"),
        avg_area_m2=("area_m2", "mean"),
        first_year=("year", "min"),
        last_year=("year", "max"),
    ).reset_index()

    # Price trend: linear regression slope of yearly average prices
    # Positive slope = appreciating area
    def _compute_trend(group: pd.DataFrame) -> float:
        """Compute price trend (slope) for a cell's transactions over time.

        Args:
            group: Transaction subset for one cell.

        Returns:
            Normalized price trend slope.
        """
        if len(group) < 2:
            return 0.0
        yearly_avg = group.groupby("year")["price_usd"].mean().sort_index()
        if len(yearly_avg) < 2:
            return 0.0
        years = yearly_avg.index.values.astype(float)
        prices = yearly_avg.values.astype(float)
        # Normalize to avoid scale issues
        if prices.mean() == 0:
            return 0.0
        x = years - years[0]
        y = prices / prices[0]
        # Simple linear regression: y = a*x + b
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x ** 2) - np.sum(x) ** 2 + 1e-10
        )
        return float(slope)

    trends = (
        df_tx.sort_values(["cell_id", "date"])
        .groupby("cell_id")
        .apply(_compute_trend, include_groups=False)
        .reset_index()
    )
    trends.columns = ["cell_id", "price_trend"]
    agg = agg.merge(trends, on="cell_id", how="left")
    agg["price_trend"] = agg["price_trend"].fillna(0.0)

    # Compute target: annualized valuation growth
    # annualized_valuation = (P_t+T / P_t)^(1/T) - 1
    def _compute_annualized_return(row: pd.Series) -> float:
        """Compute annualized return from first to last year.

        Args:
            row: Aggregated row with first_year, last_year, avg_price.

        Returns:
            Annualized return rate.
        """
        t = row["last_year"] - row["first_year"]
        if t <= 0:
            return 0.0
        # Use first vs last year average prices
        cell_tx = df_tx[df_tx["cell_id"] == row["cell_id"]]
        first_year_price = cell_tx[cell_tx["year"] == row["first_year"]]["price_usd"].mean()
        last_year_price = cell_tx[cell_tx["year"] == row["last_year"]]["price_usd"].mean()
        if first_year_price is None or last_year_price is None or first_year_price <= 0:
            return 0.0
        ratio = last_year_price / first_year_price
        if ratio <= 0:
            return 0.0
        return float(ratio ** (1.0 / t) - 1.0)

    logger.info("Computing annualized valuation target...")
    agg["annualized_valuation"] = agg.apply(_compute_annualized_return, axis=1)

    # Round numeric columns
    agg["avg_price"] = agg["avg_price"].round(2)
    agg["price_std"] = agg["price_std"].fillna(0).round(2)
    agg["avg_price_per_m2"] = agg["avg_price_per_m2"].round(2)
    agg["avg_area_m2"] = agg["avg_area_m2"].round(1)
    agg["annualized_valuation"] = agg["annualized_valuation"].round(6)
    agg["price_trend"] = agg["price_trend"].round(6)

    logger.info(
        "Aggregated to %d cells (avg %.1f transactions/cell)",
        len(agg),
        agg["transaction_count"].mean(),
    )
    return agg


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------
def _compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate derived features from merged data.

    Args:
        df: Merged DataFrame with all ETL outputs.

    Returns:
        DataFrame with additional derived feature columns.
    """
    logger.info("Computing derived features...")

    # --- Accessibility score ---
    # Composite: weighted combination of walkability, transit, and connectivity
    # Normalized to 0-100
    if all(c in df.columns for c in ["walkability_score", "transit_stops_count", "connectivity_index"]):
        max_transit = df["transit_stops_count"].quantile(0.95)
        if max_transit > 0:
            transit_norm = (df["transit_stops_count"] / max_transit * 100).clip(0, 100)
        else:
            transit_norm = pd.Series(0, index=df.index)
        df["accessibility_score"] = (
            0.4 * df["walkability_score"]
            + 0.3 * transit_norm
            + 0.3 * df["connectivity_index"]
        ).round(2)
    else:
        df["accessibility_score"] = 0.0

    # --- Service density ---
    # Total services per cell normalized by area (proxy: per cell)
    service_cols = [
        "hospitals_count", "schools_count", "supermarkets_count",
        "parks_count", "banks_count", "restaurants_count",
    ]
    available_service_cols = [c for c in service_cols if c in df.columns]
    if available_service_cols:
        df["service_density"] = df[available_service_cols].sum(axis=1).astype(float)
    else:
        df["service_density"] = 0.0

    # --- NDVI (alias for convenience) ---
    if "ndvi_mean" in df.columns:
        df["ndvi"] = df["ndvi_mean"]
    else:
        df["ndvi"] = 0.0

    # --- Noise proxy ---
    # Proxy for urban noise: combination of traffic index, population density,
    # and restaurant/commercial density
    noise_components = []
    if "traffic_index" in df.columns:
        noise_components.append(df["traffic_index"])
    if "population_density" in df.columns:
        # Normalize population density to 0-100 scale
        max_pop = df["population_density"].quantile(0.95)
        if max_pop > 0:
            noise_components.append((df["population_density"] / max_pop * 100).clip(0, 100))
    if "restaurants_count" in df.columns:
        max_rest = df["restaurants_count"].quantile(0.95)
        if max_rest > 0:
            noise_components.append((df["restaurants_count"] / max_rest * 100).clip(0, 100))

    if noise_components:
        df["noise_proxy"] = (sum(noise_components) / len(noise_components)).round(2)
    else:
        df["noise_proxy"] = 0.0

    # --- Valuation trend ---
    # Alias and enhancement of price_trend
    if "price_trend" in df.columns:
        df["valuation_trend"] = df["price_trend"]
    else:
        df["valuation_trend"] = 0.0

    # --- Connectivity index (from mobility if available) ---
    if "connectivity_index" not in df.columns and "connectivity" in df.columns:
        df["connectivity_index"] = df["connectivity"]
    elif "connectivity_index" not in df.columns:
        df["connectivity_index"] = 0.0

    logger.info("Derived features computed: %s",
                [c for c in ["accessibility_score", "service_density", "ndvi",
                             "noise_proxy", "valuation_trend", "connectivity_index"]
                 if c in df.columns])

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_features(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute the feature engineering pipeline.

    Loads all ETL outputs, aggregates transactions per cell, computes
    the valuation target, merges all data sources, generates derived
    features, and saves the final feature matrix.

    Args:
        config: Configuration dictionary. If None, loads from config.yaml.

    Returns:
        DataFrame containing the final feature matrix.
    """
    if config is None:
        config = _load_config()

    paths = config["paths"]
    processed_dir = Path(paths.get("processed_dir", "data/processed"))

    # --- Load all ETL outputs ---
    logger.info("=" * 60)
    logger.info("LOADING ETL OUTPUTS")
    logger.info("=" * 60)

    # Transactions
    df_tx = _load_csv_or_parquet(
        csv_path=str(processed_dir / "transactions.csv"),
        parquet_path=str(processed_dir / "transactions.parquet"),
    )
    logger.info("Loaded %d transactions", len(df_tx))

    # Mobility
    mobility_path = paths.get("mobility_csv", str(processed_dir / "mobility.csv"))
    df_mobility = pd.read_csv(mobility_path) if Path(mobility_path).exists() else None
    if df_mobility is not None:
        logger.info("Loaded %d mobility records", len(df_mobility))
    else:
        logger.warning("Mobility data not found at %s — skipping", mobility_path)
        df_mobility = pd.DataFrame()

    # Satellite
    satellite_path = paths.get("satellite_csv", str(processed_dir / "satellite_features.csv"))
    df_satellite = pd.read_csv(satellite_path) if Path(satellite_path).exists() else None
    if df_satellite is not None:
        logger.info("Loaded %d satellite records", len(df_satellite))
    else:
        logger.warning("Satellite data not found at %s — skipping", satellite_path)
        df_satellite = pd.DataFrame()

    # Services
    services_path = paths.get("services_csv", str(processed_dir / "services.csv"))
    df_services = pd.read_csv(services_path) if Path(services_path).exists() else None
    if df_services is not None:
        logger.info("Loaded %d service records", len(df_services))
    else:
        logger.warning("Services data not found at %s — skipping", services_path)
        df_services = pd.DataFrame()

    # Socioeconomic
    socio_path = paths.get("socioeconomic_csv", str(processed_dir / "socioeconomic.csv"))
    df_socio = pd.read_csv(socio_path) if Path(socio_path).exists() else None
    if df_socio is not None:
        logger.info("Loaded %d socioeconomic records", len(df_socio))
    else:
        logger.warning("Socioeconomic data not found at %s — skipping", socio_path)
        df_socio = pd.DataFrame()

    # --- Aggregate transactions ---
    logger.info("=" * 60)
    logger.info("AGGREGATING TRANSACTIONS")
    logger.info("=" * 60)
    df_tx_agg = _aggregate_transactions(df_tx)

    # --- Merge all data ---
    logger.info("=" * 60)
    logger.info("MERGING DATA SOURCES")
    logger.info("=" * 60)

    # Start from the canonical grid so every cell has lat/lon and city,
    # then attach transaction aggregates and all ETL feature tables.
    from src.etl.grid import load_or_create_grid

    df = load_or_create_grid(config)
    df = df.merge(df_tx_agg, on="cell_id", how="left")

    # Merge mobility (drop duplicate city/lat/lon, keep mobility-specific columns)
    if not df_mobility.empty:
        mobility_cols = ["cell_id", "avg_travel_time_cbd_min", "traffic_index",
                         "peak_hour_speed_kmh", "transit_stops_count",
                         "walkability_score", "connectivity_index"]
        available_mob_cols = [c for c in mobility_cols if c in df_mobility.columns]
        df = df.merge(df_mobility[available_mob_cols], on="cell_id", how="left")
        logger.info("Merged mobility data: %d columns", len(available_mob_cols))

    # Merge satellite features
    if not df_satellite.empty:
        sat_cols = [c for c in df_satellite.columns if c not in ("city", "lat", "lon")]
        df = df.merge(df_satellite[sat_cols], on="cell_id", how="left")
        logger.info("Merged satellite data: %d columns", len(sat_cols))

    # Merge services
    if not df_services.empty:
        svc_cols = [c for c in df_services.columns if c not in ("city", "lat", "lon")]
        df = df.merge(df_services[svc_cols], on="cell_id", how="left")
        logger.info("Merged services data: %d columns", len(svc_cols))

    # Merge socioeconomic
    if not df_socio.empty:
        socio_cols = [c for c in df_socio.columns if c not in ("city", "lat", "lon")]
        df = df.merge(df_socio[socio_cols], on="cell_id", how="left")
        logger.info("Merged socioeconomic data: %d columns", len(socio_cols))

    # Add city info from transactions
    city_map = df_tx[["cell_id", "city"]].drop_duplicates("cell_id").set_index("cell_id")["city"]
    if "city" not in df.columns:
        df["city"] = df["cell_id"].map(city_map)

    logger.info("Merged dataset: %d rows × %d columns", df.shape[0], df.shape[1])

    # --- Compute derived features ---
    logger.info("=" * 60)
    logger.info("COMPUTING DERIVED FEATURES")
    logger.info("=" * 60)
    df = _compute_derived_features(df)

    # --- Fill remaining NaNs with reasonable defaults ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            logger.debug("Filled NaNs in '%s' with median: %.4f", col, median_val)

    # --- Save outputs ---
    logger.info("=" * 60)
    logger.info("SAVING FEATURES")
    logger.info("=" * 60)

    output_csv = paths.get("features_csv", str(processed_dir / "features.csv"))
    output_parquet = paths.get("features_parquet", str(processed_dir / "features.parquet"))
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(output_csv, index=False)
    logger.info("Saved %d rows × %d columns to %s", df.shape[0], df.shape[1], output_csv)

    # Save parquet if available
    if PARQUET_AVAILABLE:
        df.to_parquet(output_parquet, index=False)
        logger.info("Saved %d rows to %s", df.shape[0], output_parquet)
    else:
        logger.info("Parquet unavailable — CSV only")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cells: %d", len(df))
    logger.info("Total features: %d", df.shape[1])
    logger.info("Cities: %s", df["city"].unique().tolist() if "city" in df.columns else "N/A")
    logger.info(
        "Avg price: $%.0f (range: $%.0f-$%.0f)",
        df["avg_price"].mean(),
        df["avg_price"].min(),
        df["avg_price"].max(),
    )
    logger.info(
        "Avg transactions per cell: %.1f",
        df["transaction_count"].mean(),
    )
    logger.info(
        "Annualized valuation: %.4f avg (range: %.4f-%.4f)",
        df["annualized_valuation"].mean(),
        df["annualized_valuation"].min(),
        df["annualized_valuation"].max(),
    )
    logger.info(
        "Price trend: %.6f avg (range: %.6f-%.6f)",
        df["price_trend"].mean(),
        df["price_trend"].min(),
        df["price_trend"].max(),
    )
    logger.info(
        "Accessibility score: %.1f avg (range: %.1f-%.1f)",
        df["accessibility_score"].mean(),
        df["accessibility_score"].min(),
        df["accessibility_score"].max(),
    )
    logger.info("Feature columns: %s", df.columns.tolist())
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    build_features()
