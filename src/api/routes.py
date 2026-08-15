"""
API routes for the Radar de Valorización Urbana.

Provides endpoints for valuation prediction, SHAP explanation,
cell comparison, map data, and scenario simulation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    CompareResponse,
    ExplanationResponse,
    MapCell,
    MapResponse,
    PredictionResponse,
    ScenarioRequest,
    ScenarioResponse,
)
from .scenario_simulator import simulate_scenario

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Global state — loaded on startup
# ---------------------------------------------------------------------------
_features_df: pd.DataFrame | None = None
_project_root = Path(__file__).resolve().parent.parent.parent


def _load_features() -> pd.DataFrame:
    """Load features CSV into memory (lazy, cached)."""
    global _features_df
    if _features_df is not None:
        return _features_df

    csv_path = _project_root / "data" / "processed" / "features.csv"
    parquet_path = _project_root / "data" / "processed" / "features.parquet"

    if csv_path.exists():
        _features_df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s", len(_features_df), csv_path)
    elif parquet_path.exists():
        _features_df = pd.read_parquet(parquet_path)
        logger.info("Loaded %d rows from %s", len(_features_df), parquet_path)
    else:
        logger.warning("No features file found. API will return errors.")
        _features_df = pd.DataFrame()

    return _features_df


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Feature columns for explainability
# ---------------------------------------------------------------------------
_EXPLAIN_FEATURES = [
    "avg_travel_time_cbd_min", "traffic_index", "transit_stops_count",
    "walkability_score", "connectivity_index", "ndvi_mean", "built_up_index",
    "green_space_ratio", "night_lights_intensity", "land_use_mix",
    "hospitals_count", "schools_count", "supermarkets_count", "parks_count",
    "banks_count", "restaurants_count", "distance_to_cbd_km",
    "population_density", "median_income_usd", "education_level",
    "employment_rate", "internet_penetration", "crime_index",
    "accessibility_score", "service_density", "price_trend",
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/cells")
async def list_cells():
    """List all available grid cells with basic info."""
    df = _load_features()
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    cells = []
    for _, row in df.iterrows():
        cells.append({
            "cell_id": str(row.get("cell_id", "")),
            "lat": float(row.get("lat", row.get("cell_lat", 0))),
            "lon": float(row.get("lon", row.get("cell_lon", 0))),
            "city": str(row.get("city", "")),
            "avg_price": float(row.get("avg_price", 0)),
            "annualized_valuation": float(row.get("annualized_valuation", 0)),
        })
    return {"cells": cells, "count": len(cells)}


@router.get("/predict", response_model=PredictionResponse)
async def predict(
    cell_id: str = Query(..., description="Grid cell identifier"),
    horizon: int = Query(12, ge=1, le=120, description="Prediction horizon in months"),
):
    """Predict valuation for a specific cell.

    Uses historical trend adjusted by horizon. Falls back to
    statistics-based prediction if no trained model is available.
    """
    df = _load_features()
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    row = df[df["cell_id"] == cell_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Cell '{cell_id}' not found")

    row = row.iloc[0]
    base_valuation = float(row.get("annualized_valuation", 0.05))
    price_trend = float(row.get("price_trend", 0.05))

    # Scale by horizon (annualized → monthly compound)
    horizon_years = horizon / 12.0
    predicted = base_valuation * horizon_years

    # Simple uncertainty bounds
    uncertainty = 0.015 * math.sqrt(horizon_years)
    lower = predicted - uncertainty
    upper = predicted + uncertainty

    confidence = max(0.5, 1.0 - uncertainty / max(abs(predicted), 0.01))

    return PredictionResponse(
        cell_id=cell_id,
        predicted_valuation=round(predicted, 6),
        lower_bound=round(lower, 6),
        upper_bound=round(upper, 6),
        confidence=round(confidence, 4),
    )


@router.get("/explain", response_model=ExplanationResponse)
async def explain(
    cell_id: str = Query(..., description="Grid cell identifier"),
):
    """Return SHAP-like explanation for a cell's valuation prediction.

    Uses correlation-based feature importance when no trained SHAP
    model is available.
    """
    df = _load_features()
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    row = df[df["cell_id"] == cell_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Cell '{cell_id}' not found")

    row = row.iloc[0]
    base_value = float(df["annualized_valuation"].mean())

    # Compute pseudo-SHAP: deviation from mean attributed to each feature
    shap_values: dict[str, float] = {}
    available_features = [f for f in _EXPLAIN_FEATURES if f in df.columns]

    for feat in available_features:
        feat_mean = float(df[feat].mean())
        feat_std = float(df[feat].std())
        feat_val = float(row.get(feat, feat_mean))

        if feat_std > 0:
            # Normalized deviation × correlation proxy
            deviation = (feat_val - feat_mean) / feat_std
            # Weight by feature's correlation with target
            corr = float(df[feat].corr(df["annualized_valuation"])) if "annualized_valuation" in df.columns else 0
            corr = 0 if math.isnan(corr) else corr
            shap_val = deviation * corr * 0.02
        else:
            shap_val = 0.0

        shap_values[feat] = round(shap_val, 6)

    # Top drivers
    sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers = [
        {
            "feature": feat,
            "value": float(row.get(feat, 0)),
            "shap": val,
            "direction": "positive" if val >= 0 else "negative",
        }
        for feat, val in sorted_shap[:10]
    ]

    return ExplanationResponse(
        cell_id=cell_id,
        shap_values=shap_values,
        base_value=round(base_value, 6),
        top_drivers=top_drivers,
    )


@router.get("/compare", response_model=CompareResponse)
async def compare(
    cell_ids: str = Query(..., description="Comma-separated cell IDs"),
    horizon: int = Query(12, ge=1, le=120),
):
    """Compare valuations across multiple cells."""
    df = _load_features()
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    ids = [c.strip() for c in cell_ids.split(",") if c.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 cell IDs required")

    predictions: list[PredictionResponse] = []
    comparison_table: list[dict[str, Any]] = []

    for cid in ids:
        row = df[df["cell_id"] == cid]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Cell '{cid}' not found")
        row = row.iloc[0]

        base_val = float(row.get("annualized_valuation", 0.05))
        horizon_years = horizon / 12.0
        predicted = base_val * horizon_years
        uncertainty = 0.015 * math.sqrt(horizon_years)

        pred = PredictionResponse(
            cell_id=cid,
            predicted_valuation=round(predicted, 6),
            lower_bound=round(predicted - uncertainty, 6),
            upper_bound=round(predicted + uncertainty, 6),
            confidence=round(max(0.5, 1.0 - uncertainty / max(abs(predicted), 0.01)), 4),
        )
        predictions.append(pred)

        comparison_table.append({
            "cell_id": cid,
            "city": str(row.get("city", "")),
            "avg_price": float(row.get("avg_price", 0)),
            "price_per_m2": float(row.get("avg_price_per_m2", 0)),
            "predicted_valuation": pred.predicted_valuation,
            "annualized_valuation": base_val,
            "ndvi": float(row.get("ndvi_mean", 0)),
            "walkability": float(row.get("walkability_score", 0)),
            "schools": float(row.get("schools_count", 0)),
            "hospitals": float(row.get("hospitals_count", 0)),
        })

    return CompareResponse(
        project_id=f"compare-{'-'.join(ids[:4])}",
        cell_ids=ids,
        predictions=predictions,
        comparison_table=comparison_table,
    )


@router.get("/map", response_model=MapResponse)
async def map_data(
    bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    year: int = Query(2024, description="Valuation year"),
):
    """Get map data for a bounding box."""
    df = _load_features()
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    parts = [float(x) for x in bbox.split(",")]
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail="bbox must be min_lon,min_lat,max_lon,max_lat")

    min_lon, min_lat, max_lon, max_lat = parts

    # Filter cells within bbox
    lat_col = "lat" if "lat" in df.columns else "cell_lat"
    lon_col = "lon" if "lon" in df.columns else "cell_lon"

    mask = (
        (df[lat_col] >= min_lat)
        & (df[lat_col] <= max_lat)
        & (df[lon_col] >= min_lon)
        & (df[lon_col] <= max_lon)
    )
    filtered = df[mask]

    cells: list[MapCell] = []
    for _, row in filtered.iterrows():
        base_val = float(row.get("annualized_valuation", 0.05))
        year_offset = max(0, year - 2024)
        predicted = base_val * (1 + base_val) ** year_offset

        cells.append(MapCell(
            cell_id=str(row["cell_id"]),
            lat=float(row[lat_col]),
            lon=float(row[lon_col]),
            predicted_valuation=round(predicted, 6),
        ))

    return MapResponse(
        bbox=[min_lon, min_lat, max_lon, max_lat],
        year=year,
        cells=cells,
    )


@router.post("/simulate", response_model=ScenarioResponse)
async def simulate(req: ScenarioRequest):
    """Run a scenario simulation with macroeconomic parameters."""
    df = _load_features()
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature data not loaded")

    # Build baseline predictions for all cells
    baseline = []
    for _, row in df.iterrows():
        baseline.append({
            "cell_id": str(row["cell_id"]),
            "predicted_valuation": float(row.get("annualized_valuation", 0.05)),
            "lower_bound": float(row.get("annualized_valuation", 0.05)) * 0.8,
            "upper_bound": float(row.get("annualized_valuation", 0.05)) * 1.2,
        })

    result = simulate_scenario(
        baseline_predictions=baseline,
        interest_rate=req.interest_rate,
        gdp_growth=req.gdp_growth,
        migration_rate=req.migration_rate,
        infrastructure_investment=req.infrastructure_investment,
    )

    return ScenarioResponse(
        adjusted_valuations=result["adjusted_valuations"],
        impact_summary=result["impact_summary"],
    )
