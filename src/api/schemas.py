"""
Pydantic schemas for the Radar de Valorización Urbana API.

Defines request and response models for prediction, explanation,
comparison, map, and scenario endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Valuation prediction for a single grid cell."""

    cell_id: str = Field(..., description="H3 or grid cell identifier")
    predicted_valuation: float = Field(..., description="Predicted valuation in USD per m²")
    lower_bound: float = Field(..., description="Lower bound of prediction interval")
    upper_bound: float = Field(..., description="Upper bound of prediction interval")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score [0, 1]")


# ─── Explanation ──────────────────────────────────────────────────────────────

class ExplanationResponse(BaseModel):
    """SHAP-based explanation for a cell's valuation prediction."""

    cell_id: str = Field(..., description="Grid cell identifier")
    shap_values: dict[str, float] = Field(
        ..., description="Mapping of feature name to SHAP value"
    )
    base_value: float = Field(..., description="Base (expected) model output")
    top_drivers: list[dict[str, Any]] = Field(
        ..., description="Top contributing features with direction and magnitude"
    )


# ─── Compare ──────────────────────────────────────────────────────────────────

class CompareResponse(BaseModel):
    """Side-by-side comparison of valuations across multiple cells."""

    project_id: str = Field(..., description="Identifier for the comparison request")
    cell_ids: list[str] = Field(..., description="List of cell IDs in the comparison")
    predictions: list[PredictionResponse] = Field(
        ..., description="Predictions for each cell"
    )
    comparison_table: list[dict[str, Any]] = Field(
        ..., description="Tabular comparison with key metrics"
    )


# ─── Map ──────────────────────────────────────────────────────────────────────

class MapCell(BaseModel):
    """A single cell on the map layer."""

    cell_id: str
    lat: float
    lon: float
    predicted_valuation: float


class MapResponse(BaseModel):
    """Aggregated map data for a bounding box."""

    bbox: list[float] = Field(
        ..., description="Bounding box [min_lon, min_lat, max_lon, max_lat]"
    )
    year: int = Field(..., description="Valuation year")
    cells: list[MapCell] = Field(..., description="Cell predictions within the bbox")


# ─── Scenario Simulation ─────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    """Macroeconomic parameters for scenario simulation."""

    interest_rate: float = Field(
        0.05, ge=-0.1, le=0.5, description="Central bank interest rate (e.g. 0.05 = 5%)"
    )
    gdp_growth: float = Field(
        0.03, ge=-0.1, le=0.2, description="GDP growth rate (e.g. 0.03 = 3%)"
    )
    migration_rate: float = Field(
        0.0, ge=-0.05, le=0.1, description="Net migration rate as fraction of population"
    )
    infrastructure_investment: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Infrastructure investment level [0 = none, 1 = massive]",
    )


class ScenarioResponse(BaseModel):
    """Results of a scenario simulation."""

    adjusted_valuations: list[dict[str, Any]] = Field(
        ..., description="Per-cell adjusted valuations under the scenario"
    )
    impact_summary: dict[str, Any] = Field(
        ..., description="Aggregate impact metrics"
    )


# ─── Urban Intelligence Index ────────────────────────────────────────────────

class CellIndex(BaseModel):
    """Urban Intelligence Index for a single cell."""

    cell_id: str
    city: str
    lat: float
    lon: float
    index: float = Field(..., ge=0, le=100, description="Urban Intelligence Index [0-100]")
    accessibility: float
    services: float
    sustainability: float
    connectivity: float
    valuation: float


class IndexResponse(BaseModel):
    """Urban Intelligence Index for all cells."""

    cells: list[CellIndex]
    city_averages: dict[str, float]
    global_average: float


# ─── Global drivers ranking ─────────────────────────────────────────────────

class DriverRank(BaseModel):
    """One ranked urban driver."""

    feature: str
    importance: float


class DriversResponse(BaseModel):
    """Global ranking of urban valuation drivers."""

    drivers: list[DriverRank]
    method: str


# ─── Price trends ───────────────────────────────────────────────────────────

class TrendPoint(BaseModel):
    """A single price observation in a trend series."""

    year: int
    avg_price: float
    transactions: int


class TrendsResponse(BaseModel):
    """Historical price trend for a cell."""

    cell_id: str
    city: str
    series: list[TrendPoint]
    price_trend: float


# ─── Global summary ─────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    """Global statistics for the dashboard."""

    total_cells: int
    total_transactions: int
    avg_valuation: float
    avg_price: float
    top_cells: list[dict[str, Any]]
    city_stats: list[dict[str, Any]]
    index_distribution: dict[str, int]
