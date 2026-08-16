"""
Tests for the FastAPI API — Radar de Valorización Urbana.

Tests all endpoints using httpx AsyncClient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_features(tmp_path) -> Path:
    """Create a small features CSV for testing."""
    rng = __import__("numpy").random.default_rng(42)
    n = 20
    df = pd.DataFrame({
        "cell_id": [f"test_{i:03d}" for i in range(n)],
        "lat": rng.uniform(-0.4, 0.1, n),
        "lon": rng.uniform(-78.7, -78.3, n),
        "city": rng.choice(["Quito", "Guayaquil"], n),
        "avg_price": rng.uniform(50000, 200000, n),
        "avg_price_per_m2": rng.uniform(500, 1500, n),
        "annualized_valuation": rng.uniform(-0.02, 0.15, n),
        "price_trend": rng.uniform(-0.01, 0.2, n),
        "avg_travel_time_cbd_min": rng.uniform(5, 70, n),
        "traffic_index": rng.uniform(0, 40, n),
        "transit_stops_count": rng.integers(0, 10, n),
        "walkability_score": rng.uniform(10, 70, n),
        "connectivity_index": rng.uniform(5, 50, n),
        "ndvi_mean": rng.uniform(0, 0.7, n),
        "built_up_index": rng.uniform(0.1, 0.9, n),
        "green_space_ratio": rng.uniform(0.1, 0.8, n),
        "night_lights_intensity": rng.uniform(5, 60, n),
        "land_use_mix": rng.uniform(0.2, 0.9, n),
        "hospitals_count": rng.integers(0, 3, n),
        "schools_count": rng.integers(0, 8, n),
        "supermarkets_count": rng.integers(0, 5, n),
        "parks_count": rng.integers(0, 4, n),
        "banks_count": rng.integers(0, 3, n),
        "restaurants_count": rng.integers(0, 15, n),
        "distance_to_cbd_km": rng.uniform(2, 25, n),
        "population_density": rng.uniform(800, 4000, n),
        "median_income_usd": rng.uniform(500, 2000, n),
        "education_level": rng.uniform(0.5, 0.9, n),
        "employment_rate": rng.uniform(0.7, 0.95, n),
        "internet_penetration": rng.uniform(0.5, 0.9, n),
        "crime_index": rng.uniform(15, 50, n),
        "accessibility_score": rng.uniform(15, 70, n),
        "service_density": rng.uniform(2, 20, n),
    })

    # Write to an isolated temp path (never touch the project's real data)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)

    # Point the API at the temp file and reset its cache
    import os
    import src.api.routes as routes
    os.environ["RADAR_FEATURES_PATH"] = str(csv_path)
    routes._features_df = None  # Reset cache

    return csv_path


@pytest_asyncio.fixture
async def client(sample_features):
    """Create an async HTTP client for the FastAPI app."""
    from src.api.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─── Tests ──────────────────────────────────────────────────────────────────

class TestHealth:
    """Health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        """GET /health should return 200 with status ok."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestCells:
    """Cells listing endpoint."""

    @pytest.mark.asyncio
    async def test_cells_returns_list(self, client):
        """GET /cells should return a list of cells."""
        resp = await client.get("/cells")
        assert resp.status_code == 200
        data = resp.json()
        assert "cells" in data
        assert "count" in data
        assert data["count"] > 0
        assert len(data["cells"]) == data["count"]

    @pytest.mark.asyncio
    async def test_cells_have_required_fields(self, client):
        """Each cell should have cell_id, lat, lon, city."""
        resp = await client.get("/cells")
        cells = resp.json()["cells"]
        for cell in cells[:3]:
            assert "cell_id" in cell
            assert "lat" in cell
            assert "lon" in cell
            assert "city" in cell


class TestPredict:
    """Prediction endpoint."""

    @pytest.mark.asyncio
    async def test_predict_valid_cell(self, client):
        """GET /predict with valid cell_id should return prediction."""
        # First get a valid cell_id
        cells_resp = await client.get("/cells")
        cell_id = cells_resp.json()["cells"][0]["cell_id"]

        resp = await client.get(f"/predict?cell_id={cell_id}&horizon=12")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cell_id"] == cell_id
        assert "predicted_valuation" in data
        assert "lower_bound" in data
        assert "upper_bound" in data
        assert "confidence" in data
        assert data["lower_bound"] <= data["predicted_valuation"]
        assert data["upper_bound"] >= data["predicted_valuation"]

    @pytest.mark.asyncio
    async def test_predict_invalid_cell(self, client):
        """GET /predict with invalid cell_id should return 404."""
        resp = await client.get("/predict?cell_id=nonexistent_cell&horizon=12")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_default_horizon(self, client):
        """GET /predict without horizon should use default."""
        cells_resp = await client.get("/cells")
        cell_id = cells_resp.json()["cells"][0]["cell_id"]

        resp = await client.get(f"/predict?cell_id={cell_id}")
        assert resp.status_code == 200


class TestExplain:
    """Explainability endpoint."""

    @pytest.mark.asyncio
    async def test_explain_valid_cell(self, client):
        """GET /explain with valid cell_id should return SHAP values."""
        cells_resp = await client.get("/cells")
        cell_id = cells_resp.json()["cells"][0]["cell_id"]

        resp = await client.get(f"/explain?cell_id={cell_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cell_id"] == cell_id
        assert "shap_values" in data
        assert "base_value" in data
        assert "top_drivers" in data
        assert isinstance(data["shap_values"], dict)
        assert isinstance(data["top_drivers"], list)

    @pytest.mark.asyncio
    async def test_explain_invalid_cell(self, client):
        """GET /explain with invalid cell_id should return 404."""
        resp = await client.get("/explain?cell_id=nonexistent_cell")
        assert resp.status_code == 404


class TestCompare:
    """Comparison endpoint."""

    @pytest.mark.asyncio
    async def test_compare_multiple_cells(self, client):
        """GET /compare with multiple cells should return comparison."""
        cells_resp = await client.get("/cells")
        cells = cells_resp.json()["cells"]
        cell_ids = ",".join([c["cell_id"] for c in cells[:3]])

        resp = await client.get(f"/compare?cell_ids={cell_ids}&horizon=12")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "comparison_table" in data
        assert len(data["predictions"]) == 3

    @pytest.mark.asyncio
    async def test_compare_single_cell_fails(self, client):
        """GET /compare with one cell should return 400."""
        cells_resp = await client.get("/cells")
        cell_id = cells_resp.json()["cells"][0]["cell_id"]

        resp = await client.get(f"/compare?cell_ids={cell_id}")
        assert resp.status_code == 400


class TestSimulate:
    """Scenario simulation endpoint."""

    @pytest.mark.asyncio
    async def test_simulate_default_params(self, client):
        """POST /simulate with default params should return adjusted valuations."""
        resp = await client.post("/simulate", json={
            "interest_rate": 0.05,
            "gdp_growth": 0.03,
            "migration_rate": 0.01,
            "infrastructure_investment": 0.1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "adjusted_valuations" in data
        assert "impact_summary" in data
        assert len(data["adjusted_valuations"]) > 0

    @pytest.mark.asyncio
    async def test_simulate_high_interest(self, client):
        """High interest rate should decrease valuations."""
        # Baseline
        resp_base = await client.post("/simulate", json={
            "interest_rate": 0.05,
            "gdp_growth": 0.025,
            "migration_rate": 0.005,
            "infrastructure_investment": 0.05,
        })
        base_mean = resp_base.json()["impact_summary"]["mean_adjusted_valuation"]

        # High interest
        resp_high = await client.post("/simulate", json={
            "interest_rate": 0.15,
            "gdp_growth": 0.025,
            "migration_rate": 0.005,
            "infrastructure_investment": 0.05,
        })
        high_mean = resp_high.json()["impact_summary"]["mean_adjusted_valuation"]

        assert high_mean < base_mean, "High interest should reduce valuations"


class TestMap:
    """Map data endpoint."""

    @pytest.mark.asyncio
    async def test_map_returns_cells(self, client):
        """GET /map with wide bbox should return cells."""
        # Bbox covering Ecuador
        resp = await client.get("/map?bbox=-81,2,-75,-5&year=2024")
        assert resp.status_code == 200
        data = resp.json()
        assert "cells" in data
        assert "bbox" in data
        assert "year" in data
