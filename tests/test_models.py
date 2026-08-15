"""
Tests for model modules — Radar de Valorización Urbana.

Tests tabular model training/prediction, calibration, and explainability.
Uses synthetic data to avoid dependency on the full ETL pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """Create a small synthetic features dataset for testing."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "cell_id": [f"test_cell_{i:03d}" for i in range(n)],
        "lat": rng.uniform(-0.45, 0.10, n),
        "lon": rng.uniform(-78.70, -78.30, n),
        "city": rng.choice(["Quito", "Guayaquil"], n),
        "avg_price": rng.uniform(50000, 300000, n),
        "avg_price_per_m2": rng.uniform(500, 2000, n),
        "avg_travel_time_cbd_min": rng.uniform(5, 80, n),
        "traffic_index": rng.uniform(0, 50, n),
        "transit_stops_count": rng.integers(0, 15, n),
        "walkability_score": rng.uniform(5, 80, n),
        "connectivity_index": rng.uniform(2, 60, n),
        "ndvi_mean": rng.uniform(-0.2, 0.8, n),
        "built_up_index": rng.uniform(0, 1, n),
        "green_space_ratio": rng.uniform(0, 1, n),
        "night_lights_intensity": rng.uniform(0, 80, n),
        "land_use_mix": rng.uniform(0, 1, n),
        "hospitals_count": rng.integers(0, 5, n),
        "schools_count": rng.integers(0, 10, n),
        "supermarkets_count": rng.integers(0, 8, n),
        "parks_count": rng.integers(0, 6, n),
        "banks_count": rng.integers(0, 5, n),
        "restaurants_count": rng.integers(0, 20, n),
        "distance_to_cbd_km": rng.uniform(1, 30, n),
        "population_density": rng.uniform(500, 5000, n),
        "median_income_usd": rng.uniform(400, 2500, n),
        "education_level": rng.uniform(0.3, 0.95, n),
        "employment_rate": rng.uniform(0.6, 0.98, n),
        "internet_penetration": rng.uniform(0.4, 0.95, n),
        "crime_index": rng.uniform(10, 60, n),
        "accessibility_score": rng.uniform(10, 80, n),
        "service_density": rng.uniform(0, 30, n),
        "price_trend": rng.uniform(-0.02, 0.35, n),
        "annualized_valuation": rng.uniform(-0.05, 0.25, n),
    })
    return df


# ─── Tests: Tabular Model ───────────────────────────────────────────────────

class TestTabularModel:
    """Tests for the tabular (LightGBM/sklearn) model module."""

    def test_model_can_train(self, synthetic_features):
        """Model should train without errors on synthetic data."""
        from src.models.tabular_model import train

        result = train(synthetic_features, n_splits=3)
        assert result is not None, "train() should return a result"
        assert len(result.models) == 3, "Should have 3 quantile models"
        assert len(result.feature_names) > 0, "Should have feature names"
        assert len(result.cv_metrics) > 0, "Should have CV metrics"

    def test_model_can_predict(self, synthetic_features):
        """Model should produce predictions after training."""
        from src.models.tabular_model import train, predict

        result = train(synthetic_features, n_splits=3)
        predictions = predict(synthetic_features, result=result)

        assert isinstance(predictions, pd.DataFrame), "Predictions should be a DataFrame"
        assert len(predictions) == len(synthetic_features), "One prediction per row"
        assert "pred_p50" in predictions.columns, "Should have median prediction column"
        assert np.all(np.isfinite(predictions.values)), "Predictions should be finite"

    def test_model_feature_importance(self, synthetic_features):
        """Model should return feature importance."""
        from src.models.tabular_model import train

        result = train(synthetic_features, n_splits=3)
        importance = result.feature_importance

        assert isinstance(importance, dict), "Importance should be a dict"
        assert len(importance) > 0, "Should have non-empty importance"
        for k, v in importance.items():
            assert isinstance(k, str), "Keys should be feature names"
            assert isinstance(v, (int, float)), "Values should be numeric"

    def test_model_quantile_ordering(self, synthetic_features):
        """Predictions should respect quantile ordering: p10 <= p50 <= p90."""
        from src.models.tabular_model import train, predict

        result = train(synthetic_features, n_splits=3)
        predictions = predict(synthetic_features, result=result)

        p10 = predictions["pred_p10"].values
        p50 = predictions["pred_p50"].values
        p90 = predictions["pred_p90"].values

        # Most rows should satisfy the ordering (allow some slack)
        assert np.mean(p10 <= p50) > 0.8, "p10 should be <= p50 for most rows"
        assert np.mean(p50 <= p90) > 0.8, "p50 should be <= p90 for most rows"


# ─── Tests: Calibration ─────────────────────────────────────────────────────

class TestCalibration:
    """Tests for the uncertainty calibration module."""

    def test_calibration_returns_intervals(self):
        """Calibration should produce valid prediction intervals."""
        from src.models.calibration import calibrate_predictions

        rng = np.random.default_rng(42)
        n_rows = 50
        n_samples = 100
        # MC-style samples: shape (n_samples, n_rows)
        base_predictions = rng.normal(loc=0.08, scale=0.02, size=(n_samples, n_rows))

        result = calibrate_predictions(base_predictions, n_samples=n_samples)

        assert "lower" in result, "Result should have 'lower' bound"
        assert "upper" in result, "Result should have 'upper' bound"
        assert "mean" in result, "Result should have 'mean'"

        lower = np.array(result["lower"])
        upper = np.array(result["upper"])

        assert lower.shape == (n_rows,), "Lower bound should have one value per row"
        assert np.all(lower <= upper), "Lower bound should be <= upper bound"
        assert np.all(np.isfinite(lower)), "Lower bounds should be finite"
        assert np.all(np.isfinite(upper)), "Upper bounds should be finite"

    def test_calibration_mean_close_to_input(self):
        """Calibrated mean should be close to input predictions."""
        from src.models.calibration import calibrate_predictions

        rng = np.random.default_rng(42)
        n_rows = 100
        n_samples = 100
        base = rng.normal(loc=0.10, scale=0.02, size=(n_samples, n_rows))
        result = calibrate_predictions(base, n_samples=n_samples)

        mean_vals = np.array(result["mean"])
        expected = base.mean(axis=0)
        assert np.allclose(mean_vals, expected, atol=0.02), \
            "Calibrated mean should be close to MC sample mean"


# ─── Tests: Explainability ──────────────────────────────────────────────────

class TestExplainability:
    """Tests for the SHAP explainability module."""

    def test_global_shap_values(self, synthetic_features):
        """global_shap_values should return arrays and feature names."""
        from src.models.tabular_model import train
        from src.models.explain import global_shap_values

        result = train(synthetic_features, n_splits=3)
        model = result.models[0.5]  # median quantile model

        from src.models.explain import _prepare_X
        X = _prepare_X(synthetic_features)

        shap_values, base_values, feature_names = global_shap_values(model, X)

        assert shap_values.shape[0] == len(X), "Should have one row per sample"
        assert shap_values.shape[1] == len(feature_names), "Should have one col per feature"
        assert len(feature_names) > 0, "Should return feature names"

    def test_rank_drivers(self, synthetic_features):
        """rank_drivers should return a ranked DataFrame."""
        from src.models.tabular_model import train
        from src.models.explain import global_shap_values, rank_drivers, _prepare_X

        result = train(synthetic_features, n_splits=3)
        model = result.models[0.5]
        X = _prepare_X(synthetic_features)

        shap_values, _, feature_names = global_shap_values(model, X)
        ranking = rank_drivers(shap_values, feature_names)

        assert isinstance(ranking, pd.DataFrame), "Should return a DataFrame"
        assert "feature" in ranking.columns, "Should have 'feature' column"
        assert "mean_abs_shap" in ranking.columns, "Should have 'mean_abs_shap' column"
        assert len(ranking) == len(feature_names), "Should rank all features"
        # Verify ranking is sorted descending
        assert ranking["mean_abs_shap"].is_monotonic_decreasing, \
            "Ranking should be sorted by mean_abs_shap descending"

    def test_explain_pipeline(self, synthetic_features):
        """Full explain() should return a dict with all expected keys."""
        from src.models.tabular_model import train
        from src.models.explain import explain
        import tempfile

        result = train(synthetic_features, n_splits=3)
        model = result.models[0.5]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "shap_test.parquet"
            explanation = explain(model, synthetic_features, output_path=str(output_path))

        assert "shap_values" in explanation
        assert "base_values" in explanation
        assert "feature_names" in explanation
        assert "driver_ranking" in explanation
        assert "local_explanations" in explanation
