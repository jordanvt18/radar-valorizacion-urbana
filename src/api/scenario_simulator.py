"""
Scenario simulator for urban valuation predictions.

Adjusts baseline valuations based on macroeconomic parameters using
sensitivity coefficients derived from econometric reasoning.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_sensitivity_coefficients() -> dict[str, float]:
    """Return sensitivity coefficients for each macroeconomic parameter.

    Each coefficient represents the expected percentage-point change in
    annualized valuation for a 1-unit change in the parameter.

    Returns:
        Dictionary mapping parameter names to sensitivity coefficients.
    """
    return {
        # Higher interest rates reduce property demand → lower valuation
        "interest_rate": -0.45,
        # Higher GDP growth increases purchasing power → higher valuation
        "gdp_growth": 0.30,
        # Net migration increases housing demand → higher valuation
        "migration_rate": 0.15,
        # Infrastructure investment improves area attractiveness → higher valuation
        "infrastructure_investment": 0.08,
    }


def simulate_scenario(
    baseline_predictions: list[dict[str, Any]],
    interest_rate: float = 0.055,
    gdp_growth: float = 0.025,
    migration_rate: float = 0.01,
    infrastructure_investment: float = 0.1,
) -> dict[str, Any]:
    """Adjust baseline valuations under a macroeconomic scenario.

    Args:
        baseline_predictions: List of dicts with at least 'cell_id' and
            'predicted_valuation' keys.
        interest_rate: Central bank interest rate (e.g. 0.055 = 5.5%).
        gdp_growth: GDP growth rate (e.g. 0.025 = 2.5%).
        migration_rate: Net migration as fraction of population.
        infrastructure_investment: Investment level [0, 1].

    Returns:
        Dictionary with 'adjusted_valuations' (list of dicts) and
        'impact_summary' (dict with aggregate metrics).
    """
    coeffs = get_sensitivity_coefficients()

    # Baseline assumptions (Ecuador 2024-ish defaults)
    baseline_interest = 0.055
    baseline_gdp = 0.025
    baseline_migration = 0.005
    baseline_infra = 0.05

    # Compute deltas from baseline
    delta_interest = interest_rate - baseline_interest
    delta_gdp = gdp_growth - baseline_gdp
    delta_migration = migration_rate - baseline_migration
    delta_infra = infrastructure_investment - baseline_infra

    # Total adjustment factor (additive in percentage points)
    adjustment = (
        coeffs["interest_rate"] * delta_interest
        + coeffs["gdp_growth"] * delta_gdp
        + coeffs["migration_rate"] * delta_migration
        + coeffs["infrastructure_investment"] * delta_infra
    )

    logger.info(
        "Scenario adjustment: %.4f (interest=%.4f, gdp=%.4f, migration=%.4f, infra=%.4f)",
        adjustment, delta_interest, delta_gdp, delta_migration, delta_infra,
    )

    adjusted_valuations: list[dict[str, Any]] = []
    baseline_values: list[float] = []
    adjusted_values: list[float] = []

    for pred in baseline_predictions:
        cell_id = pred.get("cell_id", "unknown")
        base_val = float(pred.get("predicted_valuation", 0.0))

        # Apply adjustment, clamped to [-0.5, 1.0] to avoid extreme values
        adjusted_val = base_val + adjustment
        adjusted_val = float(np.clip(adjusted_val, -0.5, 1.0))

        # Adjust confidence bounds proportionally
        lower = float(pred.get("lower_bound", base_val * 0.8)) + adjustment * 0.8
        upper = float(pred.get("upper_bound", base_val * 1.2)) + adjustment * 0.8
        lower = float(np.clip(lower, -0.5, 1.0))
        upper = float(np.clip(upper, -0.5, 1.5))

        change_pct = ((adjusted_val - base_val) / max(abs(base_val), 1e-6)) * 100

        adjusted_valuations.append({
            "cell_id": cell_id,
            "baseline_valuation": round(base_val, 6),
            "adjusted_valuation": round(adjusted_val, 6),
            "change_pct": round(change_pct, 2),
            "lower_bound": round(lower, 6),
            "upper_bound": round(upper, 6),
        })

        baseline_values.append(base_val)
        adjusted_values.append(adjusted_val)

    # Aggregate impact summary
    base_mean = float(np.mean(baseline_values)) if baseline_values else 0.0
    adj_mean = float(np.mean(adjusted_values)) if adjusted_values else 0.0
    base_std = float(np.std(baseline_values)) if baseline_values else 0.0
    adj_std = float(np.std(adjusted_values)) if adjusted_values else 0.0

    n_positive = sum(1 for v in adjusted_values if v > 0)
    n_negative = sum(1 for v in adjusted_values if v <= 0)

    impact_summary = {
        "mean_baseline_valuation": round(base_mean, 6),
        "mean_adjusted_valuation": round(adj_mean, 6),
        "mean_change_pct": round(((adj_mean - base_mean) / max(abs(base_mean), 1e-6)) * 100, 2),
        "std_baseline": round(base_std, 6),
        "std_adjusted": round(adj_std, 6),
        "n_cells_positive": n_positive,
        "n_cells_negative": n_negative,
        "n_cells_total": len(adjusted_valuations),
        "parameters": {
            "interest_rate": interest_rate,
            "gdp_growth": gdp_growth,
            "migration_rate": migration_rate,
            "infrastructure_investment": infrastructure_investment,
        },
        "adjustment_factor": round(adjustment, 6),
    }

    logger.info(
        "Simulation complete: mean change %.2f%%, %d positive, %d negative",
        impact_summary["mean_change_pct"], n_positive, n_negative,
    )

    return {
        "adjusted_valuations": adjusted_valuations,
        "impact_summary": impact_summary,
    }


if __name__ == "__main__":
    # Quick smoke test
    demo = [
        {"cell_id": "demo1", "predicted_valuation": 0.065, "lower_bound": 0.04, "upper_bound": 0.09},
        {"cell_id": "demo2", "predicted_valuation": 0.032, "lower_bound": 0.01, "upper_bound": 0.055},
        {"cell_id": "demo3", "predicted_valuation": -0.01, "lower_bound": -0.03, "upper_bound": 0.01},
    ]
    result = simulate_scenario(demo, interest_rate=0.08, gdp_growth=0.04, migration_rate=0.02, infrastructure_investment=0.3)
    import json
    print(json.dumps(result, indent=2))
