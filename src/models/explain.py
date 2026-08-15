"""
SHAP explainability for the urban valuation model.

Provides:
  * Global SHAP summary values.
  * Per-cell local SHAP values.
  * Ranking of urban valuation drivers.
  * Fallback to ``feature_importances_`` when SHAP is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional SHAP import
# ---------------------------------------------------------------------------

try:
    import shap

    _HAS_SHAP = True
except ImportError:  # pragma: no cover
    _HAS_SHAP = False
    shap = None  # type: ignore[assignment]
    logger.warning("shap not installed — falling back to feature_importances_")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "annualized_valuation"
EXCLUDE_COLS: Tuple[str, ...] = ("cell_id", "lat", "lon", TARGET_COL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix *X* with non-feature columns dropped.

    Mirrors ``tabular_model._split_features_target``: drops explicit
    exclude columns (cell_id, lat, lon, target) plus any non-numeric
    column (e.g. ``city``) so feature alignment with trained models
    is exact.
    """
    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop)
    return X.select_dtypes(include=("number", "bool"))


def _fallback_importance(model: Any, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (values, base_values) from ``feature_importances_``.

    SHAP values are simulated by scaling the normalised importances
    to the mean prediction.  This is a rough approximation used only
    when the ``shap`` package is unavailable.
    """
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_.astype(float)
        total = fi.sum() or 1.0
        fi_norm = fi / total
        base = 0.0
        # Per-row "shap" — proportional to feature value rank
        return fi_norm, np.array([base])
    raise AttributeError("Model has neither SHAP explainer nor feature_importances_")


# ---------------------------------------------------------------------------
# Global SHAP
# ---------------------------------------------------------------------------


def global_shap_values(
    model: Any,
    X: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    n_background: int = 100,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute global SHAP values for *model* on *X*.

    Parameters
    ----------
    model:
        A trained model (LightGBM, sklearn, etc.).
    X:
        Feature DataFrame.
    background:
        Background dataset for the explainer.  If *None* a subsample of
        *X* is used.
    n_background:
        Number of background samples.

    Returns
    -------
    (shap_values, base_values, feature_names)
    """
    feature_names = X.columns.tolist()

    if _HAS_SHAP:
        if background is None:
            bg_n = min(n_background, len(X))
            background = X.sample(n=bg_n, random_state=42)

        # Choose explainer
        model_class = model.__class__.__name__
        if "LGBM" in model_class:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        elif hasattr(model, "estimators_") or "GradientBoosting" in model_class:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # quantile models may return a list

        base_values = explainer.expected_value
        if isinstance(base_values, (list, np.ndarray)) and len(np.atleast_1d(base_values)) > 1:
            base_values = float(np.atleast_1d(base_values)[0])

        logger.info("SHAP global values computed | shape=%s", np.array(shap_values).shape)
        return np.array(shap_values), np.atleast_1d(base_values), feature_names

    # Fallback ---------------------------------------------------------------
    logger.info("Using feature_importance_ fallback for global explanation")
    fi, base = _fallback_importance(model, feature_names)
    # Broadcast to per-row
    shap_values = np.tile(fi, (len(X), 1))
    return shap_values, base, feature_names


# ---------------------------------------------------------------------------
# Per-cell (local) SHAP
# ---------------------------------------------------------------------------


def local_shap_values(
    model: Any,
    X: pd.DataFrame,
    cell_ids: Optional[pd.Series] = None,
    background: Optional[pd.DataFrame] = None,
    n_background: int = 100,
) -> pd.DataFrame:
    """Compute per-cell SHAP values and return as a tidy DataFrame.

    Parameters
    ----------
    model:
        Trained model.
    X:
        Feature DataFrame.
    cell_ids:
        Series of cell identifiers.  If *None* the DataFrame index is used.
    background:
        Background data for SHAP explainer.

    Returns
    -------
    DataFrame with columns: ``cell_id``, ``feature``, ``shap_value``.
    """
    shap_values, _, feature_names = global_shap_values(model, X, background=background, n_background=n_background)

    if cell_ids is None:
        cell_ids = pd.Series(X.index, name="cell_id")
    else:
        cell_ids = cell_ids.reset_index(drop=True)

    n_rows, n_features = shap_values.shape
    records: List[Dict[str, Any]] = []
    for i in range(n_rows):
        cid = cell_ids.iloc[i] if i < len(cell_ids) else i
        for j in range(n_features):
            records.append(
                {
                    "cell_id": cid,
                    "feature": feature_names[j],
                    "shap_value": float(shap_values[i, j]),
                    "feature_value": float(X.iloc[i, j]),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Urban driver ranking
# ---------------------------------------------------------------------------


def rank_drivers(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value.

    Returns
    -------
    DataFrame with columns ``feature``, ``mean_abs_shap``, ``rank``.
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranking = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_shap_values(
    shap_values: np.ndarray,
    feature_names: List[str],
    cell_ids: Optional[pd.Series],
    output_path: Union[str, Path] = "outputs/shap_values.parquet",
) -> Path:
    """Persist SHAP values to a parquet file.

    If ``pyarrow`` is not installed, falls back to CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_features = shap_values.shape
    if cell_ids is None:
        cell_ids = pd.Series(range(n_rows), name="cell_id")

    df = pd.DataFrame(shap_values, columns=feature_names)
    df.insert(0, "cell_id", cell_ids.reset_index(drop=True).values)

    try:
        df.to_parquet(output_path, index=False)
    except Exception:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.warning("Parquet write failed — saved CSV to %s", csv_path)
        return csv_path

    logger.info("SHAP values saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# All-in-one explain
# ---------------------------------------------------------------------------


def explain(
    model: Any,
    df: pd.DataFrame,
    output_path: Union[str, Path] = "outputs/shap_values.parquet",
    background: Optional[pd.DataFrame] = None,
    n_background: int = 100,
) -> Dict[str, Any]:
    """Run full explainability pipeline and save artefacts.

    Returns
    -------
    dict with keys ``shap_values``, ``base_values``, ``feature_names``,
    ``driver_ranking``, ``local_explanations``, ``output_path``.
    """
    X = _prepare_X(df)
    cell_ids = df.get("cell_id", pd.Series(X.index, name="cell_id"))

    shap_values, base_values, feature_names = global_shap_values(
        model, X, background=background, n_background=n_background
    )

    ranking = rank_drivers(shap_values, feature_names)
    local_df = local_shap_values(model, X, cell_ids=cell_ids, background=background, n_background=n_background)
    saved_path = save_shap_values(shap_values, feature_names, cell_ids, output_path)

    logger.info("Explanation complete | top driver: %s", ranking.iloc[0]["feature"])

    return {
        "shap_values": shap_values,
        "base_values": base_values,
        "feature_names": feature_names,
        "driver_ranking": ranking,
        "local_explanations": local_df,
        "output_path": str(saved_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="SHAP explainability for valuation model")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--model", default="models/tabular_q50.pkl")
    parser.add_argument("--output", default="outputs/shap_values.parquet")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    with open(args.model, "rb") as fh:
        model = pickle.load(fh)

    result = explain(model, df, output_path=args.output)
    print(result["driver_ranking"].head(10).to_string())
