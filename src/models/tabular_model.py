"""
Tabular model for real estate valuation prediction using LightGBM.

Trains a quantile-regression model to predict annualized_valuation
from urban features.  Falls back to sklearn GradientBoostingRegressor
when LightGBM is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES: Tuple[float, float, float] = (0.1, 0.5, 0.9)
TARGET_COL = "annualized_valuation"
EXCLUDE_COLS: Tuple[str, ...] = ("cell_id", "lat", "lon", TARGET_COL)

# Try to import LightGBM
try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:  # pragma: no cover
    _HAS_LGB = False
    logger.warning("lightgbm not installed — falling back to sklearn GradientBoostingRegressor")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TabularModelResult:
    """Container for a trained tabular model and its metadata."""

    models: Dict[float, object] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    cv_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    backend: str = "lightgbm"
    quantiles: Tuple[float, ...] = QUANTILES

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "feature_names": self.feature_names,
            "cv_metrics": self.cv_metrics,
            "feature_importance": self.feature_importance,
            "quantiles": list(self.quantiles),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return *(X, y)* with non-feature columns dropped.

    Drops explicit exclude columns (cell_id, lat, lon, target) plus any
    non-numeric columns (e.g. ``city``) that LightGBM cannot consume.
    """
    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # Drop non-numeric columns (object/category dtypes) — LightGBM requires
    # int/float/bool. Keep only numeric and boolean columns as features.
    numeric_types = ("number", "bool")
    X = X.select_dtypes(include=numeric_types)

    y = df[TARGET_COL] if TARGET_COL in df.columns else pd.Series(dtype=float)
    return X, y


def _sort_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the DataFrame by *lat* then *lon* to simulate temporal ordering for TimeSeriesSplit."""
    if "lat" in df.columns and "lon" in df.columns:
        return df.sort_values(["lat", "lon"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# LightGBM backend
# ---------------------------------------------------------------------------


def _train_lgb_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    quantile: float,
) -> "lgb.LGBMRegressor":
    params = dict(
        objective="quantile",
        alpha=quantile,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.early_stopping(50, verbose=False)] if X_val is not None and y_val is not None else []
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)] if X_val is not None and y_val is not None else None,
        callbacks=callbacks,
    )
    return model


# ---------------------------------------------------------------------------
# Sklearn fallback backend
# ---------------------------------------------------------------------------


def _train_sklearn_quantile(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantile: float,
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------


def _extract_feature_importance(models: Dict[float, object], feature_names: List[str]) -> Dict[str, float]:
    """Average feature importance across quantile models."""
    importances: Dict[str, float] = {n: 0.0 for n in feature_names}
    count = 0
    for q, model in models.items():
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            for i, name in enumerate(feature_names):
                importances[name] += float(fi[i])
            count += 1
    if count > 0:
        for k in importances:
            importances[k] /= count
    # normalise
    total = sum(importances.values()) or 1.0
    return {k: v / total for k, v in importances.items()}


# ---------------------------------------------------------------------------
# Public API — train
# ---------------------------------------------------------------------------


def train(
    df: pd.DataFrame,
    n_splits: int = 5,
    save_dir: Optional[Union[str, Path]] = None,
) -> TabularModelResult:
    """Train quantile-regression models on *df*.

    Parameters
    ----------
    df:
        DataFrame containing features and ``annualized_valuation``.
    n_splits:
        Number of TimeSeriesSplit folds for cross-validation.
    save_dir:
        Directory where model artefacts will be persisted.

    Returns
    -------
    TabularModelResult
    """
    logger.info("Starting tabular training | rows=%d | lgb_available=%s", len(df), _HAS_LGB)

    df = _sort_by_location(df)
    X, y = _split_features_target(df)
    feature_names = X.columns.tolist()

    if len(y) == 0:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame")

    result = TabularModelResult(feature_names=feature_names, backend="lightgbm" if _HAS_LGB else "sklearn")

    # Cross-validation -------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(df) // 10)))
    cv_r2_scores: List[float] = []
    cv_mae_scores: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        if _HAS_LGB:
            m = _train_lgb_quantile(X_tr, y_tr, X_va, y_va, 0.5)
        else:
            m = _train_sklearn_quantile(X_tr, y_tr, 0.5)
            result.backend = "sklearn"

        preds = m.predict(X_va)
        cv_r2_scores.append(float(r2_score(y_va, preds)))
        cv_mae_scores.append(float(mean_absolute_error(y_va, preds)))
        logger.info("Fold %d — R²=%.4f  MAE=%.4f", fold_idx, cv_r2_scores[-1], cv_mae_scores[-1])

    result.cv_metrics = {
        "cv_r2_mean": float(np.mean(cv_r2_scores)),
        "cv_r2_std": float(np.std(cv_r2_scores)),
        "cv_mae_mean": float(np.mean(cv_mae_scores)),
        "cv_mae_std": float(np.std(cv_mae_scores)),
    }
    logger.info("CV metrics: %s", result.cv_metrics)

    # Full-data training -----------------------------------------------------
    for q in QUANTILES:
        if _HAS_LGB:
            result.models[q] = _train_lgb_quantile(X, y, None, None, q)
        else:
            result.models[q] = _train_sklearn_quantile(X, y, q)
        logger.info("Trained quantile=%.1f", q)

    # Feature importance -----------------------------------------------------
    result.feature_importance = _extract_feature_importance(result.models, feature_names)

    # Persist ---------------------------------------------------------------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_artifacts(result, save_dir)

    return result


def _save_artifacts(result: TabularModelResult, save_dir: Path) -> None:
    """Persist model artefacts to *save_dir*."""
    backend_suffix = "pkl"
    for q, model in result.models.items():
        path = save_dir / f"tabular_q{int(q * 100):02d}.{backend_suffix}"
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
        logger.info("Saved %s", path)

    meta_path = save_dir / "tabular_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info("Saved %s", meta_path)


# ---------------------------------------------------------------------------
# Public API — predict
# ---------------------------------------------------------------------------


def predict(
    df: pd.DataFrame,
    result: Optional[TabularModelResult] = None,
    model_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate predictions with lower / median / upper quantiles.

    Parameters
    ----------
    df:
        Feature DataFrame.
    result:
        A previously-trained ``TabularModelResult``.  If *None*, models are
        loaded from *model_dir*.
    model_dir:
        Directory containing persisted artefacts.

    Returns
    -------
    pd.DataFrame with columns ``pred_p10``, ``pred_p50``, ``pred_p90``.
    """
    if result is None:
        if model_dir is None:
            raise ValueError("Either *result* or *model_dir* must be provided")
        result = _load_artifacts(Path(model_dir))

    X, _ = _split_features_target(df)
    # Align columns
    X = X[result.feature_names]

    out = pd.DataFrame(index=df.index)
    for q in result.quantiles:
        model = result.models[q]
        col = f"pred_p{int(q * 100):02d}"
        out[col] = model.predict(X)

    return out


def _load_artifacts(model_dir: Path) -> TabularModelResult:
    """Load persisted artefacts from *model_dir*."""
    meta_path = model_dir / "tabular_meta.json"
    with open(meta_path) as fh:
        meta = json.load(fh)

    result = TabularModelResult(
        feature_names=meta["feature_names"],
        cv_metrics=meta["cv_metrics"],
        feature_importance=meta["feature_importance"],
        backend=meta["backend"],
        quantiles=tuple(meta["quantiles"]),
    )
    for q in result.quantiles:
        path = model_dir / f"tabular_q{int(q * 100):02d}.pkl"
        with open(path, "rb") as fh:
            result.models[q] = pickle.load(fh)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train tabular valuation model")
    parser.add_argument("--features", default="data/processed/features.csv", help="Path to features CSV")
    parser.add_argument("--save-dir", default="models", help="Directory to save artefacts")
    parser.add_argument("--n-splits", type=int, default=5, help="CV splits")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    res = train(df, n_splits=args.n_splits, save_dir=args.save_dir)
    print(json.dumps(res.cv_metrics, indent=2))
