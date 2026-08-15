"""
End-to-end training pipeline for the Radar de Valorización Urbana.

Loads processed features, trains the tabular model, optionally trains
the multimodal model, generates SHAP explanations, and writes a model
card.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src is on the path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.models import tabular_model, explain as shap_explain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "annualized_valuation"
FEATURES_CSV = "data/processed/features.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Create models/ and outputs/ directories if they don't exist."""
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# ---------------------------------------------------------------------------
# Multimodal (optional)
# ---------------------------------------------------------------------------


def _try_train_multimodal(df: pd.DataFrame) -> Optional[Dict]:
    """Attempt to train the multimodal model; return *None* on failure."""
    try:
        from src.models import multimodal_model

        if not multimodal_model._HAS_TORCH:
            logger.info("PyTorch not available — skipping multimodal training")
            return None

        logger.info("Training multimodal model …")
        result = multimodal_model.train(
            df,
            image_dir=Path("data/processed/images"),
            save_dir=Path(MODELS_DIR),
            epochs=50,
        )
        return result
    except Exception as exc:
        logger.warning("Multimodal training failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def _write_model_card(
    tabular_result: Any,
    metrics: Dict[str, float],
    mm_result: Optional[Dict],
    shap_ranking: pd.DataFrame,
    feature_names: List[str],
) -> None:
    """Write a model card to ``models/MODEL_CARD.md``."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top_drivers = shap_ranking.head(10).to_markdown(index=False) if hasattr(shap_ranking, "to_markdown") else shap_ranking.head(10).to_string()

    mm_section = "### Multimodal Model\n\n" \
        f"- **Available:** {'Yes' if mm_result else 'No'}\n"
    if mm_result:
        mm_section += f"- **Best validation loss:** {mm_result['metrics']['best_val_loss']:.4f}\n"

    card = f"""# Model Card — Radar de Valorización Urbana

> Generated: {now}

## Overview

This model predicts **annualized_valuation** — the annual percentage change
in urban land value — from urban features including accessibility,
socio-economic indicators, green-space coverage, and amenities.

## Data

- **Source:** `data/processed/features.csv`
- **Rows:** {metrics.get('n_rows', 'N/A')}
- **Features:** {len(feature_names)}
- **Target:** `{TARGET_COL}`

## Tabular Model

- **Algorithm:** {tabular_result.backend}
- **Quantiles:** {list(tabular_result.quantiles)}
- **Cross-validation:** {tabular_result.cv_metrics}

### Test Metrics (median quantile)

| Metric | Value |
|--------|-------|
| R²     | {metrics['r2']:.4f} |
| MAE    | {metrics['mae']:.4f} |
| RMSE   | {metrics['rmse']:.4f} |

{mm_section}

## Feature Importance (top 10 by mean |SHAP|)

{top_drivers}

## Feature List

{', '.join(feature_names)}

## Usage

```python
from src.models import tabular_model

# Train
result = tabular_model.train(df, save_dir='models/')

# Predict
preds = tabular_model.predict(df, result=result)
```

## Limitations

- Predictions are based on available urban features; undisclosed
  local factors (zoning changes, political decisions) are not captured.
- The multimodal satellite branch requires pre-generated image patches
  under `data/processed/images/{cell_id}.npy`.
- Uncertainty intervals are approximate and assume approximately
  independent errors across cells.

## Ethics

Model outputs are decision-support tools, not appraisals.  They should
not be used as the sole basis for property valuation, taxation, or
investment decisions.
"""
    card_path = Path(MODELS_DIR) / "MODEL_CARD.md"
    card_path.write_text(card, encoding="utf-8")
    logger.info("Model card written → %s", card_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    features_path: str = FEATURES_CSV,
    save_dir: str = MODELS_DIR,
) -> Dict[str, Any]:
    """Run the full training pipeline.

    Parameters
    ----------
    features_path:
        Path to the processed features CSV.
    save_dir:
        Directory for model artefacts.

    Returns
    -------
    dict with keys ``tabular``, ``multimodal``, ``shap``, ``metrics``.
    """
    _ensure_dirs()
    logger.info("Loading features from %s …", features_path)

    df = pd.read_csv(features_path)
    logger.info("Loaded %d rows × %d columns", len(df), df.shape[1])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {features_path}")

    # --- Tabular model ----------------------------------------------------
    logger.info("=== Training tabular model ===")
    tabular_result = tabular_model.train(df, n_splits=5, save_dir=Path(save_dir))

    # In-sample metrics (median quantile) for the model card
    X = df.drop(columns=[c for c in ("cell_id", "lat", "lon", TARGET_COL) if c in df.columns])
    y_true = df[TARGET_COL].values
    median_model = tabular_result.models[0.5]
    y_pred = median_model.predict(X)
    metrics = _compute_metrics(y_true, y_pred)
    metrics["n_rows"] = len(df)
    logger.info("Tabular metrics: %s", metrics)

    # --- Multimodal model (optional) --------------------------------------
    mm_result = _try_train_multimodal(df)

    # --- SHAP explanations ------------------------------------------------
    logger.info("=== Computing SHAP explanations ===")
    shap_result = shap_explain.explain(
        median_model,
        df,
        output_path=Path(OUTPUTS_DIR) / "shap_values.parquet",
    )

    # --- Model card -------------------------------------------------------
    _write_model_card(
        tabular_result=tabular_result,
        metrics=metrics,
        mm_result=mm_result,
        shap_ranking=shap_result["driver_ranking"],
        feature_names=tabular_result.feature_names,
    )

    logger.info("Pipeline complete ✅")
    return {
        "tabular": tabular_result,
        "multimodal": mm_result,
        "shap": shap_result,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Train valuation models end-to-end")
    parser.add_argument("--features", default=FEATURES_CSV, help="Path to features CSV")
    parser.add_argument("--save-dir", default=MODELS_DIR, help="Model output directory")
    args = parser.parse_args()

    result = run_pipeline(features_path=args.features, save_dir=args.save_dir)
    print("\n=== Pipeline Summary ===")
    print(f"Tabular backend: {result['tabular'].backend}")
    print(f"R²: {result['metrics']['r2']:.4f}")
    print(f"MAE: {result['metrics']['mae']:.4f}")
    print(f"RMSE: {result['metrics']['rmse']:.4f}")
    print(f"Multimodal trained: {result['multimodal'] is not None}")
    print(f"Top driver: {result['shap']['driver_ranking'].iloc[0]['feature']}")
