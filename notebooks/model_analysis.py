"""
Model analysis notebook (executed as a plain Python script).

Loads features and model artefacts, then:
  1. Displays feature importance.
  2. Plots predicted vs. actual.
  3. Shows SHAP summary.

Run:  python notebooks/model_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure src is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.models import tabular_model
from src.models.explain import global_shap_values, rank_drivers, _prepare_X

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES_CSV = "data/processed/features.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FIGURES_DIR = "outputs/figures"

TARGET_COL = "annualized_valuation"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    # --- Load data --------------------------------------------------------
    logger.info("Loading features …")
    df = pd.read_csv(FEATURES_CSV)
    logger.info("Rows=%d  Cols=%d", len(df), df.shape[1])

    # --- Load model -------------------------------------------------------
    logger.info("Loading tabular model from %s …", MODELS_DIR)
    result = tabular_model._load_artifacts(Path(MODELS_DIR))
    median_model = result.models[0.5]

    # --- Predict ----------------------------------------------------------
    preds = tabular_model.predict(df, result=result)
    y_true = df[TARGET_COL].values
    y_pred = preds["pred_p50"].values

    # ====================================================================
    # 1. Feature importance
    # ====================================================================
    fi = result.feature_importance
    fi_df = pd.DataFrame(
        {"feature": list(fi.keys()), "importance": list(fi.values())}
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi_df["feature"], fi_df["importance"], color="steelblue")
    ax.set_xlabel("Normalised Importance")
    ax.set_title("Feature Importance (LightGBM median quantile)")
    fig.tight_layout()
    fig_path = Path(FIGURES_DIR) / "feature_importance.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("Saved %s", fig_path)
    plt.close(fig)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (top 10)")
    print("=" * 60)
    print(fi_df.sort_values("importance", ascending=False).head(10).to_string(index=False))

    # ====================================================================
    # 2. Predicted vs. Actual
    # ====================================================================
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, c="steelblue", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
    ax.set_xlabel("Actual Annualized Valuation")
    ax.set_ylabel("Predicted (p50)")
    ax.set_title("Predicted vs Actual")
    ax.legend()
    fig.tight_layout()
    fig_path = Path(FIGURES_DIR) / "pred_vs_actual.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("Saved %s", fig_path)
    plt.close(fig)

    # Also plot prediction interval width
    if "pred_p10" in preds.columns and "pred_p90" in preds.columns:
        interval_width = preds["pred_p90"] - preds["pred_p10"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(interval_width, bins=50, color="steelblue", edgecolor="white")
        ax.set_xlabel("Prediction Interval Width (p90 − p10)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Prediction Interval Widths")
        fig.tight_layout()
        fig_path = Path(FIGURES_DIR) / "interval_width.png"
        fig.savefig(fig_path, dpi=150)
        logger.info("Saved %s", fig_path)
        plt.close(fig)

    # ====================================================================
    # 3. SHAP summary
    # ====================================================================
    logger.info("Computing SHAP values …")
    X = _prepare_X(df)
    shap_values, base_values, feature_names = global_shap_values(median_model, X)

    # SHAP summary (beeswarm-style scatter)
    fig, ax = plt.subplots(figsize=(10, 8))
    # For each feature, plot shap value vs feature value
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    for pos, idx in enumerate(order):
        ax.scatter(
            shap_values[:, idx],
            np.full(len(shap_values), pos),
            c=X.iloc[:, idx],
            cmap="coolwarm",
            s=8,
            alpha=0.5,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP Summary Plot")
    fig.tight_layout()
    fig_path = Path(FIGURES_DIR) / "shap_summary.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("Saved %s", fig_path)
    plt.close(fig)

    # Driver ranking
    ranking = rank_drivers(shap_values, feature_names)
    print("\n" + "=" * 60)
    print("SHAP DRIVER RANKING (top 10)")
    print("=" * 60)
    print(ranking.head(10).to_string(index=False))

    # Save ranking
    ranking_path = Path(OUTPUTS_DIR) / "driver_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    logger.info("Saved %s", ranking_path)

    print("\n✅ Analysis complete. Figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
