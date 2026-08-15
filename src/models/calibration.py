"""
Uncertainty calibration for valuation predictions.

Provides:
  * MC Dropout for the PyTorch multimodal model.
  * Quantile-regression calibration for the tabular model.
  * Unified :func:`calibrate_predictions` entry point.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conditional torch import (mirrors multimodal_model)
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# MC Dropout
# ---------------------------------------------------------------------------


def _enable_dropout(model: "torch.nn.Module") -> None:
    """Enable dropout layers at inference time for MC Dropout."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def mc_dropout_predict(
    model: "torch.nn.Module",
    images: "torch.Tensor",
    tabular: "torch.Tensor",
    n_samples: int = 100,
) -> np.ndarray:
    """Run *n_samples* stochastic forward passes.

    Returns
    -------
    np.ndarray of shape ``(n_samples, n_rows)``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for MC Dropout")

    model.eval()
    _enable_dropout(model)
    device = next(model.parameters()).device
    images = images.to(device)
    tabular = tabular.to(device)

    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds = model(images, tabular)
            all_preds.append(preds.cpu().numpy())
    return np.array(all_preds)  # (n_samples, n_rows)


# ---------------------------------------------------------------------------
# Quantile calibration (conformity-score based)
# ---------------------------------------------------------------------------


def quantile_calibrate(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_median: np.ndarray,
    q_upper: np.ndarray,
    alpha: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Calibrate quantile predictions via conformity scores.

    Uses a split-conformal approach: the non-conformity score is
    ``max(q_lower - y, y - q_upper)``.  The empirical *alpha*-quantile
    of these scores is used to widen / narrow intervals.

    Returns
    -------
    dict with keys ``lower``, ``median``, ``upper``.
    """
    scores = np.maximum(q_lower - y_true, y_true - q_upper)
    threshold = np.quantile(scores, 1.0 - alpha)
    return {
        "lower": q_lower - threshold,
        "median": q_median,
        "upper": q_upper + threshold,
    }


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


def calibrate_predictions(
    predictions: Union[pd.DataFrame, np.ndarray, Dict],
    n_samples: int = 100,
    y_true: Optional[np.ndarray] = None,
    alpha: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Calibrate prediction intervals.

    Accepts either:
      * A dict from ``mc_dropout_predict`` (raw MC samples) — returns
        calibrated mean and intervals from the MC distribution.
      * A DataFrame with ``pred_p10``, ``pred_p50``, ``pred_p90``
        columns — applies conformal quantile calibration.
      * A 2-D array of shape ``(n_samples, n_rows)`` — treats each row
        as an MC sample.

    Parameters
    ----------
    predictions:
        Raw predictions to calibrate.
    n_samples:
        Number of MC samples (used only for the MC path).
    y_true:
        Ground-truth values for conformal calibration.  If *None* the
        intervals are returned un-shifted.
    alpha:
        Mis-coverage rate (0.1 → 90 % intervals).

    Returns
    -------
    dict with keys ``mean``, ``lower``, ``upper``, ``std``.
    """
    # Case 1: MC samples array ------------------------------------------------
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
        return _calibrate_mc_array(predictions, alpha)

    # Case 2: dict from mc_dropout_predict -----------------------------------
    if isinstance(predictions, dict) and "mc_samples" in predictions:
        return _calibrate_mc_array(predictions["mc_samples"], alpha)

    # Case 3: DataFrame with quantile columns --------------------------------
    if isinstance(predictions, pd.DataFrame):
        if all(c in predictions.columns for c in ("pred_p10", "pred_p50", "pred_p90")):
            return _calibrate_quantile_df(predictions, y_true, alpha)
        if "pred_mm" in predictions.columns:
            # single-point multimodal predictions — no MC available
            logger.warning("Single-point predictions; returning point estimate with zero-width interval")
            arr = predictions["pred_mm"].values
            return {"mean": arr, "lower": arr, "upper": arr, "std": np.zeros_like(arr)}

    raise ValueError(f"Unsupported predictions type: {type(predictions)}")


def _calibrate_mc_array(mc_samples: np.ndarray, alpha: float) -> Dict[str, np.ndarray]:
    """Calibrate a 2-D array of MC samples."""
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    return {
        "mean": mc_samples.mean(axis=0),
        "lower": np.quantile(mc_samples, lower_q, axis=0),
        "upper": np.quantile(mc_samples, upper_q, axis=0),
        "std": mc_samples.std(axis=0),
    }


def _calibrate_quantile_df(
    df: pd.DataFrame,
    y_true: Optional[np.ndarray],
    alpha: float,
) -> Dict[str, np.ndarray]:
    """Conformal calibration on quantile predictions."""
    q_lo = df["pred_p10"].values
    q_md = df["pred_p50"].values
    q_hi = df["pred_p90"].values

    if y_true is not None:
        calibrated = quantile_calibrate(y_true, q_lo, q_md, q_hi, alpha=alpha)
        return {
            "mean": calibrated["median"],
            "lower": calibrated["lower"],
            "upper": calibrated["upper"],
            "std": (calibrated["upper"] - calibrated["lower"]) / 4.0,  # rough std estimate
        }

    return {
        "mean": q_md,
        "lower": q_lo,
        "upper": q_hi,
        "std": (q_hi - q_lo) / 4.0,
    }


# ---------------------------------------------------------------------------
# Prediction interval helper
# ---------------------------------------------------------------------------


def prediction_intervals(
    calibrated: Dict[str, np.ndarray],
    confidence: float = 0.9,
) -> pd.DataFrame:
    """Build a human-readable DataFrame of prediction intervals.

    Parameters
    ----------
    calibrated:
        Output of :func:`calibrate_predictions`.
    confidence:
        Confidence level for labelling (e.g. 0.9 → "90% PI").

    Returns
    -------
    pd.DataFrame with columns ``mean``, ``lower``, ``upper``, ``width``.
    """
    lower = calibrated["lower"]
    upper = calibrated["upper"]
    return pd.DataFrame(
        {
            f"mean": calibrated["mean"],
            f"lower_{int(confidence * 100)}": lower,
            f"upper_{int(confidence * 100)}": upper,
            f"width_{int(confidence * 100)}": upper - lower,
        }
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Quick smoke test with synthetic data
    rng = np.random.RandomState(42)
    mc = rng.randn(100, 500) * 0.03 + 0.05
    result = calibrate_predictions(mc, n_samples=100, alpha=0.1)
    print({k: v.shape for k, v in result.items()})
    print("Mean:", result["mean"][:5])
    print("Interval width (mean):", np.mean(result["upper"] - result["lower"]))
