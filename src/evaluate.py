"""
evaluate.py
-----------
Model evaluation utilities.

Computes:
  MAE   — Mean Absolute Error (in Wh, same unit as target)
  RMSE  — Root Mean Squared Error (in Wh)
  R²    — Coefficient of Determination (fraction of variance explained)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, and R² for regression predictions.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        {'MAE': float, 'RMSE': float, 'R2': float}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def print_metrics_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Pretty-print model comparison results.

    Parameters
    ----------
    results : dict
        {model_name: {'MAE': ..., 'RMSE': ..., 'R2': ...}}
    """
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>10} {'R²':>8}")
    print("-" * 55)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} {metrics['MAE']:>8.2f} "
            f"{metrics['RMSE']:>10.2f} {metrics['R2']:>8.4f}"
        )
    print("=" * 55 + "\n")


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """
    Select the best model based on lowest RMSE.

    RMSE is chosen as the primary selection criterion because it penalises
    large errors more heavily than MAE, which is important for energy
    anomaly detection where large deviations are especially costly.

    Parameters
    ----------
    results : dict
        {model_name: {'MAE': ..., 'RMSE': ..., 'R2': ...}}

    Returns
    -------
    str
        Name of the best-performing model.
    """
    best = min(results, key=lambda m: results[m]["RMSE"])
    logger.info("Best model selected: %s (RMSE=%.4f)", best, results[best]["RMSE"])
    return best
