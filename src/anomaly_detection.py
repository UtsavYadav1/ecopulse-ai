"""
anomaly_detection.py
--------------------
Residual-based anomaly detection for energy consumption.

Methodology
-----------
After the model generates predictions on the test set, we compute:
    residual = actual_energy - predicted_energy

The anomaly threshold is derived from the training-set residual distribution
to avoid any look-ahead bias:
    threshold = mean(|residual_train|) + 2 * std(|residual_train|)

This is a statistically interpretable threshold: it flags observations
whose absolute prediction error exceeds the typical training error by
more than 2 standard deviations.

Severity Classification
-----------------------
    |residual| <= threshold                → Normal
    threshold < |residual| <= 1.5*threshold → Moderate
    |residual| > 1.5*threshold             → High

Important Disclaimer
--------------------
An anomaly flag means the observed consumption is unusually different
from the model's expectation. It does NOT prove equipment failure,
sensor malfunction, or any specific physical cause.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


def compute_threshold(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute the anomaly threshold from training-set residuals.

    Parameters
    ----------
    y_train_true : array-like
        Actual training target values.
    y_train_pred : array-like
        Predicted training target values.

    Returns
    -------
    Tuple[float, float, float]
        (threshold, mean_abs_residual, std_abs_residual)
    """
    abs_residuals = np.abs(np.array(y_train_true) - np.array(y_train_pred))
    mean_ar = float(np.mean(abs_residuals))
    std_ar = float(np.std(abs_residuals))
    threshold = mean_ar + 2 * std_ar
    logger.info(
        "Anomaly threshold: %.2f  (mean_abs_residual=%.2f, std=%.2f)",
        threshold,
        mean_ar,
        std_ar,
    )
    return threshold, mean_ar, std_ar


def classify_severity(abs_residual: float, threshold: float) -> str:
    """Assign severity label based on absolute residual vs threshold."""
    if abs_residual <= threshold:
        return "Normal"
    elif abs_residual <= 1.5 * threshold:
        return "Moderate"
    else:
        return "High"


def detect_anomalies(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    extra_features: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build a results DataFrame with anomaly flags for the test set.

    Parameters
    ----------
    dates : pd.Series
        Timestamps corresponding to y_true / y_pred.
    y_true : array-like
        Actual energy consumption (Wh).
    y_pred : array-like
        Model-predicted energy consumption (Wh).
    threshold : float
        Anomaly threshold (from compute_threshold).
    extra_features : pd.DataFrame, optional
        Additional columns to include (e.g., temperature, humidity).

    Returns
    -------
    pd.DataFrame
        One row per test observation with columns:
        date, actual_energy, predicted_energy, residual,
        absolute_error, is_anomaly, severity
        plus any extra feature columns if provided.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    results = pd.DataFrame(
        {
            "date": dates.values,
            "actual_energy": y_true,
            "predicted_energy": y_pred.round(2),
            "residual": residuals.round(2),
            "absolute_error": abs_errors.round(2),
            "is_anomaly": abs_errors > threshold,
            "severity": [classify_severity(ae, threshold) for ae in abs_errors],
        }
    )

    if extra_features is not None:
        for col in extra_features.columns:
            results[col] = extra_features[col].values

    anomaly_count = results["is_anomaly"].sum()
    logger.info(
        "Anomalies detected: %d / %d (%.1f%%)",
        anomaly_count,
        len(results),
        100 * anomaly_count / len(results),
    )
    return results


def get_anomaly_summary(results: pd.DataFrame) -> Dict:
    """Return a summary dict for display on the dashboard."""
    total = len(results)
    anomalies = results[results["is_anomaly"]]
    return {
        "total_predictions": total,
        "anomaly_count": int(anomalies.shape[0]),
        "anomaly_rate_pct": round(100 * anomalies.shape[0] / total, 2),
        "moderate_count": int((anomalies["severity"] == "Moderate").sum()),
        "high_count": int((anomalies["severity"] == "High").sum()),
    }
