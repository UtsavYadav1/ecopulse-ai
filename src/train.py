"""
train.py
--------
Main training script for EcoPulse AI.

Run with:
    python src/train.py

This script:
1. Downloads/loads the dataset
2. Cleans and preprocesses the data
3. Engineers time-based features
4. Trains two models: Linear Regression and Random Forest Regressor
5. Evaluates both on the held-out test set
6. Selects the best model based on RMSE
7. Saves the final model and metadata using joblib
8. Generates and saves evaluation plots

Output files:
    models/final_model.joblib      — trained model
    models/feature_columns.joblib  — list of feature column names
    models/metrics.json            — model comparison metrics
    models/anomaly_threshold.json  — threshold parameters
    static/plots/                  — saved charts
"""

import os
import sys
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_data
from src.preprocessing import clean_data, chronological_split
from src.feature_engineering import prepare_features
from src.evaluate import compute_metrics, print_metrics_table, select_best_model
from src.anomaly_detection import compute_threshold, detect_anomalies, get_anomaly_summary

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "final_model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "anomaly_threshold.json")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted(dates, y_true, y_pred, model_name, save_path):
    """Line chart: actual vs predicted energy over time (test set sample)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    # Show up to 2000 points for clarity
    n = min(2000, len(dates))
    ax.plot(pd.to_datetime(dates[:n]), y_true[:n], label="Actual", alpha=0.8, linewidth=1)
    ax.plot(pd.to_datetime(dates[:n]), y_pred[:n], label="Predicted", alpha=0.8, linewidth=1, linestyle="--")
    ax.set_title(f"Actual vs Predicted Energy Consumption ({model_name})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance Energy (Wh)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Plot saved: %s", save_path)


def plot_scatter(y_true, y_pred, model_name, save_path):
    """Scatter plot: actual vs predicted."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="#2196F3")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted Scatter ({model_name})", fontsize=13)
    ax.set_xlabel("Actual Energy (Wh)")
    ax.set_ylabel("Predicted Energy (Wh)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Plot saved: %s", save_path)


def plot_anomalies(results_df, threshold, save_path):
    """Plot energy consumption with anomalies highlighted."""
    fig, ax = plt.subplots(figsize=(14, 5))
    normal = results_df[~results_df["is_anomaly"]]
    anomalies = results_df[results_df["is_anomaly"]]

    ax.plot(pd.to_datetime(results_df["date"]), results_df["actual_energy"],
            color="#90CAF9", linewidth=0.8, alpha=0.9, label="Actual Energy")
    ax.scatter(pd.to_datetime(normal["date"]), normal["actual_energy"],
               s=4, color="#1565C0", alpha=0.4)
    ax.scatter(pd.to_datetime(anomalies["date"]), anomalies["actual_energy"],
               s=25, color="#E53935", zorder=5, label=f"Anomaly ({len(anomalies)})")

    ax.set_title("Energy Consumption with Anomalies Highlighted", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance Energy (Wh)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Plot saved: %s", save_path)


def plot_model_comparison(comparison_results, save_path):
    """Bar chart comparing model MAE and RMSE."""
    models = list(comparison_results.keys())
    mae_vals = [comparison_results[m]["MAE"] for m in models]
    rmse_vals = [comparison_results[m]["RMSE"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#42A5F5")
    bars2 = ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#66BB6A")

    ax.set_title("Model Comparison: MAE and RMSE", fontsize=13)
    ax.set_ylabel("Error (Wh)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Plot saved: %s", save_path)


def plot_energy_trend(train_df, test_df, save_path):
    """Full energy consumption trend (train + test)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pd.to_datetime(train_df["date"]), train_df["Appliances"],
            color="#42A5F5", linewidth=0.6, alpha=0.8, label="Training Data")
    ax.plot(pd.to_datetime(test_df["date"]), test_df["Appliances"],
            color="#EF5350", linewidth=0.6, alpha=0.8, label="Test Data")
    ax.axvline(pd.to_datetime(test_df["date"].iloc[0]), color="orange",
               linestyle="--", linewidth=1.5, label="Train/Test Split")
    ax.set_title("Appliance Energy Consumption Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance Energy (Wh)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Plot saved: %s", save_path)


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("EcoPulse AI — Training Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and preprocess data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading dataset…")
    raw_df = load_raw_data()

    logger.info("Step 2: Cleaning data…")
    cleaned_df = clean_data(raw_df)

    logger.info("Step 3: Chronological train/test split…")
    train_df, test_df = chronological_split(cleaned_df, train_ratio=0.80)

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("Step 4: Feature engineering…")
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    feature_columns = list(X_train.columns)

    # ------------------------------------------------------------------
    # 3. Define models
    # ------------------------------------------------------------------
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # ------------------------------------------------------------------
    # 4. Train and evaluate both models
    # ------------------------------------------------------------------
    logger.info("Step 5: Training models…")
    comparison_results = {}
    trained_models = {}
    test_predictions = {}

    for model_name, model in models.items():
        logger.info("  Training: %s", model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # energy cannot be negative
        metrics = compute_metrics(y_test.values, y_pred)
        comparison_results[model_name] = metrics
        trained_models[model_name] = model
        test_predictions[model_name] = y_pred
        logger.info("  %s → MAE=%.2f, RMSE=%.2f, R²=%.4f",
                    model_name, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    # ------------------------------------------------------------------
    # 5. Print comparison and select best model
    # ------------------------------------------------------------------
    print_metrics_table(comparison_results)
    best_model_name = select_best_model(comparison_results)
    best_model = trained_models[best_model_name]
    best_y_pred = test_predictions[best_model_name]

    logger.info("Selected final model: %s", best_model_name)

    # ------------------------------------------------------------------
    # 6. Anomaly detection on test set
    # ------------------------------------------------------------------
    logger.info("Step 6: Computing anomaly threshold…")
    y_train_pred = best_model.predict(X_train)
    y_train_pred = np.maximum(y_train_pred, 0)
    threshold, mean_ar, std_ar = compute_threshold(y_train.values, y_train_pred)

    logger.info("Step 7: Detecting anomalies on test set…")
    # Include key environmental features in the anomaly report
    env_cols = [c for c in ["T1", "T_out", "RH_1", "RH_out", "Windspeed",
                             "Visibility", "Tdewpoint", "lights"]
                if c in X_test.columns]
    results_df = detect_anomalies(
        dates=test_df["date"],
        y_true=y_test.values,
        y_pred=best_y_pred,
        threshold=threshold,
        extra_features=X_test[env_cols] if env_cols else None,
    )

    # Add hour and day_of_week for Gemini context
    results_df["hour"] = pd.to_datetime(results_df["date"]).dt.hour
    results_df["day_of_week"] = pd.to_datetime(results_df["date"]).dt.dayofweek

    anomaly_summary = get_anomaly_summary(results_df)
    logger.info("Anomaly summary: %s", anomaly_summary)

    # ------------------------------------------------------------------
    # 7. Save model and metadata
    # ------------------------------------------------------------------
    logger.info("Step 8: Saving model and metadata…")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)

    metrics_output = {
        "model_comparison": comparison_results,
        "best_model": best_model_name,
        "best_model_metrics": comparison_results[best_model_name],
        "anomaly_summary": anomaly_summary,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)

    threshold_output = {
        "threshold": threshold,
        "mean_abs_residual": mean_ar,
        "std_abs_residual": std_ar,
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(threshold_output, f, indent=2)

    # Save anomaly results CSV (sample for dashboard)
    results_csv = os.path.join(MODELS_DIR, "anomaly_results.csv")
    results_df.to_csv(results_csv, index=False)

    logger.info("Model saved: %s", MODEL_PATH)
    logger.info("Metrics saved: %s", METRICS_PATH)

    # ------------------------------------------------------------------
    # 8. Generate and save plots
    # ------------------------------------------------------------------
    logger.info("Step 9: Generating plots…")

    plot_actual_vs_predicted(
        results_df["date"], y_test.values, best_y_pred, best_model_name,
        os.path.join(PLOTS_DIR, "actual_vs_predicted_line.png"),
    )
    plot_scatter(
        y_test.values, best_y_pred, best_model_name,
        os.path.join(PLOTS_DIR, "actual_vs_predicted_scatter.png"),
    )
    plot_anomalies(
        results_df, threshold,
        os.path.join(PLOTS_DIR, "anomalies.png"),
    )
    plot_model_comparison(
        comparison_results,
        os.path.join(PLOTS_DIR, "model_comparison.png"),
    )
    plot_energy_trend(
        train_df, test_df,
        os.path.join(PLOTS_DIR, "energy_trend.png"),
    )

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Final model: %s", best_model_name)
    logger.info(
        "Metrics → MAE: %.2f | RMSE: %.2f | R²: %.4f",
        comparison_results[best_model_name]["MAE"],
        comparison_results[best_model_name]["RMSE"],
        comparison_results[best_model_name]["R2"],
    )
    logger.info("Anomaly threshold: %.2f Wh", threshold)
    logger.info("Anomalies detected: %d / %d", anomaly_summary["anomaly_count"],
                anomaly_summary["total_predictions"])
    logger.info("=" * 60)
    print("\n[OK] Training complete. You can now start the Flask app:")
    print("   python app.py\n")


if __name__ == "__main__":
    main()
