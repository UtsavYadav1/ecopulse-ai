"""
app.py
------
EcoPulse AI — Flask Web Application

Routes:
    GET  /            → Redirect to dashboard
    GET  /dashboard   → Main dashboard with metrics, plots, anomalies
    POST /predict     → Accept feature input and return prediction
    POST /explain     → Get Gemini explanation for an anomaly
    GET  /api/metrics → JSON: model metrics and anomaly summary
    GET  /api/anomalies → JSON: recent anomaly records

Usage:
    python app.py

The app loads a pre-trained model from models/final_model.joblib.
Train the model first with:
    python src/train.py
"""

import os
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gemini_service import get_gemini_explanation
from src.feature_engineering import add_datetime_features, EXCLUDE_FROM_FEATURES
from src.anomaly_detection import classify_severity

# ---------------------------------------------------------------------------
# Load environment variables from .env if python-dotenv is available
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "final_model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "anomaly_threshold.json")
ANOMALY_CSV = os.path.join(MODELS_DIR, "anomaly_results.csv")

# ---------------------------------------------------------------------------
# Load pre-trained model and metadata at startup
# ---------------------------------------------------------------------------

def load_model_artifacts():
    """Load all model artifacts. Returns (model, feature_columns, metrics, threshold) or None."""
    try:
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        with open(THRESHOLD_PATH) as f:
            threshold_data = json.load(f)
        logger.info("Model artifacts loaded successfully.")
        return model, feature_cols, metrics, threshold_data
    except FileNotFoundError as e:
        logger.warning("Model artifacts not found: %s", e)
        return None, None, None, None


model, feature_columns, metrics_data, threshold_data = load_model_artifacts()


def is_trained():
    return model is not None


# ---------------------------------------------------------------------------
# Helper: build feature row for a single prediction request
# ---------------------------------------------------------------------------

DATASET_FEATURE_DEFAULTS = {
    "lights": 0.0,
    "T1": 19.89, "RH_1": 47.59,
    "T2": 19.2,  "RH_2": 44.79,
    "T3": 19.79, "RH_3": 44.73,
    "T4": 19.0,  "RH_4": 45.56,
    "T5": 17.17, "RH_5": 45.41,
    "T6": 7.03,  "RH_6": 84.25,
    "T7": 17.2,  "RH_7": 41.42,
    "T8": 18.2,  "RH_8": 48.9,
    "T9": 17.03, "RH_9": 45.52,
    "T_out": 6.60, "Press_mm_hg": 733.5,
    "RH_out": 92.0, "Windspeed": 7.0,
    "Visibility": 63.0, "Tdewpoint": 5.3,
    # datetime-derived defaults
    "hour": 12, "day": 15, "day_of_week": 0,
    "month": 1, "year": 2016, "is_weekend": 0,
}


def build_feature_row(form_data: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame from user-supplied form data.
    Missing fields are filled with dataset-average defaults.
    """
    row = DATASET_FEATURE_DEFAULTS.copy()
    for key, val in form_data.items():
        if key in row:
            try:
                row[key] = float(val)
            except (ValueError, TypeError):
                pass

    # If date_input provided, override datetime features
    date_input = form_data.get("date_input", "")
    if date_input:
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%dT%H:%M")
            row["hour"] = dt.hour
            row["day"] = dt.day
            row["day_of_week"] = dt.weekday()
            row["month"] = dt.month
            row["year"] = dt.year
            row["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        except ValueError:
            pass

    df_row = pd.DataFrame([row])
    return df_row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    trained = is_trained()
    anomalies_preview = []
    if trained and os.path.isfile(ANOMALY_CSV):
        adf = pd.read_csv(ANOMALY_CSV)
        adf_anomalies = adf[adf["is_anomaly"]].sort_values("absolute_error", ascending=False)
        anomalies_preview = adf_anomalies.head(10).to_dict(orient="records")
        # Format dates
        for row in anomalies_preview:
            if "date" in row:
                row["date"] = str(row["date"])[:16]

    return render_template(
        "index.html",
        trained=trained,
        metrics=metrics_data,
        threshold=threshold_data,
        anomalies=anomalies_preview,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not is_trained():
        return jsonify({"error": "Model not trained. Run: python src/train.py"}), 503

    form_data = request.form.to_dict()
    df_row = build_feature_row(form_data)

    # Align with model's expected features
    if feature_columns:
        for col in feature_columns:
            if col not in df_row.columns:
                df_row[col] = DATASET_FEATURE_DEFAULTS.get(col, 0.0)
        df_row = df_row[feature_columns]

    raw_pred = model.predict(df_row)[0]
    prediction = max(0.0, float(raw_pred))

    # Compute anomaly status based on threshold
    actual_str = form_data.get("actual_energy", "")
    anomaly_info = None
    if actual_str:
        try:
            actual = float(actual_str)
            residual = actual - prediction
            abs_err = abs(residual)
            t = threshold_data["threshold"] if threshold_data else 100.0
            severity = classify_severity(abs_err, t)
            is_anomaly = abs_err > t
            anomaly_info = {
                "actual_energy": round(actual, 2),
                "predicted_energy": round(prediction, 2),
                "residual": round(residual, 2),
                "absolute_error": round(abs_err, 2),
                "is_anomaly": is_anomaly,
                "severity": severity,
                "threshold": round(t, 2),
                # context for Gemini
                "hour": int(df_row.get("hour", [12])[0] if hasattr(df_row.get("hour"), "__getitem__") else df_row["hour"].values[0]),
                "day_of_week": int(df_row["day_of_week"].values[0]),
                "T1": float(df_row["T1"].values[0]) if "T1" in df_row.columns else "N/A",
                "T_out": float(df_row["T_out"].values[0]) if "T_out" in df_row.columns else "N/A",
                "RH_1": float(df_row["RH_1"].values[0]) if "RH_1" in df_row.columns else "N/A",
                "RH_out": float(df_row["RH_out"].values[0]) if "RH_out" in df_row.columns else "N/A",
                "Windspeed": float(df_row["Windspeed"].values[0]) if "Windspeed" in df_row.columns else "N/A",
                "Visibility": float(df_row["Visibility"].values[0]) if "Visibility" in df_row.columns else "N/A",
                "lights": float(df_row["lights"].values[0]) if "lights" in df_row.columns else "N/A",
                "Tdewpoint": float(df_row["Tdewpoint"].values[0]) if "Tdewpoint" in df_row.columns else "N/A",
            }
        except (ValueError, TypeError):
            pass

    return render_template(
        "result.html",
        prediction=round(prediction, 2),
        anomaly_info=anomaly_info,
        form_data=form_data,
    )


@app.route("/explain", methods=["POST"])
def explain():
    """Get Gemini explanation for a provided anomaly context."""
    data = request.json or {}
    explanation = get_gemini_explanation(data)
    return jsonify({"explanation": explanation})


@app.route("/api/metrics")
def api_metrics():
    if not is_trained():
        return jsonify({"error": "Model not trained"}), 503
    return jsonify(metrics_data)


@app.route("/api/anomalies")
def api_anomalies():
    if not os.path.isfile(ANOMALY_CSV):
        return jsonify({"error": "Anomaly results not found. Run training first."}), 404
    adf = pd.read_csv(ANOMALY_CSV)
    adf_anomalies = adf[adf["is_anomaly"]].head(50)
    return jsonify(adf_anomalies.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    logger.info("Starting EcoPulse AI on http://localhost:%d", port)
    app.run(debug=debug, host="0.0.0.0", port=port)
