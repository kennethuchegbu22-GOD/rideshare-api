"""
app.py

Flask API for the Boston rideshare price model.

Endpoints:
  GET  /health   - liveness check + artifact metadata
  GET  /schema   - describes exactly what /predict expects
  POST /predict  - predict price for one or more rides

Expects artifacts produced by train.py to already exist under ./artifacts/:
  model.joblib, scaler.joblib, encoder.joblib, feature_config.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

app = Flask(__name__)

# --- Load artifacts once at startup, not per-request ---
try:
    model = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    encoder = joblib.load(os.path.join(ARTIFACT_DIR, "encoder.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "feature_config.json")) as f:
        feature_config = json.load(f)
    ARTIFACTS_LOADED = True
    LOAD_ERROR = None
except FileNotFoundError as e:
    # App still starts (so /health can report the problem clearly) but
    # /predict will return a 503 until artifacts are actually present.
    model = scaler = encoder = feature_config = None
    ARTIFACTS_LOADED = False
    LOAD_ERROR = str(e)

NUMERIC_COLS = feature_config["numeric_cols"] if feature_config else []
CATEGORICAL_COLS = feature_config["categorical_cols"] if feature_config else []
COLS_TO_BE_SCALED = feature_config["cols_to_be_scaled"] if feature_config else []
ENCODED_COLS = feature_config["encoded_cols"] if feature_config else []
FINAL_COLS = feature_config["final_cols"] if feature_config else []
RAW_INPUT_FIELDS = feature_config["raw_input_fields"] if feature_config else []
NUMERIC_RANGES = feature_config.get("numeric_ranges", {}) if feature_config else {}


def check_out_of_range(records: list[dict]) -> list[dict]:
    """
    For each record, returns a list of {field, value, trained_min, trained_max}
    dicts for any numeric field whose value falls outside the range the model
    was actually trained on. Tree-based models like this RandomForest don't
    extrapolate reliably beyond their training range, so predictions built on
    out-of-range inputs should be treated as unreliable, not just "a guess."
    """
    warnings = []
    for i, record in enumerate(records):
        for field, bounds in NUMERIC_RANGES.items():
            if field not in record:
                continue
            try:
                value = float(record[field])
            except (TypeError, ValueError):
                continue
            if value < bounds["min"] or value > bounds["max"]:
                warnings.append(
                    {
                        "record_index": i,
                        "field": field,
                        "value": value,
                        "trained_min": bounds["min"],
                        "trained_max": bounds["max"],
                    }
                )
    return warnings


def preprocess(records: list[dict]) -> pd.DataFrame:
    """
    Replicates the notebook's preprocessing pipeline on raw input rows.
    `records` is a list of dicts, each with the raw fields the model
    was trained on (see RAW_INPUT_FIELDS / GET /schema).
    """
    df = pd.DataFrame.from_records(records)

    missing = [c for c in RAW_INPUT_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = df[RAW_INPUT_FIELDS].copy()

    # Feature engineering — must mirror train.py exactly
    df["temp_range"] = df["temperatureMax"] - df["temperatureMin"]
    df = df.drop(columns=["temperatureMax", "temperatureMin"], errors="ignore")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"].astype(float) / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"].astype(float) / 24.0)
    df = df.drop(columns=["hour"], errors="ignore")

    # Scale numeric columns with the fitted scaler
    df[COLS_TO_BE_SCALED] = scaler.transform(df[COLS_TO_BE_SCALED])

    # One-hot encode categoricals with the fitted encoder
    # (handle_unknown='ignore' means a never-seen category becomes all-zero)
    encoded = encoder.transform(df[CATEGORICAL_COLS])
    df[ENCODED_COLS] = encoded

    return df[FINAL_COLS]


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok" if ARTIFACTS_LOADED else "artifacts_missing",
            "artifacts_loaded": ARTIFACTS_LOADED,
            "error": LOAD_ERROR,
            "metrics": feature_config.get("metrics") if feature_config else None,
        }
    ), (200 if ARTIFACTS_LOADED else 503)


@app.route("/schema", methods=["GET"])
def schema():
    if not ARTIFACTS_LOADED:
        return jsonify({"error": "Artifacts not loaded. Run train.py first."}), 503
    return jsonify(
        {
            "required_fields": RAW_INPUT_FIELDS,
            "categorical_fields": {
                col: feature_config["category_values"].get(col, [])
                for col in CATEGORICAL_COLS
            },
            "example": _example_payload(),
        }
    )


def _example_payload():
    example = {
        "day": 15,
        "month": 12,
        "hour": 18,
        "distance": 2.5,
        "surge_multiplier": 1.0,
        "temperature": 35.0,
        "temperatureMax": 40.0,
        "temperatureMin": 30.0,
        "precipIntensity": 0.0,
        "precipProbability": 0.1,
        "humidity": 0.6,
        "windSpeed": 5.0,
        "windGust": 8.0,
        "visibility": 10.0,
        "dewPoint": 28.0,
        "pressure": 1015.0,
        "cloudCover": 0.5,
        "uvIndex": 0,
        "ozone": 300.0,
        "precipIntensityMax": 0.0,
    }
    if feature_config:
        for col in CATEGORICAL_COLS:
            values = feature_config["category_values"].get(col, [])
            if values:
                example[col] = values[0]
    return example


@app.route("/predict", methods=["POST"])
def predict():
    if not ARTIFACTS_LOADED:
        return jsonify({"error": "Artifacts not loaded. Run train.py first.", "detail": LOAD_ERROR}), 503

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON."}), 400

    # Accept either a single ride (dict) or a batch (list of dicts)
    is_batch = isinstance(payload, list)
    records = payload if is_batch else [payload]

    if not records:
        return jsonify({"error": "Empty request."}), 400

    try:
        X = preprocess(records)
        preds = model.predict(X)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001 - surface unexpected errors as 400s, not 500s
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    preds = [round(float(p), 2) for p in preds]
    range_warnings = check_out_of_range(records)

    if is_batch:
        response = {"predictions": preds}
        if range_warnings:
            response["warnings"] = range_warnings
            response["warning_note"] = (
                "One or more inputs fall outside the range of data the model "
                "was trained on. Predictions for those records may be unreliable "
                "— tree-based models don't extrapolate well beyond training data."
            )
        return jsonify(response)

    response = {"predicted_price": preds[0]}
    if range_warnings:
        response["warnings"] = [
            {k: v for k, v in w.items() if k != "record_index"} for w in range_warnings
        ]
        response["warning_note"] = (
            "One or more inputs fall outside the range of data the model was "
            "trained on. This prediction may be unreliable — tree-based models "
            "don't extrapolate well beyond training data."
        )
    return jsonify(response)


if __name__ == "__main__":
    # Dev server only — the Dockerfile uses gunicorn for production.
    app.run(host="0.0.0.0", port=5000, debug=False)
