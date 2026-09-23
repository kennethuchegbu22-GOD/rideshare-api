"""
train.py

Reproduces the preprocessing + modeling pipeline from the
boston_rideshare_forecasting notebook, and saves everything the API
needs to serve predictions:

  artifacts/model.joblib          - trained RandomForestRegressor
  artifacts/scaler.joblib         - fitted MinMaxScaler
  artifacts/encoder.joblib        - fitted OneHotEncoder
  artifacts/feature_config.json   - column lists / ordering needed at inference time

Run this once, wherever you have access to the Kaggle dataset
(e.g. Colab, or locally with the CSV already downloaded), then copy
the resulting `artifacts/` folder into this project before building
the Docker image.

Usage:
    python train.py --csv path/to/rideshare_kaggle.csv
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)

    # Price is the only column with NaNs; rows with missing price are dropped
    # rather than imputed, since fabricating target values would corrupt training.
    raw_df.dropna(subset=["price"], inplace=True)

    raw_df = raw_df.drop(columns=["id", "product_id"], errors="ignore")
    raw_df = raw_df.drop(columns=["timestamp", "datetime", "timezone"], errors="ignore")
    raw_df = raw_df.drop(columns=["visibility.1", "apparentTemperature"], errors="ignore")
    raw_df = raw_df.drop(
        columns=[
            "windGustTime", "temperatureHighTime", "temperatureLowTime",
            "apparentTemperatureHigh", "apparentTemperatureLow",
            "apparentTemperatureHighTime", "apparentTemperatureLowTime",
            "apparentTemperatureMin", "apparentTemperatureMinTime",
            "apparentTemperatureMax", "apparentTemperatureMaxTime",
            "sunriseTime", "sunsetTime", "uvIndexTime",
            "temperatureMinTime", "temperatureMaxTime",
        ],
        errors="ignore",
    )
    raw_df = raw_df.drop(columns=["short_summary", "icon", "long_summary"], errors="ignore")
    raw_df = raw_df.drop(columns=["latitude", "longitude"], errors="ignore")
    raw_df = raw_df.drop(columns=["windBearing", "moonPhase"], errors="ignore")

    # Columns the API must receive per request: everything left at this point,
    # minus price (target) and temperatureHigh/temperatureLow (dropped below,
    # never used for anything downstream).
    raw_input_fields = [
        c for c in raw_df.columns if c not in ("price", "temperatureHigh", "temperatureLow")
    ]

    # Snapshot numeric ranges NOW, before hour gets consumed into hour_sin/hour_cos
    categorical_cols_snapshot = raw_df[raw_input_fields].select_dtypes(include="object").columns.tolist()
    numeric_input_cols = [c for c in raw_input_fields if c not in categorical_cols_snapshot]
    numeric_ranges = {
        col: {"min": float(raw_df[col].min()), "max": float(raw_df[col].max())}
        for col in numeric_input_cols
    }

    # Feature engineering: daily temperature volatility
    raw_df["temp_range"] = raw_df["temperatureMax"] - raw_df["temperatureMin"]
    raw_df = raw_df.drop(
        columns=["temperatureMax", "temperatureMin", "temperatureHigh", "temperatureLow"],
        errors="ignore",
    )

    # Cyclical hour encoding
    raw_df["hour_sin"] = np.sin(2 * np.pi * raw_df["hour"] / 24.0)
    raw_df["hour_cos"] = np.cos(2 * np.pi * raw_df["hour"] / 24.0)
    raw_df = raw_df.drop(columns=["hour"], errors="ignore")

    # Put price last for convenience
    cols = list(raw_df.columns)
    cols.remove("price")
    cols.append("price")
    raw_df = raw_df[cols]

    return raw_df, raw_input_fields, numeric_ranges


def main(csv_path: str):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    raw_df, raw_input_fields, numeric_ranges = load_and_clean(csv_path)

    train_df, temp_df = train_test_split(raw_df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    input_cols = list(train_df.columns)[:-1]
    target_col = "price"

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()
    test_inputs = test_df[input_cols].copy()
    test_targets = test_df[target_col].copy()

    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes("object").columns.tolist()
    cols_to_be_scaled = [c for c in numeric_cols if c not in ["hour_sin", "hour_cos"]]

    scaler = MinMaxScaler().fit(train_inputs[cols_to_be_scaled])
    train_inputs[cols_to_be_scaled] = scaler.transform(train_inputs[cols_to_be_scaled])
    val_inputs[cols_to_be_scaled] = scaler.transform(val_inputs[cols_to_be_scaled])
    test_inputs[cols_to_be_scaled] = scaler.transform(test_inputs[cols_to_be_scaled])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train_inputs[categorical_cols]
    )
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

    final_cols = ["hour_sin", "hour_cos"] + cols_to_be_scaled + encoded_cols
    X_train = train_inputs[final_cols].copy()
    X_val = val_inputs[final_cols].copy()
    X_test = test_inputs[final_cols].copy()

    # Winning configuration from the notebook's hyperparameter search
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        n_estimators=200,
        max_depth=20,
        min_samples_split=45,
        min_samples_leaf=15,
    )
    model.fit(X_train, train_targets)

    def mape(targets, preds):
        return np.mean(np.abs((targets - preds) / targets)) * 100

    train_mape = mape(train_targets, model.predict(X_train))
    val_mape = mape(val_targets, model.predict(X_val))
    test_mape = mean_absolute_percentage_error(test_targets, model.predict(X_test)) * 100

    print(f"Train MAPE: {train_mape:.4f}")
    print(f"Val MAPE:   {val_mape:.4f}")
    print(f"Test MAPE:  {test_mape:.4f}")

    # --- Save artifacts ---
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(ARTIFACT_DIR, "encoder.joblib"))

    feature_config = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cols_to_be_scaled": cols_to_be_scaled,
        "encoded_cols": encoded_cols,
        "final_cols": final_cols,
        # every raw field the API must receive per request, in a stable order
        "raw_input_fields": raw_input_fields,
        "metrics": {
            "train_mape": train_mape,
            "val_mape": val_mape,
            "test_mape": test_mape,
        },
    }
    # category_values needs the *original* (pre-encoding) categorical inputs;
    # recompute from the raw cleaned df instead, since train_inputs was overwritten above
    raw_categorical = raw_df[categorical_cols]
    feature_config["category_values"] = {
        col: sorted(raw_categorical[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    }

    # Save the observed min/max range for every numeric field, so the API can
    # flag requests that fall far outside what the model was actually trained
    # on (tree-based models like this RandomForest don't extrapolate well
    # beyond their training range).
    feature_config["numeric_ranges"] = numeric_ranges

    with open(os.path.join(ARTIFACT_DIR, "feature_config.json"), "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="uber-and-lyft-dataset-boston-ma/rideshare_kaggle.csv",
        help="Path to the raw Kaggle CSV",
    )
    args = parser.parse_args()
    main(args.csv)
