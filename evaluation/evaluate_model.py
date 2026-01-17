import json
import os
from datetime import datetime
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model


import sys

# Add the parent directory (project_root) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.flatten import flatten_columns
from features.technical_indicators import add_technical_indicators
from features.macro_features import get_macro_features
from features.flags import build_flag_dataframe
from training.prepare_data import prepare_data
from training.config import (
    LOOKBACK,
    TARGET_COLS,
    INDEX_CONFIG,
    BASE_MODEL_DIR,
    MODEL_FILENAME,
    SCALER_FILENAME,
    METADATA_FILENAME,
    get_train_window,
)


RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# =========================
# EVALUATION
# =========================


def evaluate_index(index_name: str, ticker: str):
    print(f"\nEvaluating {index_name}")

    model_path = f"{BASE_MODEL_DIR}/{index_name}/{MODEL_FILENAME}"
    scaler_path = f"{BASE_MODEL_DIR}/{index_name}/{SCALER_FILENAME}"
    meta_path = f"{BASE_MODEL_DIR}/{index_name}/{METADATA_FILENAME}"

    # Load artifacts
    model = load_model(model_path)
    target_scaler = joblib.load(scaler_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    features = metadata["features"]

    # Fetch data (same window as training)
    start, end = get_train_window()
    price_df = yf.download(ticker, start=start, end=end)
    assert not price_df.empty, "Price data empty"

    # Feature engineering
    price_df = add_technical_indicators(price_df)
    price_df = flatten_columns(price_df)

    macro_df = get_macro_features(price_df.index.min(), price_df.index.max())
    macro_df = macro_df.reindex(price_df.index, method="ffill")
    macro_df = flatten_columns(macro_df)

    flag_df = build_flag_dataframe(price_df.index)
    flag_df = flag_df.shift(1).fillna(0)
    flag_df.columns = flag_df.columns.astype(str)

    df = price_df.join([macro_df, flag_df])
    df.dropna(inplace=True)

    df = df.rename(columns={f"Open_{ticker}": "Open", f"Close_{ticker}": "Close"})

    df = df[features]

    # Prepare sequences (same logic as training)
    X_train, y_train, X_test, y_test, _ = prepare_data(
        df, features=features, target_cols=TARGET_COLS, time_step=LOOKBACK
    )

    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y_test)

    # Metrics (per target)
    metrics = {}
    for i, col in enumerate(TARGET_COLS):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        metrics[col] = {"RMSE": round(rmse, 4), "R2": round(r2, 4)}

    result = {
        "index": index_name,
        "ticker": ticker,
        "evaluated_on": str(datetime.now()),
        "metrics": metrics,
    }

    return result


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    for index, cfg in INDEX_CONFIG.items():
        result = evaluate_index(index, cfg["ticker"])

        output_path = f"{RESULTS_DIR}/{index}_eval.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved evaluation → {output_path}")
