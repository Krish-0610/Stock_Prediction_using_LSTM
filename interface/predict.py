import os
import json
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from interface.predict_config import *

from features.technical_indicators import add_technical_indicators
from features.macro_features import get_macro_features
from features.flags import build_flag_dataframe
from utils.flatten import flatten_columns


def load_artifacts(index_name):
    model_dir = os.path.join(BASE_MODEL_DIR, index_name)

    model = load_model(os.path.join(model_dir, MODEL_FILENAME))
    target_scaler = joblib.load(os.path.join(model_dir, TARGET_SCALER_FILENAME))
    input_scaler = joblib.load(os.path.join(model_dir, INPUT_SCALER_FILENAME))

    with open(os.path.join(model_dir, METADATA_FILENAME), "r") as f:
        metadata = json.load(f)

    return model, input_scaler, target_scaler, metadata["features"]


def build_features(ticker, start, end):
    # ---- Raw price (date backbone) ----
    price_df = yf.download(ticker, start, end)
    assert not price_df.empty, "Price data empty"

    raw_index = price_df.index.copy()

    # ---- Technical indicators ----
    price_df = add_technical_indicators(price_df)
    price_df = flatten_columns(price_df)
    print(f"\nPrice DF length: {len(price_df)}\n\n")

    # ---- Macro features (overnight) ----
    macro_df = get_macro_features(raw_index.min(), raw_index.max())
    macro_df = macro_df.reindex(raw_index, method="ffill")
    macro_df = flatten_columns(macro_df)

    # ---- Binary flags (date-based, shifted) ----
    flag_df = build_flag_dataframe(raw_index)
    flag_df = flag_df.shift(1).fillna(0)
    flag_df.columns = flag_df.columns.astype(str)

    # ---- Merge ----
    df = pd.concat([price_df, macro_df, flag_df], axis=1)
    df = flatten_columns(df)
    print(df)
    df.dropna(inplace=True)
    df = df.rename(columns={f"Open_{ticker}": "Open", f"Close_{ticker}": "Close"})

    return df


def predict_next_day(index_name):
    # ---------- Validate index ----------
    if index_name not in INDEX_CONFIG:
        raise ValueError(f"Unknown index: {index_name}")

    ticker = INDEX_CONFIG[index_name]["ticker"]

    # ---------- Load artifacts ----------
    model, input_scaler, target_scaler, feature_order = load_artifacts(index_name)

    # ---------- Fetch & build features ----------
    start, end = get_fetch_window()
    df = build_features(ticker, start, end)

    if df.empty:
        raise RuntimeError("Feature dataframe is empty")

    if len(df) < LOOKBACK:
        raise RuntimeError(
            f"Not enough rows for prediction: have {len(df)}, need {LOOKBACK}"
        )

    # ---------- Enforce exact training feature order ----------
    missing = set(feature_order) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing features at inference: {missing}")
    last_close = df["Close"].iloc[-1]
    df = df[feature_order]

    # ---------- Scale inputs ----------
    X_raw = df.values  # shape: (N, n_features)
    X_scaled = input_scaler.transform(X_raw)  # numpy array

    # Take last LOOKBACK window
    X = X_scaled[-LOOKBACK:]
    X = X.reshape(1, LOOKBACK, X.shape[1])

    # ---------- Sanity (non-fatal) ----------
    if not np.isfinite(X).all():
        raise RuntimeError("NaN or Inf detected in model input")

    # ---------- Predict ----------
    y_scaled = model.predict(X, verbose=0)

    # ---------- Inverse scale targets ----------
    if RETURN_INVERSE_SCALE:
        y_log = target_scaler.inverse_transform(y_scaled)

        pred_close = last_close * np.exp(y_log[0, 0])
        pred_open = last_close * np.exp(y_log[0, 1])

    return {"Close": pred_close.item(), "Open": pred_open.item()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="e.g. nifty50, banknifty")
    args = parser.parse_args()

    result = predict_next_day(args.index)
    print(result)
