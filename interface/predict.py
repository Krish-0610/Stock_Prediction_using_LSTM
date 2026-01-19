import os
import sys
import json
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime, timedelta
from predict_config import (
    INDEX_CONFIG,
    LOOKBACK,
    TARGET_COLS,
    BASE_MODEL_DIR,
    MODEL_FILENAME,
    SCALER_FILENAME,
    METADATA_FILENAME,
    RETURN_INVERSE_SCALE,
    get_fetch_window,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from features.technical_indicators import add_technical_indicators
from features.macro_features import get_macro_features
from features.flags import build_flag_dataframe
from utils.flatten import flatten_columns


def load_artifacts(index_name):
    model_dir = os.path.join(BASE_MODEL_DIR, index_name)

    model = load_model(os.path.join(model_dir, MODEL_FILENAME))
    target_scaler = joblib.load(os.path.join(model_dir, SCALER_FILENAME))

    with open(os.path.join(model_dir, METADATA_FILENAME), "r") as f:
        metadata = json.load(f)

    return model, target_scaler, metadata["features"]


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
    if index_name not in INDEX_CONFIG:
        raise ValueError(f"Unknown index: {index_name}")

    ticker = INDEX_CONFIG[index_name]["ticker"]

    # Load model + metadata
    model, target_scaler, feature_order = load_artifacts(index_name)

    print(f"\n\n {index_name}, {target_scaler.data_min_}, {target_scaler.data_max_}")


    # Fetch data
    start, end = get_fetch_window()
    df = build_features(ticker, start, end)
    print(df)
    if len(df) < LOOKBACK:
        print(len(df))
        raise RuntimeError("Not enough data for lookback window")

    # Enforce training feature order
    df = df[feature_order]

    # Build model input
    X = df.iloc[-LOOKBACK:].values
    X = X.reshape(1, LOOKBACK, X.shape[1])

    # Predict
    y_pred = model.predict(X)

    if RETURN_INVERSE_SCALE:
        y_pred = target_scaler.inverse_transform(y_pred)

    return dict(zip(TARGET_COLS, y_pred.flatten().tolist()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="e.g. nifty50, banknifty")
    args = parser.parse_args()

    result = predict_next_day(args.index)
    print(result)