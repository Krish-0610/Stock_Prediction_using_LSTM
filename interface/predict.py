import json
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model

from training.utils import flatten_columns
from features.technical_indicators import add_technical_indicators
from features.macro_features import get_macro_features
from features.flags import build_flag_dataframe

from interface.predict_config import (
    LOOKBACK,
    BASE_MODEL_DIR,
    MODEL_FILENAME,
    SCALER_FILENAME,
    METADATA_FILENAME,
    INDEX_CONFIG,
    FETCH_DAYS,
    INTERVAL
)

# =========================
# DATA PREPARATION
# =========================

def fetch_latest_data(ticker: str, lookback_days: int):
    df = yf.download(
        ticker,
        period=f"{lookback_days}d",
        interval=INTERVAL
    )
    assert len(df) >= LOOKBACK, "Not enough data for prediction"
    return df


def build_feature_dataframe(price_df: pd.DataFrame):
    price_df = add_technical_indicators(price_df)
    price_df = flatten_columns(price_df)

    macro_df = get_macro_features(
        price_df.index.min(),
        price_df.index.max()
    )
    macro_df = macro_df.reindex(price_df.index, method="ffill")
    macro_df = flatten_columns(macro_df)

    flag_df = build_flag_dataframe(price_df.index)
    flag_df = flag_df.shift(1).fillna(0)
    flag_df.columns = flag_df.columns.astype(str)

    df = price_df.join([macro_df, flag_df])
    df.dropna(inplace=True)

    return df


# =========================
# PREDICTION
# =========================

def predict_next(index_name: str):
    cfg = INDEX_CONFIG[index_name]
    model_path = f"{BASE_MODEL_DIR}/{index_name}/{MODEL_FILENAME}"
    scaler_path = f"{BASE_MODEL_DIR}/{index_name}/{SCALER_FILENAME}"
    meta_path = f"{BASE_MODEL_DIR}/{index_name}/{METADATA_FILENAME}"

    # Load artifacts
    model = load_model(model_path)
    target_scaler = joblib.load(scaler_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    features = metadata["features"]
    ticker = metadata["ticker"]

    # Fetch & prepare data
    price_df = fetch_latest_data(ticker, FETCH_DAYS)
    df = build_feature_dataframe(price_df)

    df = df.rename(columns={
        f"Open_{ticker}": "Open",
        f"Close_{ticker}": "Close"
    })

    df = df[features]

    latest_window = df.iloc[-LOOKBACK:].values
    X = np.expand_dims(latest_window, axis=0)

    # Predict
    y_scaled = model.predict(X)
    y_pred = target_scaler.inverse_transform(y_scaled)

    return {
        "index": index_name,
        "prediction": {
            "Close": float(y_pred[0][0]),
            "Open": float(y_pred[0][1])
        }
    }


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    for index in INDEX_CONFIG.keys():
        result = predict_next(index)
        print(result)
