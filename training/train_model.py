import os
import json
import joblib
import yfinance as yf
from datetime import datetime

from utils.flatten import flatten_columns
from features.technical_indicators import add_technical_indicators
from features.macro_features import get_macro_features
from features.flags import build_flag_dataframe
from training.prepare_data import prepare_data
from training.model import build_model_v1

from training.config import (
    LOOKBACK,
    EPOCHS,
    BATCH_SIZE,
    TARGET_COLS,
    BASE_MODEL_DIR,
    MODEL_FILENAME,
    SCALER_FILENAME,
    METADATA_FILENAME,
    INDEX_CONFIG,
    get_train_window
)


def train_index(index_name: str, ticker: str):
    print(f"\nTraining {index_name} ({ticker})")

    os.makedirs(f"{BASE_MODEL_DIR}/{index_name}", exist_ok=True)

    # 1. Fetch price data
    start , end = get_train_window()
    price_df = yf.download(ticker, start=start, end=end)
    assert not price_df.empty, "Price data empty"

    # 2. Technical indicators
    price_df = add_technical_indicators(price_df)
    price_df = flatten_columns(price_df)

    # 3. Macro features
    macro_df = get_macro_features(price_df.index.min(), price_df.index.max())
    macro_df = macro_df.reindex(price_df.index, method="ffill")
    macro_df = flatten_columns(macro_df)

    # 4. Binary flags (shifted for next-day effect)
    flag_df = build_flag_dataframe(price_df.index)
    flag_df = flag_df.shift(1).fillna(0)
    flag_df.columns = flag_df.columns.astype(str)

    # 5. Final dataset
    df = price_df.join([macro_df, flag_df])
    df.dropna(inplace=True)

    df = df.rename(columns={
        f"Open_{ticker}": "Open",
        f"Close_{ticker}": "Close"
    })

    features = df.columns.tolist()

    # 6. Prepare sequences
    X_train, y_train, X_test, y_test, target_scaler = prepare_data(
        df,
        features=features,
        target_cols=TARGET_COLS,
        time_step=LOOKBACK
    )

    # 7. Build + train model
    model = build_model_v1(LOOKBACK, X_train.shape[2])

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # 8. Save artifacts
    model.save(f"{BASE_MODEL_DIR}/{index_name}/{MODEL_FILENAME}")
    joblib.dump(
        target_scaler,
        f"{BASE_MODEL_DIR}/{index_name}/{SCALER_FILENAME}"
    )

    metadata = {
        "index": index_name,
        "ticker": ticker,
        "time_step": LOOKBACK,
        "features": features,
        "target_cols": TARGET_COLS,
        "trained_on": str(datetime.now())
    }

    with open(f"{BASE_MODEL_DIR}/{index_name}/{METADATA_FILENAME}", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model for {index_name}")

# ================= ENTRY =================

if __name__ == "__main__":
    for index, cfg in INDEX_CONFIG.items():
        train_index(index, cfg["ticker"])
