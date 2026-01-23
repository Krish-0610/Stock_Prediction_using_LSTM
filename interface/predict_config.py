import os
import pandas as pd
from datetime import datetime, timedelta

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")

LOOKBACK = 100
TARGET_COLS = ["Close", "Open"]

BASE_MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_FILENAME = "model.keras"
TARGET_SCALER_FILENAME = "target_scaler.gz"
INPUT_SCALER_FILENAME = "input_scaler.gz"
METADATA_FILENAME = "metadata.json"

INDEX_CONFIG = {
    "nifty50": {"ticker": "^NSEI"},
    "banknifty": {"ticker": "^NSEBANK"},
    "niftyit": {"ticker": "^CNXIT"},
    "niftyauto": {"ticker": "^CNXAUTO"},
    "niftymetal": {"ticker": "^CNXMETAL"},
}

FETCH_DAYS = 500

RETURN_INVERSE_SCALE = True


def get_fetch_window():
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    market_close = now.normalize() + pd.Timedelta(hours=15, minutes=30)

    if now < market_close:
        data_cutoff = now.normalize() - pd.Timedelta(days=1)
        prediction_day = now.normalize()  # today
    else:
    # Market CLOSED
        data_cutoff = now.normalize()
        prediction_day = now.normalize() + pd.Timedelta(days=1)

    end = data_cutoff + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=FETCH_DAYS)
    return start, end
