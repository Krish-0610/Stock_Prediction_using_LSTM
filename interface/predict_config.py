import os
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
    "niftyfmcg": {"ticker": "^CNXFMCG"},
}

FETCH_DAYS = 500

RETURN_INVERSE_SCALE = True

def get_fetch_window():
    end = datetime.now()
    start = end - timedelta(days=FETCH_DAYS)
    return start, end