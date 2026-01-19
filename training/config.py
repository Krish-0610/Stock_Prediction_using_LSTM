# =========================
# GLOBAL TRAINING CONFIG
# =========================

from datetime import datetime
import os

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")

# ---------- DATA ----------
LOOKBACK = 100
TRAIN_YEARS = 10
SPLIT_RATIO = 0.7

# ---------- MODEL ----------
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# ---------- TARGET ----------
TARGET_COLS = ["y_close", "y_open"]

# ---------- DIRECTORIES ----------
BASE_MODEL_DIR = os.path.join(PROJECT_DIR, "models")
TARGET_SCALER_FILENAME = "target_scaler.gz"
INPUT_SCALER_FILENAME = "input_scaler.gz"
MODEL_FILENAME = "model.keras"
METADATA_FILENAME = "metadata.json"

# ---------- INDICES ----------
INDEX_CONFIG = {
    "nifty50": {
        "ticker": "^NSEI",
        "description": "NIFTY 50 Index"
    },
    "banknifty": {
        "ticker": "^NSEBANK",
        "description": "NIFTY Bank Index"
    },
    "niftyit": {
        "ticker": "^CNXIT",
        "description": "NIFTY IT Index"
    },
    "niftyauto":{
        "ticker": "^CNXAUTO",
        "description": "NIFTY AUTO Index"
    },
    "niftymetal":{
        "ticker": "^CNXMETAL",
        "description": "NIFTY METAL Index"
    },
    "niftyfmcg":{
        "ticker": "^CNXFMCG",
        "description": "NIFTY FMCG Index"
    }
}

# ---------- TIME ----------
def get_train_window():
    end = datetime.now()
    start = datetime(end.year - TRAIN_YEARS, end.month, end.day)
    return start, end
