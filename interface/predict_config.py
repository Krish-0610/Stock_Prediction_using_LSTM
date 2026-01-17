import os

# =========================
# PREDICTION CONFIG
# =========================

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")

# ---------- MODEL INPUT ----------
LOOKBACK = 100        # must match training
TARGET_COLS = ["Close", "Open"]

# ---------- DIRECTORIES ----------
BASE_MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_FILENAME = "model.keras"
SCALER_FILENAME = "target_scaler.gz"
METADATA_FILENAME = "metadata.json"

# ---------- INDICES ----------
INDEX_CONFIG = {
    "nifty50": {
        "ticker": "^NSEI",
        "display_name": "NIFTY 50"
    },
    "banknifty": {
        "ticker": "^NSEBANK",
        "display_name": "BANK NIFTY"
    },
    "niftyit": {
        "ticker": "^CNXIT",
        "display_name": "NIFTY IT"
    }
}

# ---------- DATA FETCH ----------
FETCH_DAYS = 180      # must be > LOOKBACK
INTERVAL = "1d"

# ---------- INFERENCE ----------
PREDICT_STEPS = 1     # next trading day
RETURN_INVERSE_SCALE = True
