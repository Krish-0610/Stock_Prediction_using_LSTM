import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Resources", "model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "Resources", "scaler.gz")


def calculate_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = diff.mask(diff < 0, 0)
    loss = diff.mask(diff > 0, 0).abs()
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line


def calculate_stoch_rsi(data, window=14, k_window=3, d_window=3):
    rsi = calculate_rsi(data, window)
    min_rsi = rsi.rolling(window=window).min()
    max_rsi = rsi.rolling(window=window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    k_line = stoch_rsi.rolling(window=k_window).mean()
    d_line = k_line.rolling(window=d_window).mean()
    return k_line, d_line


def make_prediction(ticker="^NSEI"):
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    data = yf.download(ticker, start=start_date, end=end_date)

    data["RSI"] = calculate_rsi(data["Close"])
    data["MACD"], data["Signal_Line"] = calculate_macd(data["Close"])
    data["Stoch_RSI_K"], data["Stoch_RSI_D"] = calculate_stoch_rsi(data["Close"])
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()

    data.dropna(inplace=True)
    data = data.reset_index(drop=True)

    feature_cols = [
        ("Open", "^NSEI"),
        ("Close", "^NSEI"),
        ("RSI", ""),
        ("MACD", ""),
        ("MACD_Signal", ""),
        ("EMA_50", ""),
        ("SMA_20", ""),
        ("StochRSI_K", ""),
        ("StochRSI_D", ""),
    ]
    scaler = MinMaxScaler(feature_range=(0, 1))


    scaled_data = scaler.transform(data[feature_cols])

    # Prepare the input for prediction
    x_input = np.array([scaled_data[-100:]])  # Last 100 rows
    x_input = x_input.reshape(1, 100, 9)  # Reshape to (1, 100, 9)

    # Predict today's price
    y_predicted_scaled = model.predict(x_input)

    dummy_array = np.zeros(shape=(len(y_predicted_scaled), 9))

    dummy_array[:, 0] = y_predicted_scaled[:, 0]  # Predicted 'Close'
    dummy_array[:, 1] = y_predicted_scaled[:, 1]  # Predicted 'Open'

    predicted_prices = scaler.inverse_transform(dummy_array)

    return {
        "predicted_close": predicted_prices[0, 0],
        "predicted_open": predicted_prices[0, 1],
    }