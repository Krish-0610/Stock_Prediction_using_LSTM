import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow import keras
import joblib
import os

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI) for a given dataset."""
    diff = data.diff(1).dropna()
    gain = diff.mask(diff < 0, 0)
    loss = diff.mask(diff > 0, 0).abs()
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD)
    for a given dataset.
    """
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_stoch_rsi(data, window=14, k_window=3, d_window=3):
    """Calculates the Stochastic RSI for a given dataset."""
    rsi = calculate_rsi(data, window)
    min_rsi = rsi.rolling(window=window).min()
    max_rsi = rsi.rolling(window=window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    k_line = stoch_rsi.rolling(window=k_window).mean()
    d_line = k_line.rolling(window=d_window).mean()
    return k_line, d_line

def predict_stock_prices(ticker="^NSEI", days=100):
    
    # Load the trained model and scaler


    model_path = os.path.join(os.path.dirname(__file__), 'Model_v1.keras')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.gz')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Fetch historical data
    end_date = datetime.now()
    time_delta = timedelta(days=200)
    start_date = end_date - time_delta
    df = yf.download(ticker, start=start_date, end=end_date)



    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'])
    macd, signal_line = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal_line
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['StochRSI_K'], df['StochRSI_D'] = calculate_stoch_rsi(df['Close'])
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Select only the 9 features the model was trained on
    feature_cols = [
        ('Open', '^NSEI'),
        ('Close', '^NSEI'),
        ('RSI', ''),
        ('MACD', ''),
        ('MACD_Signal', ''),
        ('EMA_50', ''),
        ('SMA_20', ''),
        ('StochRSI_K', ''),
        ('StochRSI_D', ''),
    ]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols])

    # Prepare the input for prediction
    x_input = np.array([scaled_data[-100:]])  # Last 100 rows
    x_input = x_input.reshape(1, 100, 9)  # Reshape to (1, 100, 9)

    # Predict today's price
    y_predicted_scaled = model.predict(x_input)

    dummy_array = np.zeros(shape=(len(y_predicted_scaled), 9))

    dummy_array[:, 0] = y_predicted_scaled[:, 0] # Predicted 'Close'
    dummy_array[:, 1] = y_predicted_scaled[:, 1] # Predicted 'Open'

    predicted_prices = scaler.inverse_transform(dummy_array)
    print(predicted_prices)
    return {
        "predicted_open": float(predicted_prices[0, 1]),
        "predicted_close": float(predicted_prices[0, 0])
    }


def get_model_evaluation(ticker="^NSEI", days=100):
    """
    Evaluates the model and returns the R-squared scores.
    """
    model_path = os.path.join(os.path.dirname(__file__), 'Model_v1.keras')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.gz')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    now = datetime.now()
    start = datetime(now.year - 10, now.month, now.day)
    df = yf.download(ticker, start, now)

    if df.empty:
        return {"error": "Could not download data from yfinance."}

    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['StochRSI_K'], df['StochRSI_D'] = calculate_stoch_rsi(df['Close'])
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    features = ['Open','Close', 'RSI', 'MACD', 'MACD_Signal', 'EMA_50', 'SMA_20', 'StochRSI_K', 'StochRSI_D']
    data_to_process = df[features]
    
    train_size = int(len(data_to_process) * 0.7)
    data_training = data_to_process.iloc[:train_size]
    data_testing = data_to_process.iloc[train_size:]
    
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []
    time_step = 100

    close_index = features.index('Close')
    open_index = features.index('Open')

    for i in range(time_step, input_data.shape[0]):
        x_test.append(input_data[i-time_step:i])
        y_test.append(input_data[i, [close_index, open_index]])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    
    r2_close = r2_score(y_test[:, 0], y_predicted[:, 0])
    r2_open = r2_score(y_test[:, 1], y_predicted[:, 1])
    
    evaluation = {
        "r2_score_close": r2_close,
        "r2_score_open": r2_open
    }
    print(evaluation)
    return evaluation

if __name__ == '__main__':
    predictions = predict_stock_prices()
    print(predictions)
    evaluation = get_model_evaluation()