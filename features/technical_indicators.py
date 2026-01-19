import pandas as pd


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



def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["MA100"] = df["Close"].rolling(100).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    df["RSI"] = calculate_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"] = calculate_macd(df["Close"])
    df["STOCH_RSI_K"], df["STOCH_RSI_D"] = calculate_stoch_rsi(df["Close"])

    return df
