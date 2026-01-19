import yfinance as yf
import pandas as pd

tickers = {
    "sp500_ret": "^GSPC",
    "nasdaq_ret": "^IXIC",
    "vix_ret": "^INDIAVIX",
    "crude_ret": "CL=F",
    "usd_inr_ret": "USDINR=X"
}

def get_macro_features(start, end):
    dfs = []

    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start, end=end)
        df = df[["Close"]].rename(columns={"Close": name})
        df[name] = df[name].pct_change()
        dfs.append(df)

    macro_df = pd.concat(dfs, axis=1)
    macro_df = macro_df.shift(1)     # attach overnight info

    return macro_df
