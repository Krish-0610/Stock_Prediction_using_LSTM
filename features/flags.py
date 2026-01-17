import pandas as pd
import yfinance as yf
import os

# =========================
# CONFIG
# =========================

SPX_TICKER = "^GSPC"
CRUDE_TICKER = "CL=F"
project_dir = os.path.join(os.path.dirname(__file__), "..")
data_path = os.path.join(project_dir, "data","flag", "fomc_calendar.csv")

FOMC_DATES = set(pd.read_csv(data_path)["Date"])

# FED_EVENT_DATES = {
#     "2022-03-16", "2022-05-04", "2022-06-15",
#     "2022-07-27", "2022-09-21", "2022-11-02",
#     "2022-12-14",
# }

WAR_EVENT_DATES = {"2022-02-24"}


INDIA_POLICY_EVENT_DATES = {
    "2016-11-08",  # Demonetization
    "2020-03-24",  # COVID lockdown
    "2023-02-01",  # Union budget
    "2024-02-01",
}

# =========================
# DATA FETCH (ONCE)
# =========================


def fetch_returns(start_date: str, end_date: str) -> pd.DataFrame:
    tickers = [SPX_TICKER, CRUDE_TICKER]

    df = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]

    returns = df.pct_change().fillna(0.0)
    returns.columns = ["spx_ret", "crude_ret"]
    return returns


# =========================
# FLAG LOGIC (PURE)
# =========================


def build_flag_dataframe(dates: pd.DatetimeIndex) -> pd.DataFrame:
    returns = fetch_returns(
        start_date=dates.min().strftime("%Y-%m-%d"),
        end_date=(dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    flags = []

    for d in dates:
        ds = d.strftime("%Y-%m-%d")

        spx_ret = returns.loc[d, "spx_ret"] if d in returns.index else 0.0
        crude_ret = returns.loc[d, "crude_ret"] if d in returns.index else 0.0

        flags.append(
            {
                "fed_policy_flag": int(ds in FOMC_DATES),
                "global_crash_flag": int(spx_ret <= -0.02),
                "oil_shock_flag": int(crude_ret >= 0.03),
                "war_flag": int(ds in WAR_EVENT_DATES),
                "india_policy_flag": int(ds in INDIA_POLICY_EVENT_DATES),
            }
        )

    return pd.DataFrame(flags, index=dates)
