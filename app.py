import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import datetime
import os
import sys

# ------------------ Path Fix ------------------
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from interface.predict import predict_next_day
from interface.predict_config import INDEX_CONFIG

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Market Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

# ------------------ Global Styles ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1.5rem;
}
.metric-card {
    background: #161b22;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #30363d;
}
.metric-title {
    font-size: 0.85rem;
    color: #8b949e;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
}
.metric-delta-up {
    color: #3fb950;
}
.metric-delta-down {
    color: #f85149;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Helpers ------------------
def get_ticker(index_name):
    return INDEX_CONFIG.get(index_name, {}).get("ticker")

@st.cache_data(ttl=3600)
def load_price_data(ticker):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365 * 5)
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)
    return df

def price_delta(pred, last):
    delta = pred - last
    pct = (delta / last) * 100
    return delta, pct

# ------------------ Sidebar ------------------
st.sidebar.title("Configuration")

selected_index = st.sidebar.selectbox(
    "Market Index",
    list(INDEX_CONFIG.keys())
)

run_model = st.sidebar.button("Run Prediction")

ticker = get_ticker(selected_index)

st.sidebar.markdown("---")
st.sidebar.caption("Model: LSTM (Next-Day Forecast)")
st.sidebar.caption("Data: Yahoo Finance")

# ------------------ Header ------------------
st.title(f"{selected_index.upper()} — Market Forecast")

# ------------------ Prediction Section ------------------
if run_model:
    with st.spinner("Running model inference..."):
        result = predict_next_day(selected_index)

        pred_open = float(result["Open"])
        pred_close = float(result["Close"])

        df = load_price_data(ticker)
        last_close = float(df["Close"].iloc[-1])

        delta_val, delta_pct = price_delta(pred_close, last_close)
        delta_class = "metric-delta-up" if delta_val >= 0 else "metric-delta-down"

        st.markdown("### Next Trading Day Forecast")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Open</div>
                <div class="metric-value">{pred_open:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted Close</div>
                <div class="metric-value">{pred_close:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Expected Move vs Last Close</div>
                <div class="metric-value {delta_class}">
                    {delta_val:+.2f} ({delta_pct:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ------------------ Chart Section ------------------
if ticker:
    df = load_price_data(ticker)

    st.markdown("### Price Action (Context)")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.update_layout(
        height=550,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Ticker not configured for selected index.")
