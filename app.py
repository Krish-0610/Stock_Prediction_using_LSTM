import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz

# Try/Except to handle potential missing local modules during copy-paste
try:
    from interface.predict import predict_next_day
    from interface.predict_config import INDEX_CONFIG
    from features.technical_indicators import add_technical_indicators
except ImportError:
    # Fallback config for demonstration if local modules are missing
    st.error("Local modules not found. Ensure 'interface' and 'features' packages exist.")
    st.stop()

# ---------------- Configuration ----------------
THEME = {
    "background": "#0e1117",
    "text": "#fafafa",
    "bull": "#00C805",  # Bright Green
    "bear": "#FF333A",  # Bright Red
    "grid": "#262730",
}

st.set_page_config(
    page_title="MarketSeq | High-Fidelity Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS Injection ----------------
def inject_css():
    st.markdown("""
        <style>
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #a0a0a0;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(123, 47, 247, 0.1));
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #00d4ff;
        font-weight: 600;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1.5rem;
    }
    
    /* Radio buttons */
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        color: #e0e0e0;
    }
    
    /* Checkboxes */
    [data-testid="stSidebar"] .stCheckbox > label {
        font-weight: 500;
        color: #e0e0e0;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.5), transparent);
    }
    
    /* Caption styling */
    .stCaption {
        font-size: 0.9rem;
        color: #808080;
        font-style: italic;
    }
    
    /* Prediction badge */
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
    }
</style>
    """, unsafe_allow_html=True)

# ---------------- Logic Layer ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, days):
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
    start = end - pd.Timedelta(days=days)
    
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        if df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)

        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def get_market_status():
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.datetime.now(tz)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now.weekday() >= 5:
        return "Closed (Weekend)", "off"
    elif market_open <= now <= market_close:
        return "Market Open", "active"
    else:
        return "Market Closed", "off"

# ---------------- Charting Layer ----------------
def render_chart(df, prediction_day, pred_close, last_close, indicators):
    rows = 1 + int(indicators['rsi']) + int(indicators['macd']) + int(indicators['stoch'])
    
    # Dynamic row heights: Price chart gets 60% of space
    row_heights = [0.6] + [0.4 / (rows - 1)] * (rows - 1) if rows > 1 else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=["Price Action"] + 
                       (["RSI (14)"] if indicators['rsi'] else []) + 
                       (["MACD"] if indicators['macd'] else []) + 
                       (["Stoch RSI"] if indicators['stoch'] else [])
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", increasing_line_color=THEME['bull'], decreasing_line_color=THEME['bear']
    ), row=1, col=1)

    # 2. Moving Averages
    if indicators['ma']:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA100"], name="MA 100", line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA 200", line=dict(color='purple', width=1)), row=1, col=1)
    
    if indicators['ema']:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color='cyan', width=1)), row=1, col=1)

    # 3. Prediction Line
    fig.add_trace(go.Scatter(
        x=[df.index[-1], prediction_day], y=[last_close, pred_close],
        mode="lines+markers+text", name="Model Prediction",
        line=dict(color='yellow', dash="dot", width=2),
        text=[None, f"{pred_close:.2f}"], textposition="top right"
    ), row=1, col=1)

    curr_row = 2
    
    # 4. RSI
    if indicators['rsi']:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color='#A3A3A3')), row=curr_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=curr_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=curr_row, col=1)
        curr_row += 1

    # 5. MACD
    if indicators['macd']:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color='cyan')), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color='orange')), row=curr_row, col=1)
        fig.add_bar(x=df.index, y=df["MACD"]-df["MACD_Signal"], name="Hist", marker_color='gray', row=curr_row, col=1)
        curr_row += 1

    # 6. Stochastic
    if indicators['stoch']:
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_K"], name="%K", line=dict(color='cyan')), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_D"], name="%D", line=dict(color='orange')), row=curr_row, col=1)
        curr_row += 1

    fig.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)')
    )
    
    # Hide weekends on x-axis logic would go here, but requires complex range-breaking
    
    return fig

# ---------------- Main UI ----------------
def main():
    inject_css()
    
    # Sidebar
    with st.sidebar:
        st.header("Control Panel")
        selected_index = st.radio("Instrument", list(INDEX_CONFIG.keys()), index=0)
        
        st.subheader("Timeframe")
        range_map = {"6M": 180, "1Y": 365, "3Y": 1095}
        range_choice = st.segmented_control("Lookback", list(range_map.keys()), default="1Y")
        
        st.subheader("Technical Overlays")
        indicators = {
            "ma": st.toggle("MA 100/200", True),
            "ema": st.toggle("EMA 50", True),
            "rsi": st.toggle("RSI", True),
            "macd": st.toggle("MACD", True),
            "stoch": st.toggle("Stochastic", False)
        }
        
        st.divider()
        st.caption(f"v2.1 | MarketSeq Engine")

    ticker = INDEX_CONFIG[selected_index]["ticker"]
    
    # Header
    status_text, status_state = get_market_status()
    c_head_1, c_head_2 = st.columns([3, 1])
    with c_head_1:
        st.title(f"{selected_index}")
        st.caption(f"Ticker: {ticker} | Strategy: Hybrid LSTM/Technical")
    with c_head_2:
        color = "green" if status_state == "active" else "red"
        st.markdown(f"<div style='text-align: right; color: {color}; padding-top: 20px;'>● {status_text}</div>", unsafe_allow_html=True)

    # Load Data & Prediction
    with st.spinner("Crunching numbers..."):
        df = load_data(ticker, range_map[range_choice])
        if df is None:
            st.warning("No data returned from API. Try a different timeframe.")
            st.stop()
            
        df = add_technical_indicators(df)
        result = predict_next_day(selected_index)

    # Prediction Logic
    pred_open = float(result["Open"])
    pred_close = float(result["Close"])
    last_close = float(df["Close"].iloc[-1])
    delta = pred_close - last_close
    pct = (delta / last_close) * 100
    
    # Determine Trend
    ma_cross = df["MA100"].iloc[-1] > df["MA200"].iloc[-1]
    regime = "BULLISH" if ma_cross else "BEARISH"
    regime_color = "normal" if ma_cross else "inverse"

    # Prediction Hero Section
    st.markdown("### Model Forecast")
    m1, m2, m3, m4 = st.columns(4)
    
    m1.metric("Predicted Close", f"{pred_close:,.2f}", delta_color="off")
    m2.metric("Expected Move", f"{delta:+.2f}", f"{pct:+.2f}%")
    m3.metric("Predicted Open", f"{pred_open:,.2f}")
    m4.metric("Market Regime", regime, delta_color=regime_color)
    
    st.divider()

    # Tabs for Visuals vs Data
    tab_chart, tab_data = st.tabs(["📈 Technical Analysis", "📄 Raw Data"])
    
    with tab_chart:
        # Calculate next trading day
        now_ts = pd.Timestamp.now(tz="Asia/Kolkata")
        prediction_day = now_ts + pd.Timedelta(days=1) if now_ts.hour > 15 else now_ts
        
        fig = render_chart(df, prediction_day, pred_close, last_close, indicators)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab_data:
        st.dataframe(
            df.sort_index(ascending=False).style.format("{:.2f}"),
            height=400,
            use_container_width=True
        )

if __name__ == "__main__":
    main()