import streamlit as st
import sys
import os
import json
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from interface.predict import predict_next
from interface.predict_config import BASE_MODEL_DIR, METADATA_FILENAME

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock Prediction Portal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# HELPER FUNCTIONS
# =========================

def get_available_indices():
    """Dynamically discover available model indices."""
    indices = []
    if os.path.exists(BASE_MODEL_DIR):
        for item in os.listdir(BASE_MODEL_DIR):
            model_path = os.path.join(BASE_MODEL_DIR, item)
            if os.path.isdir(model_path):
                metadata_path = os.path.join(model_path, METADATA_FILENAME)
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            indices.append({
                                'key': item,
                                'name': item.upper().replace('NIFTY', 'NIFTY '),
                                'ticker': metadata.get('ticker', ''),
                                'trained_on': metadata.get('trained_on', 'Unknown')
                            })
                    except:
                        indices.append({
                            'key': item,
                            'name': item.upper().replace('NIFTY', 'NIFTY '),
                            'ticker': '',
                            'trained_on': 'Unknown'
                        })
    return sorted(indices, key=lambda x: x['name'])

def format_index_name(key):
    """Format index name for display."""
    name = key.upper()
    if name.startswith('NIFTY') and len(name) > 5:
        return name[:5] + ' ' + name[5:]
    return name

def format_date(date_str):
    """Format date string for display."""
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.strftime('%B %d, %Y at %I:%M %p')
    except:
        return date_str

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🎯 Configuration")
    st.markdown("---")
    
    # Get available indices
    available_indices = get_available_indices()
    
    if not available_indices:
        st.error("⚠️ No models found. Please ensure models are available in the models directory.")
        st.stop()
    
    # Index selection
    index_options = {idx['name']: idx['key'] for idx in available_indices}
    selected_name = st.selectbox(
        "Select Stock Index",
        options=list(index_options.keys()),
        help="Choose the stock index you want to predict"
    )
    selected_index = index_options[selected_name]
    
    # Get selected index details
    selected_idx_info = next(idx for idx in available_indices if idx['key'] == selected_index)
    
    st.markdown("---")
    st.markdown("### 📊 Index Information")
    st.info(f"**Ticker:** {selected_idx_info['ticker']}")
    st.info(f"**Trained On:** {format_date(selected_idx_info['trained_on'])}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This portal uses deep learning models to predict 
    the next day's **Open** and **Close** prices for 
    various NIFTY indices.
    
    Predictions are based on:
    - Technical indicators
    - Macro-economic features
    - Historical price patterns
    """)

# =========================
# MAIN CONTENT
# =========================

# Header
st.markdown('<h1 class="main-header">📈 Stock Prediction Portal</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Stock Price Predictions for NIFTY Indices</p>', unsafe_allow_html=True)

# Prediction Section
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"### 🎯 Predicting: {selected_name}")
    
    if st.button("🚀 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {selected_name} and generating prediction..."):
            try:
                # Make prediction
                result = predict_next(selected_index)
                
                # Store in session state
                st.session_state['last_prediction'] = result
                st.session_state['prediction_time'] = datetime.now()
                
                st.success("✅ Prediction generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating prediction: {str(e)}")
                st.session_state['last_prediction'] = None

# Display Results
if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
    result = st.session_state['last_prediction']
    prediction_time = st.session_state.get('prediction_time', datetime.now())
    
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Main prediction cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Predicted Close Price")
        predicted_close = result['prediction']['Close']
        st.metric(
            label="Close Price (₹)",
            value=f"{predicted_close:,.2f}",
            delta=None
        )
    
    with col2:
        st.markdown("### 🟢 Predicted Open Price")
        predicted_open = result['prediction']['Open']
        st.metric(
            label="Open Price (₹)",
            value=f"{predicted_open:,.2f}",
            delta=None
        )
    
    # Additional information
    st.markdown("---")
    
    with st.expander("📋 Prediction Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Index:**")
            st.info(result['index'].upper())
        
        with col2:
            st.markdown("**Ticker:**")
            st.info(result['ticker'])
        
        with col3:
            st.markdown("**Prediction Time:**")
            st.info(prediction_time.strftime('%B %d, %Y at %I:%M:%S %p'))
    
    # Price difference
    price_diff = predicted_close - predicted_open
    price_diff_pct = (price_diff / predicted_open) * 100 if predicted_open > 0 else 0
    
    st.markdown("---")
    st.markdown("### 📈 Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Expected Price Change",
            value=f"₹{price_diff:,.2f}",
            delta=f"{price_diff_pct:.2f}%"
        )
    
    with col2:
        if price_diff > 0:
            st.success(f"📈 Expected upward movement: {price_diff_pct:.2f}%")
        elif price_diff < 0:
            st.warning(f"📉 Expected downward movement: {abs(price_diff_pct):.2f}%")
        else:
            st.info("➡️ Expected neutral movement")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning("""
    **Important:** These predictions are generated by AI models and are for informational purposes only. 
    They should not be considered as financial advice. Stock market investments carry risks, and past 
    performance does not guarantee future results. Always conduct your own research and consult with 
    financial advisors before making investment decisions.
    """)

else:
    # Initial state message
    st.info("👈 Select an index from the sidebar and click 'Generate Prediction' to get started.")
    
    # Show available indices
    st.markdown("---")
    st.markdown("### 📋 Available Indices")
    
    indices_df = pd.DataFrame([
        {
            'Index': idx['name'],
            'Ticker': idx['ticker'],
            'Model Status': '✅ Available'
        }
        for idx in available_indices
    ])
    
    st.dataframe(
        indices_df,
        use_container_width=True,
        hide_index=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Stock Prediction Portal | Powered by Deep Learning | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
    "</div>",
    unsafe_allow_html=True
)
