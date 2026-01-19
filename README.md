# Stock Price Prediction Using LSTM
## Overview
This project implements a **time-series forecasting pipeline** using **Long Short-Term Memory (LSTM)** neural networks to predict **next-day Open and Close prices** of stock indices.  
The system is designed as an **academic and experimental framework**, focusing on how historical price data, technical indicators, and macro-level signals can be combined for sequential prediction.

The goal is **modeling and analysis**, not trading or profit generation.

## Problem Statement
Traditional stock prediction models rely only on historical prices: `P(t−n) … P(t) → P(t+1)` such models mostly learn **price mapping**, not market reaction.

This project extends the formulation to: `[Historical Prices + Technical Indicators + Macro Signals] → P(t+1)`
The intent is to make the model **macro-aware**, so that large overnight or global shocks can influence predictions.

## Project Structure
```bash
Stock_Prediction_using_LSTM/
├── app
│   └── streamlit_app.py      # Streamlit dashboard entry point
│
├── data
│   ├── flag                  # Event markers (FOMC, Macro events)
│   ├── news                  # Raw sentiment/news data
│   └── processed             # Cleaned datasets ready for training
│
├── features                  # Feature Engineering logic
│   ├── flags.py              # Event boolean logic
│   ├── macro_features.py     # Macroeconomic factors
│   └── technical_indicators.py
│
├── interface                 # Inference/Prediction logic
│   ├── predict.py            # Inference pipeline
│   └── predict_config.py     # Inference configurations
│
├── models                    # Serialized models & artifacts
│   ├── banknifty             # Index-specific artifacts
│   │   ├── model.keras       # Trained Model
│   │   ├── metadata.json     # Hyperparams & threshold data
│   │   └── *.gz              # Scalers (Input/Target)
│   ├── nifty50
│   └── [niftyauto, niftyfmcg, niftyit, niftymetal]
│
├── training                  # Training Pipeline
│   ├── config.py             # Hyperparameters & paths
│   ├── model.py              # Architecture definition
│   ├── prepare_data.py       # Preprocessing scripts
│   └── train_model.py        # Training loop execution
│
└── utils
    └── flatten.py            # Utility scripts

```

## Data Sources
- **Historical market data**: Yahoo Finance (`yfinance`)
- **Price data**:
    - Open
    - High
    - Low
    - Close
    - Volume
- **Global and macro proxies** (used selectively depending on index):
    - S&P 500
    - NASDAQ
    - Dow Jones
    - USD/INR
    - Crude Oil
    - India VIX
## Feature Engineering
### 1. Price-Based features
- Raw OHLC prices
- Log returns
- Moving Averages
### 2. Technical Indicators
- SMA (20, 100, 200)
- EMA
- RSI
- MACD
- Stochastic RSI
### 3. Macro / Proxy Features
Used to approximate **market reaction**, not sentiment:
- Global index returns
- Volatility index (VIX)
- Currency movement
- Commodity prices
### 4. Binary Event Flags (Experimental)
High-signal, low-noise indicators:
- Fed policy event
- Global market crash
- Oil shock
- Geopolitical stress
Flags are attached **only to the next trading day** to avoid leakage.
## Data Preprocessing Pipeline
1. Missing value handling
2. Feature alignment by date
3. Scaling using `MinMaxScaler`
4. Sliding window sequence creation:
    `lookback window → next-day prediction`
5. Train / validation split (time-aware)
## Model Architecture
### Core Model
- Stacked LSTM layers
- Dropout for regularization
- Dense output layer
## Training
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Sequence length: configurable
- Training is performed using both notebooks and script pipelines
## Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Directional Accuracy (up/down correctness)
- Visual comparison of predicted vs actual prices
Metrics are interpreted cautiously due to the stochastic nature of markets.

## Prediction Pipeline
1. Load trained model and scaler
2. Fetch latest available market data
3. Recompute required features
4. Generate last sequence window
5. Predict:
    - Next-day Open
    - Next-day Close
6. Inverse scale predictions
## Key Design Decisions
- **No future data leakage**
- **Macro signals treated as context, not truth**
- **Prediction ≠ trading strategy**
- Focus on **explainability and structure** rather than accuracy claims
## Limitations
- Market efficiency limits predictability
- Macro features are proxies, not causal signals 
- Binary event flags are heuristic-based
- Not suitable for real-world trading without risk management
## Future Work
- Regime-aware models (bull/bear detection)
- Attention-based architectures
- Probabilistic forecasting instead of point prediction
- Better evaluation under stress events
## Disclaimer
This project is strictly for **educational and research purposes**.  
It does **not** provide financial advice or trading recommendations.
