# Stock Price Prediction Using LSTM

A deep learning project that leverages Long Short-Term Memory (LSTM) neural networks to predict stock market prices based on historical data. This implementation uses stacked LSTM layers to capture temporal dependencies and forecast future stock prices with improved accuracy.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)


## About the Project

This project implements a stock market prediction system using LSTM (Long Short-Term Memory) networks, a specialized type of Recurrent Neural Network (RNN) designed to handle sequential data and capture long-term dependencies. The model analyzes historical stock price patterns to forecast future price movements, providing valuable insights for investment decision-making.

**Disclaimer:** This project is strictly for educational purposes and does not constitute financial advice. Investment decisions should not be based solely on these predictions.

## Features

- **Time Series Forecasting**: Predicts future stock prices based on historical data patterns
- **Stacked LSTM Architecture**: Multiple LSTM layers for enhanced pattern recognition
- **Data Visualization**: Interactive plots showing predicted vs actual stock prices
- **Real-time Data Fetching**: Integration with Yahoo Finance API for live stock data
- **Preprocessing Pipeline**: Data normalization using MinMaxScaler for optimal model performance
- **Performance Metrics**: Evaluation using MSE, RMSE, MAE, and MAPE metrics


## Technologies Used

- **Python 3.x** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework for building LSTM models
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Data visualization and plotting
- **Scikit-learn** - Data preprocessing and evaluation metrics
- **yfinance** - Fetching real-time stock market data
- **Jupyter Notebook** - Interactive development environment


## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/Stock_Prediction_using_LSTM.git
cd Stock_Prediction_using_LSTM
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow yfinance jupyter
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```


## Usage 

1. **Data Collection**: Use the `yfinance` library to fetch historical stock data (Past 200 days).
2. **Data Preprocessing**: Normalize the data using `MinMaxScaler` and create sequences for LSTM input.
3. **Use Models**: 
   - For basic prediction, use `Model_v0.keras`.
   - For advanced features, use `Model_v1.keras`.
   - For NIFTY50 predictions, use `Model_v2.keras`.
4. **Load the model and make predictions**: Refer to the `Resources/Models/Prediction.ipynb` notebook for detailed steps.

**Note**: Every models requires different features as input. Please refer to the respective model notebooks for details.

## Model Overview

This repository has 3 main models:
- **Model_v0**: Basic LSTM model with 100MA and 200MA as features.
- **Model_v1**: Stacked LSTM model with 100MA, 200MA, RSI, MACD, EMA50, SMA20, Stochastic RSI as features.
- **Model_v2**: Focused on predicting NIFTY50 using same features as Model_v1 and GiftNifty (NIFTY50 futures) as an additional feature.

## Dataset

The project uses historical stock market data fetched from **Yahoo Finance**. Key features include:

- **Open Price**: Opening price of the trading day.
- **High Price**: Highest price during the trading day.
- **Low Price**: Lowest price during the trading day.
- **Close Price**: Closing price (primary target variable).
- **Volume**: Number of shares traded.

Data is preprocessed using `MinMaxScaler` for normalization (scaled to a 0-1 range).

## Results

The model performance is evaluated using multiple metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: Square root of MSE for interpretable error magnitude.
- **Mean Absolute Error (MAE)**: Average absolute difference.

1. **Model_v0**:
| Metric | Value |
|--------|-------|
| MSE | 32868.791841153405 |
| RMSE | 181.2975229868114 |
| R-Squared | 0.995645 |

2. **Model_v1**:
| Metric | Value (Open Price) | Value (Close Price) |
|--------|-------|-------|
| MSE | 0.001081810617622401 | 0.0017351335350170734 |
| RMSE | 0.032890889583931916 | 0.04165493410170125 |
| R-Squared | 0.9801708238443777 | 0.9686058305494714 |

3. **Model_v2**:
| Metric | Value (Open Price) | Value (Close Price) |
|--------|-------|-------|
| MSE | 0.0018591559768940211 | 0.0019617906915963797 |
| RMSE | 0.04311793103679745 | 0.044292106425370875 |
| R-Squared | 0.9558426692092862 | 0.9518093332372863 |


## Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

### Contributors

- **Krish Patel** - [@Krish-0610](https://github.com/Krish-0610)
- **Banti Kushwaha** - [@Bantikushwaha](https://github.com/Bantikushwaha)


## Acknowledgments

- **Yahoo Finance API** for providing free stock market data.
- **TensorFlow/Keras** community for excellent deep learning documentation.
- Inspiration from various stock prediction research papers and projects.

***

**Remember**: Past performance does not guarantee future results. Always conduct thorough research before making investment decisions.