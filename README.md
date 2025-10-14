# Stock Price Prediction Using LSTM

A deep learning project that employs **Long Short-Term Memory (LSTM)** neural networks to predict stock market prices from historical data. The implementation utilizes **stacked LSTM layers** to learn temporal dependencies and forecast future prices with improved accuracy.

---

## **About the Project**

This project implements a stock price prediction pipeline using **LSTM (Long Short-Term Memory)** networks — a variant of **Recurrent Neural Networks (RNNs)** built to handle sequential data and long-term dependencies.  
The model studies historical stock price movements to forecast future trends, aiming to assist in data-driven investment analysis.

> **Disclaimer:** This project is for educational and research purposes only. It does **not** provide financial advice or investment recommendations.

---

## **Features**

- **Time Series Forecasting** – Predicts future stock prices using historical patterns.  
- **Stacked LSTM Architecture** – Multiple LSTM layers for improved feature learning.  
- **Data Visualization** – Interactive plots comparing predicted and actual prices.  
- **Live Data Integration** – Fetches real-time data via **Yahoo Finance API**.  
- **Preprocessing Pipeline** – Normalization using **MinMaxScaler** for stable convergence.  
- **Comprehensive Evaluation** – Metrics include **MSE**, **RMSE**, **MAE**, and **MAPE**.

---

## **Technologies Used**

- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**, **Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **yfinance**
- **Jupyter Notebook**

---

## **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/Stock_Prediction_using_LSTM.git
   cd Stock_Prediction_using_LSTM
   ```

2. **Create a Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow yfinance jupyter
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## **Usage**

1. **Data Collection** – Fetch historical stock data (e.g., past 200 days) using `yfinance`.  
2. **Data Preprocessing** – Normalize the data with `MinMaxScaler` and create input sequences.  
3. **Model Selection** –  
   - `Model_v0.keras` → Basic prediction model.  
   - `Model_v1.keras` → Advanced feature model.  
   - `Model_v2.keras` → NIFTY50-focused model with futures data.  
4. **Prediction** – Use `Resources/Models/Prediction.ipynb` for prediction workflow.

> Each model expects specific input features. Refer to the corresponding notebook for details.

---

## **Model Overview**

| Model | Description |
|--------|--------------|
| **Model_v0** | Basic LSTM model using 100MA and 200MA as input features. |
| **Model_v1** | Stacked LSTM model with 100MA, 200MA, RSI, MACD, EMA50, SMA20, and Stochastic RSI. |
| **Model_v2** | Extended LSTM model for **NIFTY50**, adding GiftNifty (NIFTY50 futures) as a new feature. |

---

## **Dataset**

Data sourced from **Yahoo Finance**, containing:

- Open, High, Low, Close Prices  
- Volume (number of shares traded)

All features are scaled between 0 and 1 using **MinMaxScaler** for effective LSTM training.

---

## **Model Performance**

### **Model_v0**

| Metric | Value |
|---------|--------|
| **MSE** | 32868.7918 |
| **RMSE** | 181.2975 |
| **R-Squared** | 0.9956 |

---

### **Model_v1**

| Metric | **Value (Open Price)** | **Value (Close Price)** |
|---------|------------------------|--------------------------|
| **MSE** | 0.0011 | 0.0017 |
| **RMSE** | 0.0329 | 0.0417 |
| **R-Squared** | 0.9802 | 0.9686 |

---

### **Model_v2**

| Metric | **Value (Open Price)** | **Value (Close Price)** |
|---------|------------------------|--------------------------|
| **MSE** | 0.0019 | 0.0020 |
| **RMSE** | 0.0431 | 0.0443 |
| **R-Squared** | 0.9558 | 0.9518 |

---

### **Performance Summary**

- **Model_v0** achieves high R² but operates on unscaled data, leading to large MSE/RMSE.  
- **Model_v1** and **Model_v2** utilize normalized inputs, achieving lower error values.  
- **Model_v1** provides the best generalization balance for both open and close price predictions.

---

## **Contributing**

Contributions are encouraged.  
Follow standard GitHub workflow:

1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:  
   ```bash
   git commit -m "Add some AmazingFeature"
   ```
4. Push to your branch and open a Pull Request.

---

## **Contributors**

- **Krish Patel** – [@Krish-0610](https://github.com/Krish-0610)  
- **Banti Kushwaha** – [@Bantikushwaha](https://github.com/Bantikushwaha)

---

## **Acknowledgments**

- **Yahoo Finance API** for open access to financial data.  
- **TensorFlow/Keras** community for robust deep learning tools.  
- Research literature and open-source projects that inspired this implementation.



---

> **Note:** Past performance is not indicative of future results. Always perform independent research before making financial decisions.

