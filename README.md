# Portfolio Forecasting System

A Python-based portfolio forecasting system that demonstrates multiple forecasting models from classical to deep learning approaches through a simple web UI.

## Features

- Classical forecasting models (ARIMA, VAR)
- Deep learning models (DeepAR, Temporal Fusion Transformer)
- Interactive Streamlit dashboard
- Mock stock data for testing and development
- Performance evaluation metrics (RMSE, MAPE)

## Models Included

1. **Baseline Models**:
   - Naive forecast
   - ARIMA

2. **Multivariate Models**:
   - Vector Autoregression (VAR)

3. **Deep Learning Models**:
   - DeepAR (Amazon's probabilistic forecasting)
   - Temporal Fusion Transformer (TFT)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
portfolio_forecaster/
├── data/                   # Mock stock data
├── models/                 # Forecasting models
├── utils/                  # Utility functions
├── app.py                  # Streamlit dashboard
├── run_baselines.py        # Baseline model runner
├── run_backtest.py         # Backtesting script
└── requirements.txt        # Dependencies
```

## Data

This project uses mock stock data (`data/mock_stock_data.csv`) containing OHLCV data for 5 tickers over ~500 days for demonstration purposes.

## Note

This is a hackathon prototype designed to showcase various forecasting approaches in a unified framework.