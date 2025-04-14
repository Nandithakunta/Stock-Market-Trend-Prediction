# Stock-Market-Trend-Prediction

A full end-to-end data science project that predicts stock price trends using LSTM deep learning models and Flask for web deployment.

## 🔍 Overview
This project uses historical stock market data to predict the next day's closing price using an LSTM model. Users can input a stock ticker (e.g., AAPL, TSLA, GOOGL), and the app returns a predicted price with a visual chart.

## 📦 Tools Used
- Python
- Flask
- TensorFlow / Keras
- Scikit-learn
- yFinance (Yahoo Finance API)
- Bootstrap (Frontend)
- Matplotlib

## 🚀 Features
- Pulls real-time stock data from Yahoo Finance
- Preprocesses and scales data for LSTM model
- Trains model (if not already trained)
- Predicts next-day price
- Returns prediction and price chart on a stylish web app

## 📂 Project Structure
. ├── stock-price-predictor/ │ ├── app.py │ ├── model_lstm.py │ ├── templates/ │ │ └── index.html │ ├── lstm_model.h5 │ ├── scaler.save │ └── static/

