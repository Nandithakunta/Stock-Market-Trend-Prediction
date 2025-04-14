# app.py
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend to avoid tkinter warnings

from flask import Flask, request, jsonify, render_template
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.save"

# ✅ Train model if not already trained
def train_model(ticker='AAPL'):
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    seq_length = 60
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

# ✅ Load model and scaler if exists, otherwise train
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    model, scaler = train_model()
else:
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

# ✅ Prepare data for prediction
def prepare_input(data, seq_length=60):
    scaled_data = scaler.transform(data)
    X = []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X

# ✅ Generate prediction plot as base64
def plot_prediction(actual, predicted, dates):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates[-len(predicted):], actual[-len(predicted):], label="Actual Price")
    ax.plot(dates[-len(predicted):], predicted, label="Predicted Price")
    ax.legend()
    ax.set_title("Stock Price Prediction")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')

    df = yf.download(ticker, period='1y')
    df = df[['Close']].dropna()
    if len(df) < 60:
        return jsonify({'error': 'Not enough data for prediction'})

    X = prepare_input(df)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    current_price = df['Close'].values[-1].item()         # ✅ Proper scalar extraction
    predicted_price = predictions[-1].item()              # ✅ Proper scalar extraction
    delta = predicted_price - current_price

    chart = plot_prediction(df['Close'].values, predictions, df.index)

    return jsonify({
        'ticker': ticker,
        'current_price': round(current_price, 2),
        'predicted_price': round(predicted_price, 2),
        'delta': round(delta, 2),
        'chart': chart
    })

if __name__ == '__main__':
    app.run(debug=True)
