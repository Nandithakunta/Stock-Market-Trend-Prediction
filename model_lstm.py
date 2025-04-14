import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")[['Close']]
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

X, y = [], []
seq = 60
for i in range(seq, len(scaled_data)):
    X.append(scaled_data[i - seq:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(seq, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16)

model.save("lstm_model.h5")
joblib.dump(scaler, "scaler.save")
print("✅ Model and scaler saved!")
