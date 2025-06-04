import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
import pandas_ta as ta
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go
import requests
import random
import time
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon')

# -----------------------------
# File Paths & Directories
# -----------------------------
BASE_DIR = r"C:\Users\saipr\OneDrive\Desktop\MAIN UPDATED PROJECT\DATASETS"
DAILY_DIR = os.path.join(BASE_DIR, "Daily_data")
WEEKLY_DIR = os.path.join(BASE_DIR, "Weekly_data")
MONTHLY_DIR = os.path.join(BASE_DIR, "Monthly_data")
SECTORS_FILE = os.path.join(BASE_DIR, "sectors with symbols.csv")

# -----------------------------
# Functions
# -----------------------------
def get_latest_date():
    today = dt.date.today()
    return today.strftime("%Y-%m-%d")

def clean_and_save_data(data, filepath):
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data.to_csv(filepath, index=False)

def fetch_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1d")
    return vix_data['Close'].iloc[0] if not vix_data.empty else None

def getRSI14_and_BB(csvfilename):
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename)
            if df.empty or 'Close' not in df.columns:
                return 0.00, 0.00, 0.00, 0.00
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['rsi14'] = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)
            if bb is None or df['rsi14'] is None:
                return 0.00, 0.00, 0.00, 0.00
            df['lowerband'] = bb['BBL_20_2.0']
            df['middleband'] = bb['BBM_20_2.0']
            rsival = df['rsi14'].iloc[-1].round(2)
            ltp = df['Close'].iloc[-1].round(2)
            lowerband = df['lowerband'].iloc[-1].round(2)
            middleband = df['middleband'].iloc[-1].round(2)
            return rsival, ltp, lowerband, middleband
        except Exception:
            return 0.00, 0.00, 0.00, 0.00
    else:
        return 0.00, 0.00, 0.00, 0.00

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def predict_future(model, last_sequence, scaler, days=5):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))
        predicted_value = next_pred[0, 0]
        predictions.append(predicted_value)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_value
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_predicted_values(data, epochs=25, start_date=None, end_date=None):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    
    if len(df) < 50:
        print("Not enough data in selected date range. Please select a wider range.")
        return None

    close_data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    
    time_step = 13
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=1)
    
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    train_dates = df['Date'].iloc[time_step + 1 : train_size]
    test_start = train_size + time_step
    test_end = test_start + len(y_test_actual)
    test_dates = df['Date'].iloc[test_start : test_end]
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    return train_r2, test_r2, future_preds, train_dates, test_dates

# -----------------------------
# Execution
# -----------------------------
if __name__ == "__main__":
    stock_symbol = "AAPL"
    stock_data = yf.download(stock_symbol, start="2020-01-01", end=get_latest_date(), interval="1d")
    stock_data.reset_index(inplace=True)

    if 'Close' in stock_data.columns:
        results = get_predicted_values(stock_data, epochs=25)
        if results:
            train_r2, test_r2, future_preds, train_dates, test_dates = results
            print(f"Train R2 Score: {train_r2:.4f}")
            print(f"Test R2 Score: {test_r2:.4f}")
            print(f"Future Predictions: {future_preds}")
        else:
            print("Failed to predict values.")
    else:
        print("Stock data not available.")

