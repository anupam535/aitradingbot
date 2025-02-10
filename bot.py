import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from alpaca_trade_api.rest import REST, TimeFrame
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# 🔹 Set API Keys (Replace with actual keys)
ALPACA_API_KEY = "PKT5UWN61WQH0D8UOYPS"
ALPACA_SECRET_KEY = "VpvTIZXUDEBhdeIpmL6ubQFMWO48thNRSabb9tmp"
TELEGRAM_BOT_TOKEN = "6198191947:AAHnUnTQU3BDWoG6Qr5vTerqMXhQdvbvQyM"
BASE_URL = "https://paper-api.alpaca.markets"

# 🔹 Initialize Alpaca API
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)

# 🔹 AI Model: LSTM for Stock Prediction
def create_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# 🔹 Data Preprocessing
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    return scaled_data, scaler

# 🔹 Fetch Real-Time Stock Data for Indian Stocks (NSE/BSE)
def get_indian_stock_data(symbol, period="60d", interval="1d"):
    try:
        stock = yf.Ticker(symbol + ".NS")  # Add ".NS" for NSE stocks
        df = stock.history(period=period, interval=interval)
        return df[["Close"]].reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Error fetching data for {symbol}: {e}")
        return None

# 🔹 Fetch Real-Time Stock Data for US Stocks
def get_us_stock_data(symbol, timeframe="1Day"):
    try:
        bars = alpaca.get_bars(symbol, TimeFrame.Day if timeframe == "1Day" else TimeFrame.Hour, limit=60).df
        return bars[["close"]].reset_index(drop=True).rename(columns={"close": "Close"})
    except Exception as e:
        print(f"⚠️ Error fetching US stock data for {symbol}: {e}")
        return None

# 🔹 AI-Based Price Prediction
def predict_price(model, data):
    data, scaler = preprocess_data(data)
    X_test = np.array([data[-60:]])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    prediction = model.predict(X_test)
    return scaler.inverse_transform(prediction)[0][0]

# 🔹 Swing Trade Strategy for Indian & US Stocks
def generate_swing_signals(symbols):
    results = []
    model = create_lstm_model()

    for symbol in symbols:
        if symbol.endswith(".NS") or symbol.endswith(".BSE"):  # Indian Stocks
            data = get_indian_stock_data(symbol.replace(".NS", "").replace(".BSE", ""))
        else:  # US Stocks
            data = get_us_stock_data(symbol)

        if data is None or len(data) < 60:
            results.append(f"⚠️ No sufficient swing data for {symbol}")
            continue

        predicted_price = predict_price(model, data)
        last_price = data["Close"].iloc[-1]
        percent_change = ((predicted_price - last_price) / last_price) * 100

        if percent_change > 2:
            signal = "STRONG BUY ✅"
        elif percent_change > 0:
            signal = "BUY ✅"
        elif percent_change < -2:
            signal = "STRONG SELL ❌"
        else:
            signal = "HOLD 🔄"

        results.append(f"{symbol} → {signal} (Pred: {predicted_price:.2f}, Now: {last_price:.2f}, Change: {percent_change:.2f}%)")

    return "\n".join(results)

# 🔹 Intraday Trade Strategy for Indian & US Stocks
def generate_intraday_signals(symbols):
    results = []
    model = create_lstm_model()

    for symbol in symbols:
        if symbol.endswith(".NS") or symbol.endswith(".BSE"):  # Indian Stocks
            data = get_indian_stock_data(symbol.replace(".NS", "").replace(".BSE", ""), period="2d", interval="1h")
        else:  # US Stocks
            data = get_us_stock_data(symbol, timeframe="1Hour")

        if data is None or len(data) < 60:
            results.append(f"⚠️ No sufficient intraday data for {symbol}")
            continue

        predicted_price = predict_price(model, data)
        last_price = data["Close"].iloc[-1]
        percent_change = ((predicted_price - last_price) / last_price) * 100

        if percent_change > 1:
            signal = "INTRADAY BUY ✅"
        elif percent_change < -1:
            signal = "INTRADAY SELL ❌"
        else:
            signal = "HOLD 🔄"

        results.append(f"{symbol} → {signal} (Pred: {predicted_price:.2f}, Now: {last_price:.2f}, Change: {percent_change:.2f}%)")

    return "\n".join(results)

# 🔹 Telegram Bot Integration
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("📈 AI Trading Bot Activated! Use:\n/swing <stocks> - Swing Trading\n/intraday <stocks> - Intraday Trading")

def swing(update: Update, context: CallbackContext) -> None:
    if len(context.args) == 0:
        update.message.reply_text("⚠️ Provide stock symbols. Example: /swing AAPL TSLA RELIANCE.NS TCS.NS")
        return
    
    symbols = [arg.upper() for arg in context.args]
    try:
        result = generate_swing_signals(symbols)
        update.message.reply_text(result)
    except Exception as e:
        update.message.reply_text(f"⚠️ Error processing swing stocks: {e}")

def intraday(update: Update, context: CallbackContext) -> None:
    if len(context.args) == 0:
        update.message.reply_text("⚠️ Provide stock symbols. Example: /intraday AAPL TSLA INFY.NS HDFC.NS")
        return
    
    symbols = [arg.upper() for arg in context.args]
    try:
        result = generate_intraday_signals(symbols)
        update.message.reply_text(result)
    except Exception as e:
        update.message.reply_text(f"⚠️ Error processing intraday stocks: {e}")

def main():
    updater = Updater(TELEGRAM_BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("swing", swing))
    dp.add_handler(CommandHandler("intraday", intraday))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
