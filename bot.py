import os
import numpy as np
import pandas as pd
import yfinance as yf
import logging
import asyncio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Load Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# ✅ Load Alpaca API Credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Default to Paper Trading

# ✅ Debugging: Print only if necessary (REMOVE in production)
print("Loaded Telegram Token:", bool(TELEGRAM_BOT_TOKEN))  # Will print True if loaded, False if not
print("Loaded Alpaca API Key:", bool(ALPACA_API_KEY))
print("Loaded Alpaca Secret Key:", bool(ALPACA_SECRET_KEY))
print("Alpaca Base URL:", ALPACA_BASE_URL)

# ✅ Raise an error if any critical environment variable is missing
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("⚠️ TELEGRAM_BOT_TOKEN is missing in .env file!")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("⚠️ Alpaca API Key or Secret Key is missing in .env file!")
# ✅ Logging for Debugging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ✅ Define Stock Symbols (Indian + US)
STOCKS = ["HDFC.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS", "TSLA", "AAPL", "GOOGL"]

# ✅ LSTM Model for Predicting Stock Trends
def create_model():
    model = Sequential([
        Input(shape=(60, 1)),  # Fix TensorFlow warning
        LSTM(50, activation="relu", return_sequences=True),
        LSTM(50, activation="relu"),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ✅ Fetch Stock Data (Fix Yahoo Finance Errors)
def fetch_stock_data(symbol, period="60d"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)

        if df.empty:
            logging.warning(f"No data found for {symbol}. Check if it's listed on Yahoo Finance.")
            return None

        df = df[['Close']]  # Fix Column Name Issue
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# ✅ Predict Stock Trend
def predict_stock(symbol):
    df = fetch_stock_data(symbol)

    if df is None or df.empty:
        return f"⚠️ No data found for {symbol}."

    model = create_model()
    data = df['Close'].values.reshape(-1, 1)
    data = np.reshape(data, (data.shape[0], 1, 1))  # Fix TensorFlow input

    prediction = model.predict(data[-60:].reshape(1, 60, 1))[0][0]
    latest_price = df['Close'].iloc[-1]

    signal = "📈 BUY" if prediction > latest_price else "📉 SELL"
    return f"{symbol} Prediction: {prediction:.2f} ({signal})"

# ✅ Telegram Bot Setup
app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

# ✅ Command: /start
async def start(update: Update, context):
    await update.message.reply_text("Hello! I am your AI Trading Bot. Use /predict <symbol>.")

# ✅ Command: /predict <symbol>
async def predict(update: Update, context):
    if len(context.args) == 0:
        await update.message.reply_text("❌ Please provide a stock symbol. Example: /predict HDFC.NS")
        return

    symbol = context.args[0].upper()
    if symbol not in STOCKS:
        await update.message.reply_text("⚠️ Invalid symbol. Try: HDFC.NS, TCS.NS, TSLA, AAPL, etc.")
        return

    result = predict_stock(symbol)
    await update.message.reply_text(result)

# ✅ Add Handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("predict", predict))

# ✅ Run Telegram Bot
if __name__ == "__main__":
    logging.info("✅ AI Trading Bot is running...")
    app.run_polling()
