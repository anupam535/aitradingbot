import os
import telegram
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# **🔑 Load Environment Variables**
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

if not TELEGRAM_BOT_TOKEN or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Missing API keys in environment variables!")

# **📊 Initialize Alpaca API**
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

# **📊 Fetch Intraday Data**
def fetch_intraday_data(symbol, timeframe="1Min", limit=200):
    try:
        bars = api.get_barset(symbol, timeframe, limit=limit)
        df = bars.df
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df.rename(columns={'time': 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching intraday data for {symbol}: {e}")
        return None

# **📈 Calculate Technical Indicators**
def calculate_indicators(df):
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["SMA_5"] = df["close"].rolling(window=5).mean()
    df["SMA_15"] = df["close"].rolling(window=15).mean()
    df["Volatility"] = df["close"].rolling(window=10).std()
    df["MACD"] = ta.trend.MACD(df["close"]).macd()
    df.dropna(inplace=True)
    return df

# **📊 Train Machine Learning Model**
def train_intraday_model(data):
    if data is None or data.empty:
        return None
    data = calculate_indicators(data)
    data["Target"] = (data["close"].shift(-1) > data["close"]).astype(int)
    X = data[["SMA_5", "SMA_15", "Volatility", "RSI", "MACD"]]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# **📊 Get Trading Signal**
def get_trading_signal(symbol, model):
    data = fetch_intraday_data(symbol)
    if data is None or data.empty or model is None:
        return None
    data = calculate_indicators(data)
    X_recent = data[["SMA_5", "SMA_15", "Volatility", "RSI", "MACD"]].tail(1)
    if X_recent.empty:
        return None
    prediction = model.predict(X_recent)[0]
    return "Buy" if prediction == 1 else "Sell/Hold"

# **📊 Telegram Bot Handlers**
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Welcome to the AI Trading Bot! Use /analyze <symbol> or /intraday_signals <symbol> to get started.")

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("❌ Usage: /analyze <symbol> (e.g., /analyze AAPL)")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"📊 Analyzing {symbol}...")
    try:
        data = fetch_intraday_data(symbol)
        if data is None or data.empty:
            await update.message.reply_text(f"⚠️ Could not retrieve data for {symbol}.")
            return
        data = calculate_indicators(data)
        latest_rsi = data["RSI"].iloc[-1]
        signal = "Buy" if latest_rsi < 30 else "Sell" if latest_rsi > 70 else "Hold"
        await update.message.reply_text(f"📈 Stock: {symbol}\n📊 RSI: {latest_rsi:.2f}\n🔹 Signal: {signal}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing {symbol}: {str(e)}")

async def intraday_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("❌ Usage: /intraday_signals <symbol> (e.g., /intraday_signals AAPL)")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"📊 Fetching intraday signals for {symbol}...")
    try:
        data = fetch_intraday_data(symbol)
        if data is None or data.empty:
            await update.message.reply_text(f"⚠️ Could not retrieve data for {symbol}.")
            return
        model = train_intraday_model(data)
        if model is None:
            await update.message.reply_text(f"⚠️ Could not train model for {symbol}.")
            return
        signal = get_trading_signal(symbol, model)
        await update.message.reply_text(f"📊 Intraday Signal for {symbol}: {signal}")
    except Exception as e:
        await update.message.reply_text(f"❌ Error processing {symbol}: {str(e)}")

async def intraday_signals_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT", "NVDA", "META", "NFLX", "AMD", "INTC"]
    await update.message.reply_text("📊 Fetching intraday signals for multiple stocks...")
    signals = []
    for symbol in symbols:
        try:
            data = fetch_intraday_data(symbol)
            if data is None or data.empty:
                signals.append(f"⚠️ No data for {symbol}.")
                continue
            model = train_intraday_model(data)
            if model is None:
                signals.append(f"⚠️ Could not train model for {symbol}.")
                continue
            signal = get_trading_signal(symbol, model)
            signals.append(f"📊 {symbol}: {signal}")
        except Exception as e:
            signals.append(f"❌ Error processing {symbol}: {str(e)}")
    
    await update.message.reply_text("\n".join(signals))

# **🔧 Run Telegram Bot**
def run_bot():
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("intraday_signals", intraday_signals))
    application.add_handler(CommandHandler("intraday_signals_all", intraday_signals_all))
    application.run_polling()

if __name__ == "__main__":
    run_bot()
