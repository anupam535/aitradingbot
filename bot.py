import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import alpaca_trade_api as tradeapi

# Load environment variables from .env file
load_dotenv()

# Configuration (Loaded from .env)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

# List of Indian stocks (NSE symbols)
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "AXISBANK.NS", "BAJFINANCE.NS"
]

# Initialize Alpaca API
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Function to fetch historical stock data
def fetch_stock_data(stock_symbol, period='1y'):
    try:
        stock_data = yf.download(stock_symbol, period=period)
        stock_data['Return'] = stock_data['Close'].pct_change()
        stock_data['Target'] = (stock_data['Return'].shift(-1) > 0).astype(int)
        stock_data.dropna(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None

# Function to calculate technical indicators
def calculate_indicators(stock_data):
    try:
        # Moving Averages
        stock_data['MA_5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()

        # RSI (Relative Strength Index)
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        stock_data['BB_Upper'] = stock_data['MA_20'] + 2 * stock_data['Close'].rolling(window=20).std()
        stock_data['BB_Lower'] = stock_data['MA_20'] - 2 * stock_data['Close'].rolling(window=20).std()

        stock_data.dropna(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

# Function to generate buy/sell/stop-loss points
def generate_signals(stock_data):
    try:
        buy_points = []
        sell_points = []
        stop_loss_points = []

        for i in range(1, len(stock_data)):
            current_row = stock_data.iloc[i]
            prev_row = stock_data.iloc[i - 1]

            # Buy Signal: MA_5 crosses above MA_20 and RSI < 30
            if (current_row['MA_5'] > current_row['MA_20']) and (prev_row['MA_5'] <= prev_row['MA_20']) and (current_row['RSI'] < 30):
                buy_points.append(current_row['Close'])

            # Sell Signal: MA_5 crosses below MA_20 or RSI > 70
            if (current_row['MA_5'] < current_row['MA_20']) and (prev_row['MA_5'] >= prev_row['MA_20']) or (current_row['RSI'] > 70):
                sell_points.append(current_row['Close'])

            # Stop-Loss: Below BB_Lower
            if current_row['Close'] < current_row['BB_Lower']:
                stop_loss_points.append(current_row['Close'])

        return {
            'buy_points': buy_points,
            'sell_points': sell_points,
            'stop_loss_points': stop_loss_points
        }
    except Exception as e:
        print(f"Error generating signals: {e}")
        return None

# Telegram Bot Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the AI-Powered Trading Bot! Use /signals for buy/sell alerts.")

async def get_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market... Please wait.")
    results = []

    for stock_symbol in STOCKS:
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data is not None:
            stock_data = calculate_indicators(stock_data)
            signals = generate_signals(stock_data)

            if signals:
                buy_points = signals['buy_points']
                sell_points = signals['sell_points']
                stop_loss_points = signals['stop_loss_points']

                result = (
                    f"{stock_symbol}:\n"
                    f"Buy Points: {buy_points}\n"
                    f"Sell Points: {sell_points}\n"
                    f"Stop-Loss Points: {stop_loss_points}\n"
                )
                results.append(result)

    if results:
        await update.message.reply_text("\n".join(results))
    else:
        await update.message.reply_text("No signals found for today.")

# Main Function
def main():
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("signals", get_signals))

    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()
