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
import threading
import time
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
    # Add more stocks here
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

# Function to train ML model
def train_model(stock_data):
    try:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = stock_data[features]
        y = stock_data['Target']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return model, accuracy
    except Exception as e:
        print(f"Error training model: {e}")
        return None, 0

# Function to generate buy/sell signals
def generate_signals(model, stock_data):
    try:
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = stock_data[features]
        predictions = model.predict(X)
        stock_data['Signal'] = predictions
        return stock_data
    except Exception as e:
        print(f"Error generating signals: {e}")
        return None

# Function to execute trades via Alpaca API
def execute_trades(signals):
    try:
        account = alpaca_api.get_account()
        cash = float(account.cash)
        portfolio_value = float(account.equity)

        print(f"Available Cash: ${cash:.2f}, Portfolio Value: ${portfolio_value:.2f}")

        for stock_symbol, signal in signals.items():
            if signal == 1:  # Buy signal
                stock_price = yf.Ticker(stock_symbol).history(period='1d')['Close'].iloc[-1]
                qty = int(cash / stock_price)  # Calculate quantity based on available cash
                if qty > 0:
                    print(f"Buying {qty} shares of {stock_symbol} at ${stock_price:.2f}")
                    alpaca_api.submit_order(
                        symbol=stock_symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
            elif signal == 0:  # Sell signal
                position = alpaca_api.get_position(stock_symbol)
                qty = int(position.qty)
                if qty > 0:
                    print(f"Selling {qty} shares of {stock_symbol}")
                    alpaca_api.submit_order(
                        symbol=stock_symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
    except Exception as e:
        print(f"Error executing trades: {e}")

# Telegram Bot Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the AI-Powered Trading Bot! Use /analyze or /trade for actions.")

async def analyze_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market... Please wait.")
    results = []
    for stock_symbol in STOCKS:
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data is not None:
            model, accuracy = train_model(stock_data)
            if model is not None:
                signals = generate_signals(model, stock_data)
                last_signal = signals['Signal'].iloc[-1]
                action = "Buy" if last_signal == 1 else "Sell"
                results.append(f"{stock_symbol}: {action} (Accuracy: {accuracy * 100:.2f}%)")
    await update.message.reply_text("\n".join(results))

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Executing trades... Please wait.")
    signals = {}
    for stock_symbol in STOCKS:
        stock_data = fetch_stock_data(stock_symbol)
        if stock_data is not None:
            model, _ = train_model(stock_data)
            if model is not None:
                signals_df = generate_signals(model, stock_data)
                signals[stock_symbol] = signals_df['Signal'].iloc[-1]
    execute_trades(signals)
    await update.message.reply_text("Trades executed successfully!")

# Main Function
def main():
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_market))
    application.add_handler(CommandHandler("trade", trade))

    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()
