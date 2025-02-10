import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca API configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# Telegram Bot Token
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_ID = int(os.getenv('ADMIN_ID'))  # Replace with your Telegram user ID

# Initialize Telegram bot
updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# Global variables
model = None
alerts_enabled = True
high_profit_stocks = []

# Function to fetch historical data from Alpaca API
def get_historical_data(symbol, timeframe='1Day', limit=100):
    try:
        barset = api.get_bars(symbol, timeframe, limit=limit).df
        return barset
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to prepare data for ML model
def prepare_data(data):
    data['Return'] = data['close'].pct_change()
    data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
    data.dropna(inplace=True)
    features = ['open', 'high', 'low', 'close', 'volume']
    X = data[features]
    y = data['Target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train the ML model
def train_model():
    global model
    stock_data = get_historical_data('AAPL')  # Example stock
    if stock_data is not None:
        X_train, X_test, y_train, y_test = prepare_data(stock_data)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
    else:
        print("Failed to train model due to insufficient data.")

# Function to predict buy/sell points
def predict_signal(data):
    if model is None:
        return "Model not trained yet."
    features = ['open', 'high', 'low', 'close', 'volume']
    prediction = model.predict(data[features].tail(1))
    return "Buy" if prediction[0] == 1 else "Sell"

# Function to calculate stop-loss and take-profit
def calculate_stop_loss_take_profit(current_price, signal):
    if signal == "Buy":
        stop_loss = current_price * 0.98  # 2% below current price
        take_profit = current_price * 1.05  # 5% above current price
    else:
        stop_loss = current_price * 1.02  # 2% above current price
        take_profit = current_price * 0.95  # 5% below current price
    return stop_loss, take_profit

# WebSocket handler for real-time alerts
async def on_bar(conn, bar):
    global alerts_enabled
    if not alerts_enabled:
        return
    symbol = bar.S
    current_price = bar.c
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        stop_loss, take_profit = calculate_stop_loss_take_profit(current_price, signal)
        message = f"Alert: {symbol} -> Signal: {signal}, Price: {current_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}"
        dispatcher.bot.send_message(chat_id=ADMIN_ID, text=message)

# Function to find high-profit stocks for the day
def find_high_profit_stocks():
    global high_profit_stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Example stocks
    high_profit_stocks = []
    for symbol in symbols:
        data = get_historical_data(symbol)
        if data is not None:
            daily_return = data['close'].pct_change().iloc[-1]
            if daily_return > 0.01:  # Filter stocks with >1% daily return
                high_profit_stocks.append((symbol, daily_return))
    high_profit_stocks.sort(key=lambda x: x[1], reverse=True)  # Sort by return
    print("High-profit stocks identified.")

# Telegram command handlers
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Welcome to the AI Trading Bot! Use /stock or /crypto to analyze markets.")

def stock_analysis(update: Update, context: CallbackContext):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text("You do not have permission to use this command.")
        return
    symbol = 'AAPL'  # Example stock
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        update.message.reply_text(f"Stock Analysis for {symbol}: Signal -> {signal}")
    else:
        update.message.reply_text("Failed to fetch stock data.")

def crypto_analysis(update: Update, context: CallbackContext):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text("You do not have permission to use this command.")
        return
    symbol = 'BTCUSD'  # Example crypto
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        update.message.reply_text(f"Crypto Analysis for {symbol}: Signal -> {signal}")
    else:
        update.message.reply_text("Failed to fetch crypto data.")

def trading_signal(update: Update, context: CallbackContext):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text("You do not have permission to use this command.")
        return
    symbol = 'AAPL'  # Example asset
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        update.message.reply_text(f"Trading Signal for {symbol}: {signal}")
    else:
        update.message.reply_text("Failed to generate trading signal.")

def high_profit_stocks_command(update: Update, context: CallbackContext):
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text("You do not have permission to use this command.")
        return
    if not high_profit_stocks:
        find_high_profit_stocks()
    if high_profit_stocks:
        response = "High-Profit Stocks Today:\n"
        for stock, return_rate in high_profit_stocks:
            response += f"{stock}: {return_rate * 100:.2f}%\n"
        update.message.reply_text(response)
    else:
        update.message.reply_text("No high-profit stocks found today.")

def toggle_alerts(update: Update, context: CallbackContext):
    global alerts_enabled
    if update.message.from_user.id != ADMIN_ID:
        update.message.reply_text("You do not have permission to use this command.")
        return
    alerts_enabled = not alerts_enabled
    status = "enabled" if alerts_enabled else "disabled"
    update.message.reply_text(f"Real-time alerts are now {status}.")

# Add command handlers
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('stock', stock_analysis))
dispatcher.add_handler(CommandHandler('crypto', crypto_analysis))
dispatcher.add_handler(CommandHandler('signal', trading_signal))
dispatcher.add_handler(CommandHandler('highprofit', high_profit_stocks_command))
dispatcher.add_handler(CommandHandler('togglealerts', toggle_alerts))

# Main function
if __name__ == '__main__':
    # Train the model at startup
    train_model()

    # Start the Telegram bot
    updater.start_polling()
    print("Telegram bot started...")

    # Start WebSocket stream for real-time alerts
    conn = Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)
    conn.subscribe_bars(on_bar, *['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])  # Subscribe to stocks
    conn.run()

    # Keep the bot running
    updater.idle()
