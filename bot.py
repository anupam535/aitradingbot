import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackContext, MessageHandler, filters, CallbackQueryHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca API configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets/v2'  # Use paper trading for testing

api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# Telegram Bot Token
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# Global variables
model = None
alerts_enabled = True
high_profit_stocks = []

# Function to fetch historical data from Alpaca API
def get_historical_data(symbol, timeframe='1Day', limit=100):
    try:
        barset = api.get_bars(symbol, timeframe, limit=limit).df
        # Flatten the MultiIndex columns into a single-level index
        barset.columns = [col[1] if col[1] != '' else col[0] for col in barset.columns]
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
        await application.bot.send_message(chat_id=ADMIN_ID, text=message)

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
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Welcome to the AI Trading Bot! Use /stock or /crypto to analyze markets.")

async def stock_analysis(update: Update, context: CallbackContext):
    symbol = 'AAPL'  # Example stock
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        await update.message.reply_text(f"Stock Analysis for {symbol}: Signal -> {signal}")
    else:
        await update.message.reply_text("Failed to fetch stock data.")

async def crypto_analysis(update: Update, context: CallbackContext):
    symbol = 'BTCUSD'  # Example crypto
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        await update.message.reply_text(f"Crypto Analysis for {symbol}: Signal -> {signal}")
    else:
        await update.message.reply_text("Failed to fetch crypto data.")

async def trading_signal(update: Update, context: CallbackContext):
    symbol = 'AAPL'  # Example asset
    data = get_historical_data(symbol)
    if data is not None:
        signal = predict_signal(data)
        await update.message.reply_text(f"Trading Signal for {symbol}: {signal}")
    else:
        await update.message.reply_text("Failed to generate trading signal.")

async def high_profit_stocks_command(update: Update, context: CallbackContext):
    if not high_profit_stocks:
        find_high_profit_stocks()
    if high_profit_stocks:
        response = "High-Profit Stocks Today:\n"
        for stock, return_rate in high_profit_stocks:
            response += f"{stock}: {return_rate * 100:.2f}%\n"
        await update.message.reply_text(response)
    else:
        await update.message.reply_text("No high-profit stocks found today.")

async def toggle_alerts(update: Update, context: CallbackContext):
    global alerts_enabled
    alerts_enabled = not alerts_enabled
    status = "enabled" if alerts_enabled else "disabled"
    await update.message.reply_text(f"Real-time alerts are now {status}.")

async def stock_info(update: Update, context: CallbackContext):
    query = update.callback_query
    if query is None:
        return
    symbol = query.data
    data = get_historical_data(symbol)
    if data is not None:
        close_price = data['close'].iloc[-1]
        open_price = data['open'].iloc[-1]
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        volume = data['volume'].iloc[-1]
        signal = predict_signal(data)
        await query.edit_message_text(
            f"Stock Info for {symbol}:\n"
            f"Close Price: {close_price:.2f}\n"
            f"Open Price: {open_price:.2f}\n"
            f"High: {high:.2f}\n"
            f"Low: {low:.2f}\n"
            f"Volume: {volume}\n"
            f"Signal: {signal}"
        )
    else:
        await query.edit_message_text(f"Failed to fetch stock data for {symbol}.")

# Add command handlers
application = Application.builder().token(TELEGRAM_TOKEN).build()

application.add_handler(CommandHandler('start', start))
application.add_handler(CommandHandler('stock', stock_analysis))
application.add_handler(CommandHandler('crypto', crypto_analysis))
application.add_handler(CommandHandler('signal', trading_signal))
application.add_handler(CommandHandler('highprofit', high_profit_stocks_command))
application.add_handler(CommandHandler('togglealerts', toggle_alerts))

# Inline keyboard for stock info
stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
keyboard = [[InlineKeyboardButton(symbol, callback_data=symbol)] for symbol in stock_symbols]
reply_markup = InlineKeyboardMarkup(keyboard)

application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: update.message.reply_text("Please use the available commands.", reply_markup=reply_markup)))
application.add_handler(CallbackQueryHandler(stock_info))

# Train the model at startup
train_model()

# Start the Telegram bot
print("Telegram bot started...")
application.run_polling()

# Start WebSocket stream for real-time alerts
conn = Stream(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url=BASE_URL)
conn.subscribe_bars(on_bar, *stock_symbols)  # Subscribe to stocks
conn.run()
