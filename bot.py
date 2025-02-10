import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# List of Cryptocurrencies (Yahoo Finance symbols)
CRYPTOS = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"]

# Initialize Alpaca API
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Function to fetch historical stock/crypto data
def fetch_data(symbol, period='5y', interval='1d'):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            print(f"No data found for {symbol}")
            return None
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to calculate technical indicators
def calculate_indicators(data):
    try:
        # Moving Averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        data['BB_Middle'] = data['MA_20']
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

# Function to generate short-term buy/sell/stop-loss signals
def generate_short_term_signals(data):
    try:
        buy_points = []
        sell_points = []
        stop_loss_points = []

        for i in range(1, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i - 1]

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
        print(f"Error generating short-term signals: {e}")
        return None

# Function to predict long-term returns
def predict_long_term_returns(symbol, periods=['3m', '6m', '1y', '5y']):
    try:
        results = {}
        for period in periods:
            data = fetch_data(symbol, period=period)
            if data is not None:
                features = ['Open', 'High', 'Low', 'Close', 'Volume']
                X = data[features]
                y = data['Close'].shift(-1)  # Predict next day's close
                X = X[:-1]  # Remove last row (no target value)
                y = y[:-1].values.ravel()  # Ensure y is a 1D array

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train Random Forest Regressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Predict future price
                future_price = model.predict([X.iloc[-1]])[0]
                current_price = data['Close'].iloc[-1]
                expected_return = (future_price - current_price) / current_price * 100
                results[period] = expected_return  # Store scalar value

        return results
    except Exception as e:
        print(f"Error predicting long-term returns for {symbol}: {e}")
        return None

# Error handler for Telegram
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

# Telegram Bot Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello Mikey Sir, Welcome to the AI-Powered Trading Bot! Use:\n"
        "/intraday_stock for intraday stock signals.\n"
        "/longterm_stock for long-term stock recommendations.\n"
        "/intraday_crypto for intraday crypto signals.\n"
        "/longterm_crypto for long-term crypto recommendations."
    )

async def get_intraday_stock_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for intraday stock signals... Please wait.")
    results = []

    for stock_symbol in STOCKS:
        stock_data = fetch_data(stock_symbol, period='1mo', interval='1d')  # Use 1 month of data for intraday
        if stock_data is not None:
            stock_data = calculate_indicators(stock_data)
            if stock_data is not None:
                signals = generate_short_term_signals(stock_data)

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
        await update.message.reply_text("No intraday stock signals found for today.")

async def get_long_term_stock_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for long-term stock investment opportunities... Please wait.")
    results = []

    for stock_symbol in STOCKS:
        predictions = predict_long_term_returns(stock_symbol)
        if predictions:
            result = f"{stock_symbol}:\n"
            for period, return_rate in predictions.items():
                result += f"Expected Return ({period}): {return_rate:.2f}%\n"
            results.append(result)

    if results:
        await update.message.reply_text("\n".join(results))
    else:
        await update.message.reply_text("No long-term stock recommendations found.")

async def get_intraday_crypto_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for intraday crypto signals... Please wait.")
    results = []

    for crypto_symbol in CRYPTOS:
        crypto_data = fetch_data(crypto_symbol, period='1mo', interval='1h')  # Use 1 month of hourly data for intraday
        if crypto_data is not None:
            crypto_data = calculate_indicators(crypto_data)
            if crypto_data is not None:
                signals = generate_short_term_signals(crypto_data)

                if signals:
                    buy_points = signals['buy_points']
                    sell_points = signals['sell_points']
                    stop_loss_points = signals['stop_loss_points']

                    result = (
                        f"{crypto_symbol}:\n"
                        f"Buy Points: {buy_points}\n"
                        f"Sell Points: {sell_points}\n"
                        f"Stop-Loss Points: {stop_loss_points}\n"
                    )
                    results.append(result)

    if results:
        await update.message.reply_text("\n".join(results))
    else:
        await update.message.reply_text("No intraday crypto signals found for today.")

async def get_long_term_crypto_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for long-term crypto investment opportunities... Please wait.")
    results = []

    for crypto_symbol in CRYPTOS:
        predictions = predict_long_term_returns(crypto_symbol)
        if predictions:
            result = f"{crypto_symbol}:\n"
            for period, return_rate in predictions.items():
                result += f"Expected Return ({period}): {return_rate:.2f}%\n"
            results.append(result)

    if results:
        await update.message.reply_text("\n".join(results))
    else:
        await update.message.reply_text("No long-term crypto recommendations found.")

# Main Function
def main():
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("intraday_stock", get_intraday_stock_signals))
    application.add_handler(CommandHandler("longterm_stock", get_long_term_stock_recommendations))
    application.add_handler(CommandHandler("intraday_crypto", get_intraday_crypto_signals))
    application.add_handler(CommandHandler("longterm_crypto", get_long_term_crypto_recommendations))

    # Register Error Handler
    application.add_error_handler(error_handler)

    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()
