import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
    "HINDUNILVR.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "AXISBANK.NS", "BAJFINANCE.NS",
    "LT.NS", "ITC.NS", "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS",
    "WIPRO.NS", "MARUTI.NS", "ULTRACEMCO.NS", "TITAN.NS", "ADANIENT.NS",
    "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS", "IOC.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "TATASTEEL.NS", "GRASIM.NS", "BAJAJFINSV.NS",
    "BRITANNIA.NS", "DIVISLAB.NS", "SHREECEM.NS", "HEROMOTOCO.NS", "UPL.NS",
    "NESTLEIND.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "TECHM.NS",
    "ADANIPORTS.NS", "HDFCLIFE.NS", "SBILIFE.NS", "INDUSINDBK.NS", "M&M.NS",
    "APOLLOHOSP.NS", "TATAMOTORS.NS", "PIDILITIND.NS", "HDFCAMC.NS", "BANDHANBNK.NS",
    "AUROPHARMA.NS", "DLF.NS", "BERGEPAINT.NS", "GAIL.NS", "AMBUJACEM.NS"
]

# Initialize Alpaca API
alpaca_api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Function to fetch historical stock data
def fetch_stock_data(stock_symbol, period='5y'):
    try:
        stock_data = yf.download(stock_symbol, period=period)
        if stock_data.empty:
            print(f"No data found for {stock_symbol}")
            return None
        stock_data['Return'] = stock_data['Close'].pct_change()
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
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        stock_data['BB_Middle'] = stock_data['MA_20']
        stock_data['BB_Upper'] = stock_data['BB_Middle'] + 2 * stock_data['Close'].rolling(window=20).std()
        stock_data['BB_Lower'] = stock_data['BB_Middle'] - 2 * stock_data['Close'].rolling(window=20).std()

        stock_data.dropna(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None

# Function to generate short-term signals
def generate_short_term_signals(stock_data):
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
        print(f"Error generating short-term signals: {e}")
        return None

# Function to predict long-term returns
def predict_long_term_returns(stock_symbol, periods=['3m', '6m', '1y', '5y']):
    try:
        results = {}
        for period in periods:
            stock_data = fetch_stock_data(stock_symbol, period=period)
            if stock_data is not None:
                features = ['Open', 'High', 'Low', 'Close', 'Volume']
                X = stock_data[features]
                y = stock_data['Close'].shift(-1)  # Predict next day's close
                X = X[:-1]  # Remove last row (no target value)
                y = y[:-1]

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train Random Forest Regressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Predict future price
                future_price = model.predict([X.iloc[-1]])[0]
                current_price = stock_data['Close'].iloc[-1]
                expected_return = (future_price - current_price) / current_price * 100
                results[period] = expected_return

        return results
    except Exception as e:
        print(f"Error predicting long-term returns for {stock_symbol}: {e}")
        return None

# Telegram Bot Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello Mikey Sir, welcome to the AI-Powered Trading Bot! Use /shortterm for intraday signals and /longterm for long-term investment recommendations.")

async def get_short_term_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for short-term signals... Please wait.")
    results = []

    for stock_symbol in STOCKS:
        stock_data = fetch_stock_data(stock_symbol)
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
        await update.message.reply_text("No short-term signals found for today.")

async def get_long_term_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for long-term investment opportunities... Please wait.")
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
        await update.message.reply_text("No long-term recommendations found.")

# Main Function
def main():
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("shortterm", get_short_term_signals))
    application.add_handler(CommandHandler("longterm", get_long_term_recommendations))

    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()
