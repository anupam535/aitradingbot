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

# Function to predict long-term returns
def predict_long_term_returns(stock_symbol, periods=['3m', '6m', '1y', '5y']):
    try:
        results = {}
        for period in periods:
            stock_data = fetch_stock_data(stock_symbol, period=period)
            if stock_data is None or len(stock_data) < 50:  # Ensure sufficient data
                print(f"Not enough data for {stock_symbol} ({period})")
                continue

            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            X = stock_data[features]
            y = stock_data['Close'].shift(-1)  # Predict next day's close
            X = X[:-1]  # Remove last row (no target value)
            y = y[:-1].values.ravel()  # Ensure y is a 1D array

            if len(X) < 50 or len(y) < 50:  # Ensure sufficient data for training
                print(f"Not enough data for {stock_symbol} ({period}) after preprocessing")
                continue

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Predict future price
            future_price = model.predict([X.iloc[-1]])[0]
            current_price = stock_data['Close'].iloc[-1]
            expected_return = (future_price - current_price) / current_price * 100
            results[period] = round(expected_return, 2)  # Store scalar value with 2 decimal places

        return results
    except Exception as e:
        print(f"Error predicting long-term returns for {stock_symbol}: {e}")
        return None

# Error handler for Telegram
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")

# Telegram Bot Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello Mikey Sir, Welcome to the AI-Powered Trading Bot! Use /intraday for intraday signals and /longterm for long-term investment recommendations.")

async def get_intraday_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for intraday signals... Please wait.")
    results = []

    for stock_symbol in STOCKS:
        stock_data = fetch_stock_data(stock_symbol, period='1mo')  # Use 1 month of data for intraday
        if stock_data is not None:
            stock_data = calculate_indicators(stock_data)
            if stock_data is not None:
                signals = generate_intraday_signals(stock_data)

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
        await update.message.reply_text("No intraday signals found for today.")

async def get_long_term_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing market for long-term investment opportunities... Please wait.")
    results = []

    for stock_symbol in STOCKS:
        predictions = predict_long_term_returns(stock_symbol)
        if predictions:
            result = f"{stock_symbol}:\n"
            for period, return_rate in predictions.items():
                result += f"Expected Return ({period}): {return_rate}%\n"
            results.append(result)
        else:
            results.append(f"No long-term predictions available for {stock_symbol}.")

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
    application.add_handler(CommandHandler("intraday", get_intraday_signals))
    application.add_handler(CommandHandler("longterm", get_long_term_recommendations))

    # Register Error Handler
    application.add_error_handler(error_handler)

    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()
