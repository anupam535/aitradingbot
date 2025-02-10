import os  # For environment variables
import telegram  # For Telegram bot
from telegram import Update  # For Telegram updates
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes  # Telegram bot components
from dotenv import load_dotenv  # For loading environment variables from .env file
import alpaca_trade_api as tradeapi  # For Alpaca API
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
# The calculate_rsi function.
# ... other imports (if you have them)

# --- 1. Load Environment Variables ---
load_dotenv()  # Loads the .env file (if it's in the same directory)

# --- 2. Get API Tokens and Keys from Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Get Telegram token
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")  # Get Alpaca API key
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")  # Get Alpaca secret key
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Add this line
# --- 3. Error Handling (Check if variables are set) ---
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set!")
if not ALPACA_API_KEY:
    raise ValueError("ALPACA_API_KEY environment variable not set!")
if not ALPACA_SECRET_KEY:
    raise ValueError("ALPACA_SECRET_KEY environment variable not set!")

# --- 4. Alpaca API Initialization (after checks) ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, "https://paper-api.alpaca.markets")  # Or your base URL


# Stock Symbols (Indian Stocks - Up to 50+ for intraday)
STOCK_SYMBOLS_INTRADAY = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "HDFC.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "ADANIENT.NS",
    "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "ITC.NS",
    "HCLTECH.NS", "WIPRO.NS", "NESTLEIND.NS", "TITAN.NS", "BAJAJAUTO.NS",
    "ULTRACEMCO.NS", "TECHM.NS", "M&M.NS", "SUNPHARMA.NS", "CIPLA.NS",
    "POWERGRID.NS", "NTPC.NS", "JSWSTEEL.NS", "BHARTIARTL.NS", "GRASIM.NS",
    "ONGC.NS", "TATAMOTORS.NS", "UPL.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS",
    "BPCL.NS", "IOC.NS", "HINDALCO.NS", "COALINDIA.NS", "TATAPOWER.NS",
    "ADANIPORTS.NS", "JSWENERGY.NS", "PIDILITIND.NS", "SRF.NS", "DIVISLAB.NS",
    "LUPIN.NS", "DRREDDY.NS", "PEL.NS", "DLF.NS", "GODREJCP.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS", "MRF.NS", "AMBUJACEM.NS", "ACC.NS",
    "SHREECEM.NS", "BERGEPAINT.NS", "CROMPTON.NS", "POLYCAB.NS", "HAL.NS",
    "BEL.NS", "GAIL.NS", "MGL.NS", "PIIND.NS", "SRTRANSFIN.NS", "CHOLAFIN.NS",
    "LTI.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRMF.NS"  # Added More
]

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


# Function to fetch historical data from Alpaca (for intraday)
def fetch_intraday_data(symbol, timeframe="1Min", limit=200): # Increased Limit for Intraday
    try:
        barset = api.get_barset(symbol, timeframe, limit=limit)
        df = barset.df
        df = df.reset_index()
        df = df.rename(columns={'time': 'Date'})
        df = df.set_index('Date')
        return df
    except Exception as e:
        print(f"Error fetching intraday data for {symbol}: {e}")
        return None

# Function to train the model (for intraday - adapt features)
def train_intraday_model(data):
    if data is None or data.empty:
        return None
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window=window).mean()
    ema_down = down.rolling(window=window).mean()
    rsi = 100 - (100 / (1 + (ema_up / ema_down)))
    return rsi
    
    # Feature Engineering (Intraday - Examples)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()  # Shorter SMA
    data['SMA_15'] = data['Close'].rolling(window=15).mean() # Shorter SMA
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    # Target Variable (Intraday - Example - short term price movement)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # Next bar close > current close

    data.dropna(inplace=True)
    X = data[['SMA_5', 'SMA_15', 'Volatility','RSI']]  # Intraday Features
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Or another suitable model
    model.fit(X_train, y_train)
    return model

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window=window).mean()
    ema_down = down.rolling(window=window).mean()
    rsi = 100 - (100 / (1 + (ema_up / ema_down)))
    return rsi

def create_features(data): # function where u are using it.
  data["RSI"] = calculate_rsi(data["Close"],window=14)
  return data
    
# Function to get intraday trading signals
def get_intraday_signals(symbol, model):
    data = fetch_intraday_data(symbol, timeframe="1Min", limit=200) # Recent intraday data
    if data is None or data.empty or model is None:
        return None

    # ... (Feature engineering same as in train_intraday_model)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()  # Shorter SMA
    data['SMA_15'] = data['Close'].rolling(window=15).mean() # Shorter SMA
    data['Volatility'] = data['Close'].rolling(window=10).std()
    
    data.dropna(inplace=True)
    X_recent = data[['SMA_5', 'SMA_15', 'Volatility','RSI']].tail(1)

    if X_recent.empty:
        return None

    prediction = model.predict(X_recent)[0]
    return prediction

# Telegram Bot Handlers (modified for multiple stocks)

def intraday_signals_all(update, context): # New command
    for symbol in STOCK_SYMBOLS_INTRADAY:  # Loop through stocks
        try:
            data = fetch_intraday_data(symbol)
            if data is None or data.empty:
                bot.send_message(chat_id=update.effective_chat.id, text=f"Could not retrieve intraday data for {symbol}.")
                continue # Go to the next stock

            model = train_intraday_model(data)
            if model is None:
                bot.send_message(chat_id=update.effective_chat.id, text=f"Could not train intraday model for {symbol}.")
                continue

            signal = get_intraday_signals(symbol, model)
            if signal is None:
                bot.send_message(chat_id=update.effective_chat.id, text=f"Could not generate intraday signal for {symbol}.")
                continue

            signal_text = f"Intraday Signal for {symbol}: "
            signal_text += "Buy" if signal == 1 else "Sell/Hold"
            bot.send_message(chat_id=update.effective_chat.id, text=signal_text)

        except Exception as e: # Catch any errors during processing
            bot.send_message(chat_id=update.effective_chat.id, text=f"Error processing {symbol}: {e}")


def intraday_signals(update, context): # Existing single stock command
    symbol = context.args[0] if context.args else None
    if not symbol:
        update.bot.send_message(chat_id=update.effective_chat.id, text="Please provide a stock symbol (e.g., /intraday_signals TCS.NS)")
        return
    data = fetch_intraday_data(symbol)
    if data is None or data.empty:
        update.bot.send_message(chat_id=update.effective_chat.id, text=f"Could not retrieve intraday data for {symbol}.")
        return

    model = train_intraday_model(data)
    if model is None:
        update.bot.send_message(chat_id=update.effective_chat.id, text=f"Could not train intraday model for {symbol}.")
        return

    signal = get_intraday_signals(symbol, model)
    if signal is None:
        update.bot.send_message(chat_id=update.effective_chat.id, text=f"Could not generate intraday signal for {symbol}.")
        return

    signal_text = f"Intraday Signal for {symbol}: "
    signal_text += "Buy" if signal == 1 else "Sell/Hold"
    update.bot.send_message(chat_id=update.effective_chat.id, text=signal_text)

# Define the start function BEFORE run_bot()
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to the AI Trading Bot! Use /analyze or /signals to get started.")
async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing the market... (This function needs implementation)")
    
def run_bot():
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # Get token from environment
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN environment variable not set!")
        return  # Exit if token is missing

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))  
    application.add_handler(CommandHandler("analyze", analyze))  
    application.add_handler(CommandHandler("signals", signals))  
    application.add_handler(CommandHandler("intraday_signals", intraday_signals))  
    application.add_handler(CommandHandler("intraday_signals_all", intraday_signals_all))  

    application.run_polling()

if __name__ == "__main__":
    run_bot()
