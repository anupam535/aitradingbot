import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import telegram
import time
import logging

# Configuration
TELEGRAM_TOKEN = "8060691368:AAHTq4sArzPu5CK3W7ZNLH_bJv6Ri0uiEeM"  # Replace with your Telegram bot token
CHAT_ID = "5956248751"  # Replace with your Telegram chat ID

# 50 Indian stocks (as previously selected)
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "ITC.NS",
    "KOTAKBANK.NS", "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "GRASIM.NS", "BAJAJFINSV.NS", "NESTLEIND.NS",
    "DRREDDY.NS", "CIPLA.NS", "TECHM.NS", "DIVISLAB.NS", "BRITANNIA.NS",
    "M&M.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "SHRIRAMFIN.NS", "GODREJCP.NS",
    "PIDILITIND.NS", "HAVELLS.NS", "DABUR.NS", "BPCL.NS", "IOC.NS",
    "HINDALCO.NS", "COALINDIA.NS", "VEDL.NS", "GAIL.NS", "INDUSINDBK.NS"
]

# Logging setup for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Telegram bot
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# Fetch live stock data (1-minute interval for real-time)
def fetch_live_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d", interval="1m")  # 1-minute live data
        if hist.empty:
            logging.warning(f"No data for {symbol}")
            return None
        return hist
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# Technical Indicators
def calculate_indicators(df):
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

# Pre-trained LSTM model (simplified for speed)
def load_or_train_model(symbol, hist_data):
    model_path = f"{symbol}_lstm_model.h5"
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hist_data['Close'].values.reshape(-1, 1))
    
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Loaded pre-trained model for {symbol}")
    except:
        logging.info(f"Training new model for {symbol}")
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(1 if scaled_data[i] > scaled_data[i-1] else 0)
        X, y = np.array(X), np.array(y)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        model.save(model_path)
    return model, scaler

# Real-time stock analysis
def analyze_stock(symbol):
    live_data = fetch_live_data(symbol)
    if live_data is None or len(live_data) < 60:
        return None
    
    live_data = calculate_indicators(live_data)
    hist_data = yf.Ticker(symbol).history(period="5d")  # 5 days for initial training
    
    # Load or train model
    model, scaler = load_or_train_model(symbol, hist_data)
    last_60 = scaler.transform(live_data['Close'].values[-60:].reshape(-1, 1))
    pred = model.predict(np.array([last_60]), verbose=0)[0][0]
    trend = "Bullish" if pred > 0.5 else "Bearish"
    
    # Buy/Sell Signals
    current_price = live_data['Close'][-1]
    buy_price = live_data['BB_Low'][-1] if trend == "Bullish" and current_price <= live_data['BB_Low'][-1] else None
    sell_price = live_data['BB_High'][-1] if trend == "Bearish" and current_price >= live_data['BB_High'][-1] else None
    
    return {
        "symbol": symbol,
        "trend": trend,
        "current_price": current_price,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "rsi": live_data['RSI'][-1],
        "macd_diff": live_data['MACD'][-1] - live_data['MACD_Signal'][-1]
    }

# Send Telegram alert
def send_trade_alert(result):
    if result['buy_price']:
        msg = (f"🚨 BUY ALERT 🚨\n"
               f"Stock: {result['symbol']}\n"
               f"Trend: {result['trend']}\n"
               f"Buy Price: ₹{result['buy_price']:.2f}\n"
               f"Current Price: ₹{result['current_price']:.2f}\n"
               f"RSI: {result['rsi']:.2f}")
        bot.send_message(chat_id=CHAT_ID, text=msg)
        logging.info(f"Sent BUY alert for {result['symbol']}")
    elif result['sell_price']:
        msg = (f"🚨 SELL ALERT 🚨\n"
               f"Stock: {result['symbol']}\n"
               f"Trend: {result['trend']}\n"
               f"Sell Price: ₹{result['sell_price']:.2f}\n"
               f"Current Price: ₹{result['current_price']:.2f}\n"
               f"RSI: {result['rsi']:.2f}")
        bot.send_message(chat_id=CHAT_ID, text=msg)
        logging.info(f"Sent SELL alert for {result['symbol']}")

# Main real-time loop
def main():
    logging.info("Starting real-time stock market analysis...")
    while True:
        current_time = time.localtime()
        # Run only during Indian market hours (9:15 AM - 3:30 PM IST)
        if 9 <= current_time.tm_hour < 15 or (current_time.tm_hour == 15 and current_time.tm_min <= 30):
            for stock in STOCKS:
                try:
                    result = analyze_stock(stock)
                    if result and (result['buy_price'] or result['sell_price']):
                        send_trade_alert(result)
                except Exception as e:
                    logging.error(f"Error analyzing {stock}: {e}")
                time.sleep(1)  # Small delay between stocks to avoid API rate limits
            time.sleep(60)  # Check every minute
        else:
            logging.info("Market closed. Waiting for next trading session...")
            time.sleep(300)  # Sleep 5 minutes during off-hours

if __name__ == "__main__":
    main()
