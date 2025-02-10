import os
import time
import logging
import ccxt
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize exchanges and telegram bot
exchange = ccxt.kraken({
    'apiKey': os.getenv('CCXT_API_KEY'),
    'secret': os.getenv('CCXT_SECRET'),
})
telegram_bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')

# Constants
TIMEFRAME = '1h'
HISTORICAL_DATA_LIMIT = 500
MODEL_UPDATE_INTERVAL = 3600  # 1 hour

# Load historical data
def fetch_historical_data(symbol):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=HISTORICAL_DATA_LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Calculate technical indicators
def calculate_indicators(df):
    df['rsi'] = momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = trend.MACD(df['close']).macd()
    df['macd_signal'] = trend.MACD(df['close']).macd_signal()
    df['sma'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema'] = trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['bollinger_hband'] = volatility.BollingerBands(df['close']).bollinger_hband()
    df['bollinger_lband'] = volatility.BollingerBands(df['close']).bollinger_lband()
    df['volume_ma'] = volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    return df

# Train machine learning model
def train_model(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df.drop(['timestamp', 'target'], axis=1).dropna().values
    labels = df['target'].dropna().values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    return model, scaler

# Predict using trained model
def predict(model, scaler, df):
    features = df.drop(['timestamp'], axis=1).values
    features = scaler.transform(features)
    predictions = model.predict(features)
    return predictions

# Rank stocks based on predicted returns
def rank_stocks(predictions, symbols):
    ranked_stocks = sorted(zip(symbols, predictions), key=lambda x: x[1], reverse=True)
    return ranked_stocks

# Send message to telegram
def send_telegram_message(message):
    telegram_bot.send_message(chat_id=telegram_chat_id, text=message)

# Main function
def main():
    symbols = ['XBT/USD', 'ETH/USD', 'ADA/USD']  # Example symbols for Kraken
    model_cache = {}
    scaler_cache = {}

    while True:
        try:
            for symbol in symbols:
                df = fetch_historical_data(symbol)
                df = calculate_indicators(df)

                if symbol not in model_cache or time.time() - model_cache[symbol]['last_updated'] > MODEL_UPDATE_INTERVAL:
                    model, scaler = train_model(df)
                    model_cache[symbol] = {'model': model, 'last_updated': time.time()}
                    scaler_cache[symbol] = scaler

                predictions = predict(model_cache[symbol]['model'], scaler_cache[symbol], df)
                ranked_stocks = rank_stocks(predictions, symbols)

                top_stocks = ranked_stocks[:10]
                message = f"Top 10 stocks:\n{top_stocks}"
                send_telegram_message(message)

        except Exception as e:
            logger.error(f"Error: {e}")
            send_telegram_message(f"An error occurred: {e}")

        time.sleep(60)  # Check every minute

if __name__ == '__main__':
    main()
