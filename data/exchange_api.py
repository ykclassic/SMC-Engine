import ccxt
import pandas as pd
import logging

class ExchangeInterface:
    def __init__(self, config):
        self.exchange_id = config['exchanges'][0]['name'] # Uses first enabled exchange
        self.exchange = getattr(ccxt, self.exchange_id)({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
        })
        self.logger = logging.getLogger(__name__)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100):
        """Fetches historical data and returns a cleaned DataFrame."""
        try:
            self.logger.info(f"Fetching {limit} candles for {symbol} on {timeframe}...")
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure numeric types for calculations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from {self.exchange_id}: {e}")
            return pd.DataFrame()
