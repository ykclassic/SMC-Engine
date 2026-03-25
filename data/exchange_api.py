import ccxt
import pandas as pd
import logging

class ExchangeInterface:
    def __init__(self, config):
        self.logger = logging.getLogger("Nexus-ExchangeAPI")
        
        # Mapping config/env to CCXT requirements
        # Note: 'password' is the CCXT key name for Bitget's API Passphrase
        try:
            self.exchange = ccxt.bitget({
                'apiKey': config.get('api_key'),
                'secret': config.get('api_secret'),
                'password': config.get('passphrase'), 
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap', # Targets USDT-Margined Futures
                    'recvWindow': 5000
                }
            })
            self.logger.info("Exchange Interface initialized for Bitget.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Exchange: {e}")
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200):
        """
        Fetches historical candle data and returns a cleaned DataFrame.
        """
        try:
            # CCXT returns: [timestamp, open, high, low, close, volume]
            raw_data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                raw_data, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime and numeric types to float
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)
            
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame() # Return empty DF to prevent engine crash

    def get_account_balance(self):
        """Returns the total USDT balance available for trading."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['total'].get('USDT', 0))
        except Exception as e:
            self.logger.error(f"Could not fetch balance: {e}")
            return 0.0
