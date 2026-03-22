import pandas as pd
import numpy as np

class MarketStructure:
    def __init__(self, config):
        self.lookback = config['market_structure']['lookback_period']
        self.confirm_type = config['market_structure']['structure_type']

    def detect_fractals(self, df: pd.DataFrame):
        """
        Identifies Swing Highs and Swing Lows based on a lookback window.
        A fractal high is a peak higher than 'n' candles to its left and right.
        """
        df['is_high'] = df['high'].rolling(window=self.lookback*2+1, center=True).apply(
            lambda x: 1 if x.iloc[self.lookback] == x.max() else 0, raw=False
        )
        df['is_low'] = df['low'].rolling(window=self.lookback*2+1, center=True).apply(
            lambda x: 1 if x.iloc[self.lookback] == x.min() else 0, raw=False
        )
        return df

    def get_structure_points(self, df: pd.DataFrame):
        """
        Labels points as HH, HL, LH, LL by comparing current fractal to previous.
        """
        df = self.detect_fractals(df)
        highs = df[df['is_high'] == 1]['high']
        lows = df[df['is_low'] == 1]['low']
        
        # Logic to determine HH/LH
        df['label'] = ""
        last_high = None
        for idx, val in highs.items():
            if last_high is None:
                last_high = val
                continue
            df.at[idx, 'label'] = "HH" if val > last_high else "LH"
            last_high = val
            
        # Logic to determine HL/LL
        last_low = None
        for idx, val in lows.items():
            if last_low is None:
                last_low = val
                continue
            df.at[idx, 'label'] = "HL" if val > last_low else "LL"
            last_low = val
            
        return df

    def detect_bos(self, df: pd.DataFrame):
        """
        Detects a Break of Structure (BOS).
        Bullish BOS: Price closes above the previous Higher High.
        Bearish BOS: Price closes below the previous Lower Low.
        """
        df['bos'] = False
        last_hh = None
        last_ll = None

        for i in range(1, len(df)):
            # Update targets when a fractal is confirmed
            if df.at[i, 'label'] == "HH":
                last_hh = df.at[i, 'high']
            if df.at[i, 'label'] == "LL":
                last_ll = df.at[i, 'low']

            # Check for Break
            if last_hh and df.at[i, 'close'] > last_hh:
                df.at[i, 'bos'] = "BULLISH_BOS"
                last_hh = None # Reset until next HH is formed
            
            if last_ll and df.at[i, 'close'] < last_ll:
                df.at[i, 'bos'] = "BEARISH_BOS"
                last_ll = None # Reset until next LL is formed
                
        return df

# Example Usage logic for main.py integration:
# engine = MarketStructure(config)
# df_with_structure = engine.detect_bos(engine.get_structure_points(raw_data))
