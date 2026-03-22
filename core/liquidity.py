import pandas as pd
import numpy as np

class LiquidityEngine:
    def __init__(self, config):
        self.threshold = config['liquidity']['eq_threshold']
        self.sweep_buffer = config['liquidity']['sweep_buffer']

    def identify_liquidity_pools(self, df: pd.DataFrame):
        """
        Finds Equal Highs and Equal Lows where price has hit a level 
        multiple times without breaking it.
        """
        df['liquidity_pool'] = None
        
        # We look for recent fractal highs/lows that are 'nearly' equal
        highs = df[df['is_high'] == 1]
        lows = df[df['is_low'] == 1]

        # Identify EQH
        for i in range(len(highs) - 1):
            price_a = highs.iloc[i]['high']
            price_b = highs.iloc[i+1]['high']
            if abs(price_a - price_b) / price_a <= self.threshold:
                idx = highs.index[i+1]
                df.at[idx, 'liquidity_pool'] = "EQH"

        # Identify EQL
        for i in range(len(lows) - 1):
            price_a = lows.iloc[i]['low']
            price_b = lows.iloc[i+1]['low']
            if abs(price_a - price_b) / price_a <= self.threshold:
                idx = lows.index[i+1]
                df.at[idx, 'liquidity_pool'] = "EQL"
        
        return df

    def detect_sweeps(self, df: pd.DataFrame):
        """
        Detects a 'Liquidity Sweep' where price wicks past an EQH/EQL 
        but fails to close above/below it, reversing immediately.
        """
        df['liquidity_sweep'] = False
        
        for i in range(1, len(df)):
            # Check for Sell-side Sweep (Price dips below EQL then bounces)
            prev_eql = df.iloc[:i][df['liquidity_pool'] == "EQL"]['low'].last_valid_index()
            if prev_eql:
                level = df.at[prev_eql, 'low']
                if df.at[i, 'low'] < level and df.at[i, 'close'] > level:
                    df.at[i, 'liquidity_sweep'] = "BULLISH_SWEEP"

            # Check for Buy-side Sweep (Price spikes above EQH then drops)
            prev_eqh = df.iloc[:i][df['liquidity_pool'] == "EQH"]['high'].last_valid_index()
            if prev_eqh:
                level = df.at[prev_eqh, 'high']
                if df.at[i, 'high'] > level and df.at[i, 'close'] < level:
                    df.at[i, 'liquidity_sweep'] = "BEARISH_SWEEP"

        return df
