import pandas as pd
import numpy as np
import logging

class LiquidityEngine:
    def __init__(self, config):
        self.threshold = config['liquidity'].get('eq_threshold', 0.001)
        self.sweep_buffer = config['liquidity'].get('sweep_buffer', 0.0005)
        self.logger = logging.getLogger("Nexus-Liquidity")

    def identify_liquidity_pools(self, df: pd.DataFrame):
        """
        Finds Equal Highs and Equal Lows where price has hit a level 
        multiple times without breaking it.
        """
        df['liquidity_pool'] = None
        
        highs = df[df['is_high'] == 1]
        lows = df[df['is_low'] == 1]

        for i in range(len(highs) - 1):
            price_a = highs.iloc[i]['high']
            price_b = highs.iloc[i+1]['high']
            if abs(price_a - price_b) / price_a <= self.threshold:
                idx = highs.index[i+1]
                df.at[idx, 'liquidity_pool'] = "EQH"

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
        OR a previous major fractal high/low, but fails to close beyond it.
        """
        df['liquidity_sweep'] = None
        
        for i in range(1, len(df)):
            sub_df = df.iloc[:i]
            
            # Check for Sell-side Sweep (Sweeping Lows)
            recent_lows = sub_df[sub_df['is_low'] == 1]
            if not recent_lows.empty:
                # Target either the last EQL or the most recent fractal low
                target_low = recent_lows['low'].iloc[-1]
                
                if df.at[i, 'low'] < target_low and df.at[i, 'close'] > target_low:
                    df.at[i, 'liquidity_sweep'] = "BULLISH_SWEEP"

            # Check for Buy-side Sweep (Sweeping Highs)
            recent_highs = sub_df[sub_df['is_high'] == 1]
            if not recent_highs.empty:
                # Target either the last EQH or the most recent fractal high
                target_high = recent_highs['high'].iloc[-1]
                
                if df.at[i, 'high'] > target_high and df.at[i, 'close'] < target_high:
                    df.at[i, 'liquidity_sweep'] = "BEARISH_SWEEP"

        return df
