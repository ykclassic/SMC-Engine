import pandas as pd
import numpy as np

class SMCEngine:
    def __init__(self, config):
        self.lookback = config['market_structure']['lookback_period']
        self.confirm_type = config['market_structure']['structure_type']

    def detect_fractals(self, df: pd.DataFrame):
        """
        Identifies Swing Highs and Swing Lows safely for live-market processing.
        Shifts the evaluation so the live edge is not converted to NaN.
        """
        window = self.lookback * 2 + 1
        
        df['is_high'] = 0
        df['is_low'] = 0
        
        # Iterate over the valid windows without blinding the live edge
        for i in range(window - 1, len(df)):
            slice_high = df['high'].iloc[i - window + 1 : i + 1]
            slice_low = df['low'].iloc[i - window + 1 : i + 1]
            
            # If the peak of the window is exactly in the middle (lookback), it is a fractal
            if slice_high.max() == slice_high.iloc[self.lookback]:
                target_idx = i - self.lookback
                df.iloc[target_idx, df.columns.get_loc('is_high')] = 1
                
            if slice_low.min() == slice_low.iloc[self.lookback]:
                target_idx = i - self.lookback
                df.iloc[target_idx, df.columns.get_loc('is_low')] = 1
                
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
        # INITIALIZATION FIX: Use None to allow object/string insertion without Pandas dtype warnings
        df['bos'] = None
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

    def process_market(self, macro_df: pd.DataFrame, current_micro: pd.DataFrame = None):
        """
        Processes market data through the SMC pipeline (fractals, structure points, BOS)
        and generates a trading signal tuple: (signal, structure_state, details).
        """
        df = self.get_structure_points(macro_df)
        df = self.detect_bos(df)
        
        latest = df.iloc[-1]
        signal = None
        
        bos_status = latest.get('bos', None)
        if bos_status == "BULLISH_BOS":
            signal = {"type": "LONG", "entry": latest.get('close'), "zone_limit": latest.get('low'), "reason": "Bullish Break of Structure confirmed"}
        elif bos_status == "BEARISH_BOS":
            signal = {"type": "SHORT", "entry": latest.get('close'), "zone_limit": latest.get('high'), "reason": "Bearish Break of Structure confirmed"}
            
        structure_state = latest.get('label', "")
        details = {
            "last_close": latest.get('close'),
            "bos": bos_status,
            "micro_available": current_micro is not None
        }
        
        return signal, structure_state, details

# Backward-compatibility alias to prevent import mismatches across modules
MarketStructure = SMCEngine
