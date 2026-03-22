import pandas as pd
import numpy as np

class ZoneEngine:
    def __init__(self, config):
        self.min_gap = config['imbalance']['min_gap_size']
        self.require_fvg = config['order_blocks']['require_fvg']
        self.mitigation_limit = config['order_blocks']['mitigation_threshold']

    def detect_fvg(self, df: pd.DataFrame):
        """Identifies Fair Value Gaps (Imbalances)."""
        df['fvg'] = None
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan

        for i in range(2, len(df)):
            if df.at[i, 'low'] > df.at[i-2, 'high']:
                gap_size = (df.at[i, 'low'] - df.at[i-2, 'high']) / df.at[i-2, 'high']
                if gap_size >= self.min_gap:
                    df.at[i-1, 'fvg'] = "BULLISH_FVG"
                    df.at[i-1, 'fvg_top'] = df.at[i, 'low']
                    df.at[i-1, 'fvg_bottom'] = df.at[i-2, 'high']
            elif df.at[i, 'high'] < df.at[i-2, 'low']:
                gap_size = (df.at[i-2, 'low'] - df.at[i, 'high']) / df.at[i-2, 'low']
                if gap_size >= self.min_gap:
                    df.at[i-1, 'fvg'] = "BEARISH_FVG"
                    df.at[i-1, 'fvg_top'] = df.at[i-2, 'low']
                    df.at[i-1, 'fvg_bottom'] = df.at[i, 'high']
        return df

    def find_order_blocks(self, df: pd.DataFrame):
        """
        Detects Order Blocks. 
        Bullish OB: The last bearish candle before a Bullish BOS.
        Bearish OB: The last bullish candle before a Bearish BOS.
        """
        df['order_block'] = None
        df['ob_top'] = np.nan
        df['ob_bottom'] = np.nan
        df['ob_mitigated'] = False

        # Iterate through detected BOS points
        bos_indices = df[df['bos'] != False].index

        for idx in bos_indices:
            bos_type = df.at[idx, 'bos']
            
            # Look backward for the 'origin' candle
            search_range = range(idx - 1, max(0, idx - 20), -1)
            
            if bos_type == "BULLISH_BOS":
                for s_idx in search_range:
                    if df.at[s_idx, 'close'] < df.at[s_idx, 'open']: # Found last Down candle
                        # Optional: Validate if an FVG exists nearby
                        has_fvg = df.iloc[s_idx:idx]['fvg'].notnull().any()
                        if self.require_fvg and not has_fvg:
                            continue
                            
                        df.at[s_idx, 'order_block'] = "BULLISH_OB"
                        df.at[s_idx, 'ob_top'] = df.at[s_idx, 'high']
                        df.at[s_idx, 'ob_bottom'] = df.at[s_idx, 'low']
                        break

            elif bos_type == "BEARISH_BOS":
                for s_idx in search_range:
                    if df.at[s_idx, 'close'] > df.at[s_idx, 'open']: # Found last Up candle
                        has_fvg = df.iloc[s_idx:idx]['fvg'].notnull().any()
                        if self.require_fvg and not has_fvg:
                            continue
                            
                        df.at[s_idx, 'order_block'] = "BEARISH_OB"
                        df.at[s_idx, 'ob_top'] = df.at[s_idx, 'high']
                        df.at[s_idx, 'ob_bottom'] = df.at[s_idx, 'low']
                        break
        
        return self._check_mitigation(df)

    def _check_mitigation(self, df: pd.DataFrame):
        """Marks OBs as mitigated if future price has returned to the zone."""
        ob_indices = df[df['order_block'].notnull()].index
        
        for ob_idx in ob_indices:
            ob_top = df.at[ob_idx, 'ob_top']
            ob_bottom = df.at[ob_idx, 'ob_bottom']
            
            # Look at all candles AFTER the OB candle
            future_prices = df.iloc[ob_idx + 1:]
            
            if df.at[ob_idx, 'order_block'] == "BULLISH_OB":
                # Mitigated if any future Low enters the zone
                if (future_prices['low'] <= ob_top).any():
                    df.at[ob_idx, 'ob_mitigated'] = True
            
            elif df.at[ob_idx, 'order_block'] == "BEARISH_OB":
                # Mitigated if any future High enters the zone
                if (future_prices['high'] >= ob_bottom).any():
                    df.at[ob_idx, 'ob_mitigated'] = True
                    
        return df
