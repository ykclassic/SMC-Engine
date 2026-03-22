import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_smc_features(self, df: pd.DataFrame):
        """
        Converts SMC labels (BOS, FVG, OB) into numerical features for the model.
        """
        # Create numerical proxies for SMC events
        df['bos_numeric'] = df['bos'].map({'BULLISH_BOS': 1, 'BEARISH_BOS': -1}).fillna(0)
        df['fvg_numeric'] = df['fvg'].map({'BULLISH_FVG': 1, 'BEARISH_FVG': -1}).fillna(0)
        df['ob_numeric'] = df['order_block'].map({'BULLISH_OB': 1, 'BEARISH_OB': -1}).fillna(0)
        
        # Standard Technical Features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=10).std()
        
        # Select features for the AI
        feature_cols = ['close', 'volatility', 'bos_numeric', 'fvg_numeric', 'ob_numeric']
        df_cleaned = df[feature_cols].fillna(0)
        
        return self.scaler.fit_transform(df_cleaned)
