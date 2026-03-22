import pandas as pd

class DataProcessor:
    @staticmethod
    def clean_data(df: pd.DataFrame):
        """Removes duplicates and handles missing values."""
        if df.empty:
            return df
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df = df.ffill() # Forward fill any gaps in price
        return df.reset_index(drop=True)

    @staticmethod
    def validate_for_smc(df: pd.DataFrame, lookback: int):
        """Ensures the dataframe has enough rows to calculate fractals."""
        min_required = (lookback * 2) + 1
        return len(df) >= min_required
