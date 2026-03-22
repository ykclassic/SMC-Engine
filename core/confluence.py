import pandas as pd

class ConfluenceEngine:
    def __init__(self, config):
        self.require_alignment = config['confluence']['require_macro_alignment']

    def validate_signal(self, macro_df: pd.DataFrame, micro_df: pd.DataFrame):
        """
        Validates if a 15m setup aligns with 4H structure.
        1. Get current Macro Bias (Last BOS on 4H).
        2. Check if Micro (15m) has a matching BOS.
        3. Check if Micro is currently inside a Macro Order Block.
        """
        # 1. Determine Macro Bias
        last_macro_bos = macro_df[macro_df['bos'] != False].iloc[-1]
        macro_bias = "BULLISH" if last_macro_bos['bos'] == "BULLISH_BOS" else "BEARISH"
        
        # 2. Get Latest Micro State
        latest_micro = micro_df.iloc[-1]
        micro_bos = micro_df[micro_df['bos'] != False].iloc[-1] if not micro_df[micro_df['bos'] != False].empty else None
        
        # 3. Check for Macro OB Mitigation (Are we in a high-value zone?)
        macro_obs = macro_df[(macro_df['order_block'].notnull()) & (macro_df['ob_mitigated'] == False)]
        in_macro_zone = False
        active_zone = None

        current_price = latest_micro['close']
        
        for _, ob in macro_obs.iterrows():
            if ob['order_block'] == "BULLISH_OB":
                if ob['ob_bottom'] <= current_price <= ob['ob_top']:
                    in_macro_zone = True
                    active_zone = ob
            elif ob['order_block'] == "BEARISH_OB":
                if ob['ob_bottom'] >= current_price >= ob['ob_top']:
                    in_macro_zone = True
                    active_zone = ob

        # 4. Final Logic Decision
        signal = None
        if macro_bias == "BULLISH" and in_macro_zone:
            if micro_bos and micro_bos['bos'] == "BULLISH_BOS":
                signal = {
                    "type": "LONG",
                    "reason": "Macro Bullish + Price in Macro OB + Micro BOS Confirmed",
                    "entry": current_price,
                    "zone_limit": active_zone['ob_bottom']
                }
        
        elif macro_bias == "BEARISH" and in_macro_zone:
            if micro_bos and micro_bos['bos'] == "BEARISH_BOS":
                signal = {
                    "type": "SHORT",
                    "reason": "Macro Bearish + Price in Macro OB + Micro BOS Confirmed",
                    "entry": current_price,
                    "zone_limit": active_zone['ob_top']
                }

        return signal
