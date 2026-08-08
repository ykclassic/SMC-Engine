import pandas as pd
from models.inference import ModelInference
from models.feature_engineering import FeatureEngineer

class ConfluenceEngine:
    def __init__(self, config):
        self.min_ai_confidence = 0.75 # 75% confidence required
        self.ai_engine = ModelInference(config)
        self.fe = FeatureEngineer()

    def validate_signal(self, macro_df: pd.DataFrame, micro_df: pd.DataFrame):
        """
        Evaluates the SMC structures on macro/micro timeframes, formulates a 
        base signal, and validates it through the AI inference engine.
        """
        # INITIALIZATION FIX: Ensure variable exists before conditional evaluation
        smc_signal_detected = None 
        
        latest_micro = micro_df.iloc[-1]
        latest_macro = macro_df.iloc[-1]
        
        # Read the detected sweeps from the liquidity engine
        sweep = latest_micro.get('liquidity_sweep', False)
        macro_bos = latest_macro.get('bos', "No Recent BOS")
        
        # Formulate initial SMC Signal based on micro timeframe sweeps
        if sweep == "BULLISH_SWEEP":
            smc_signal_detected = {
                "type": "LONG",
                "entry": latest_micro['close'],
                "reason": f"Bullish Liquidity Sweep. Macro context: {macro_bos}"
            }
        elif sweep == "BEARISH_SWEEP":
            smc_signal_detected = {
                "type": "SHORT",
                "entry": latest_micro['close'],
                "reason": f"Bearish Liquidity Sweep. Macro context: {macro_bos}"
            }

        # Run AI Validation if a structural setup was found
        if smc_signal_detected:
            features = self.fe.prepare_smc_features(micro_df)
            confidence = self.ai_engine.predict_confidence(features)
            
            if confidence >= self.min_ai_confidence:
                smc_signal_detected['ai_confidence'] = f"{confidence*100:.1f}%"
                return smc_signal_detected
            else:
                return None # AI rejected the trade
                
        return None
