import pandas as pd
import logging
from models.inference import ModelInference
from models.feature_engineering import FeatureEngineer

class ConfluenceEngine:
    def __init__(self, config):
        self.min_ai_confidence = 0.75 
        self.ai_engine = ModelInference(config)
        self.fe = FeatureEngineer()
        self.logger = logging.getLogger("Nexus-Confluence")

    def validate_signal(self, daily_df: pd.DataFrame, h4_df: pd.DataFrame, h1_df: pd.DataFrame, m15_df: pd.DataFrame):
        """
        Evaluates the SMC structures on macro/micro timeframes, formulates a 
        base signal, and validates it through the AI inference engine.
        """
        smc_signal_detected = None 
        
        recent_micro = m15_df.tail(3)
        latest_macro = h4_df.iloc[-1]
        
        macro_bos = latest_macro.get('bos', "No Recent BOS")
        sweep = None
        trigger_price = None
        
        for _, row in recent_micro.iterrows():
            if pd.notna(row.get('liquidity_sweep')):
                sweep = row['liquidity_sweep']
                trigger_price = row['close']
                break
        
        if not sweep:
            self.logger.debug("Confluence Gate 1 Failed: No liquidity sweep detected in the last 3 micro candles.")
            return None

        if sweep == "BULLISH_SWEEP":
            smc_signal_detected = {
                "type": "LONG",
                "entry": trigger_price,
                "reason": f"Bullish Liquidity Sweep. Macro context: {macro_bos}"
            }
        elif sweep == "BEARISH_SWEEP":
            smc_signal_detected = {
                "type": "SHORT",
                "entry": trigger_price,
                "reason": f"Bearish Liquidity Sweep. Macro context: {macro_bos}"
            }

        if smc_signal_detected:
            self.logger.info(f"Structural Setup Found: {smc_signal_detected['reason']}. Pushing features to AI Inference...")
            
            features = self.fe.prepare_smc_features(m15_df)
            confidence = self.ai_engine.predict_confidence(features)
            
            if confidence >= self.min_ai_confidence:
                self.logger.info(f"AI Validation Passed. Confidence: {confidence*100:.1f}%")
                smc_signal_detected['ai_confidence'] = f"{confidence*100:.1f}%"
                return smc_signal_detected
            else:
                self.logger.info(f"AI Validation Failed. Confidence {confidence*100:.1f}% is below {self.min_ai_confidence*100}% threshold.")
                return None 
                
        return None
