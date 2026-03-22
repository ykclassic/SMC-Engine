from models.inference import ModelInference
from models.feature_engineering import FeatureEngineer

class ConfluenceEngine:
    def __init__(self, config):
        self.min_ai_confidence = 0.75 # 75% confidence required
        self.ai_engine = ModelInference(config)
        self.fe = FeatureEngineer()

    def validate_signal(self, macro_df, micro_df):
        # ... [Previous SMC Logic] ...

        if smc_signal_detected:
            # Run AI Validation
            features = self.fe.prepare_smc_features(micro_df)
            confidence = self.ai_engine.predict_confidence(features)
            
            if confidence >= self.min_ai_confidence:
                smc_signal_detected['ai_confidence'] = f"{confidence*100:.1f}%"
                return smc_signal_detected
            else:
                return None # AI rejected the trade
