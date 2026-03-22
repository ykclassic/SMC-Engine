import torch
import os

class ModelInference:
    def __init__(self, config):
        self.model_path = os.path.join('models', 'weights', 'latest_gru.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SignalValidatorGRU().to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()

    def predict_confidence(self, processed_features):
        """Returns a probability (0-1) that the current setup will succeed."""
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(processed_features).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor)
            return prediction.item()
