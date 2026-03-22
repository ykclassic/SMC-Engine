import yaml
import os
from dotenv import load_dotenv

def load_all_configs():
    """Loads YAML settings and Environment variables."""
    load_dotenv() # Load .env file
    
    config_path = os.path.join('config', 'settings.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Inject sensitive keys from .env into the config dict
    config['api_key'] = os.getenv('EXCHANGE_API_KEY')
    config['api_secret'] = os.getenv('EXCHANGE_API_SECRET')
    config['discord_webhook'] = os.getenv('DISCORD_WEBHOOK_URL')
    
    return config
