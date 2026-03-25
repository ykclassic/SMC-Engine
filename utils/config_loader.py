import yaml
import os
import logging
from dotenv import load_dotenv

def load_all_configs():
    """
    Loads YAML settings and Environment variables.
    Bridges GitHub Secrets / .env into the application's config dictionary.
    """
    # 1. Load local .env (useful for local development)
    load_dotenv() 
    
    logger = logging.getLogger("Nexus-ConfigLoader")
    
    # 2. Load Static YAML Settings
    config_path = os.path.join('config', 'settings.yaml')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            if config is None:
                config = {}
    except FileNotFoundError:
        logger.error(f"❌ config/settings.yaml not found at {config_path}")
        raise

    # 3. Inject Sensitive Keys from Environment (GitHub Secrets or .env)
    # We map these to the keys expected by the ExchangeInterface and DiscordBot
    config['api_key'] = os.getenv('EXCHANGE_API_KEY')
    config['api_secret'] = os.getenv('EXCHANGE_API_SECRET')
    config['passphrase'] = os.getenv('EXCHANGE_PASSPHRASE') # Required for Bitget
    config['discord_webhook_url'] = os.getenv('DISCORD_WEBHOOK_URL')

    # 4. Mandatory Secret Validation
    required_secrets = {
        'EXCHANGE_API_KEY': config['api_key'],
        'EXCHANGE_API_SECRET': config['api_secret'],
        'EXCHANGE_PASSPHRASE': config['passphrase'],
        'DISCORD_WEBHOOK_URL': config['discord_webhook_url']
    }

    missing = [k for k, v in required_secrets.items() if not v]
    
    if missing:
        logger.error(f"❌ CRITICAL: Missing required environment variables: {', '.join(missing)}")
        # We raise an error to stop the app from running in an invalid state
        raise EnvironmentError(f"Missing environment variables: {missing}")
    
    logger.info("✅ Configuration and Secrets loaded and validated successfully.")
    
    return config
