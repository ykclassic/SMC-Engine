import os
import ccxt
import requests
import logging
from dotenv import load_dotenv

def run_smoke_test():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Nexus-SmokeTest")
    load_dotenv()

    # 1. Check API Keys
    api_key = os.getenv('EXCHANGE_API_KEY')
    api_secret = os.getenv('EXCHANGE_API_SECRET')
    webhook = os.getenv('DISCORD_WEBHOOK_URL')

    if not all([api_key, api_secret, webhook]):
        logger.error("❌ Missing environment variables in .env")
        return

    # 2. Test Exchange Connection
    try:
        exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        balance = exchange.fetch_balance()
        usdt_balance = balance['total'].get('USDT', 0)
        logger.info(f"✅ Exchange Connected. USDT Balance: {usdt_balance}")
    except Exception as e:
        logger.error(f"❌ Exchange Connection Failed: {e}")
        return

    # 3. Test Discord Webhook
    try:
        payload = {"content": "🚀 **Nexus-SMC Smoke Test:** System API Connection Verified!"}
        response = requests.post(webhook, json=payload)
        if response.status_code == 204:
            logger.info("✅ Discord Webhook Verified.")
    except Exception as e:
        logger.error(f"❌ Discord Webhook Failed: {e}")

if __name__ == "__main__":
    run_smoke_test()
