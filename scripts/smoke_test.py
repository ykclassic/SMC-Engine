import os
import ccxt
import requests
import logging
from dotenv import load_dotenv

def run_smoke_test():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("Nexus-SmokeTest")
    load_dotenv()

    # 1. Load Credentials from Environment
    api_key = os.getenv('EXCHANGE_API_KEY')
    api_secret = os.getenv('EXCHANGE_API_SECRET')
    passphrase = os.getenv('EXCHANGE_PASSPHRASE') # Required for Bitget
    webhook = os.getenv('DISCORD_WEBHOOK_URL')

    logger.info("Starting Nexus API Connectivity Test...")

    # Validate presence of all keys
    missing = []
    if not api_key: missing.append("EXCHANGE_API_KEY")
    if not api_secret: missing.append("EXCHANGE_API_SECRET")
    if not passphrase: missing.append("EXCHANGE_PASSPHRASE")
    if not webhook: missing.append("DISCORD_WEBHOOK_URL")

    if missing:
        logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
        return

    # 2. Initialize Exchange with Bitget-specific 'password' field
    try:
        exchange = ccxt.bitget({
            'apiKey': api_key,
            'secret': api_secret,
            'password': passphrase,  # <--- Critical for Bitget/OKX
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'} # Default to USDT-M Futures
        })

        # Test Connectivity: Fetch Balance
        balance = exchange.fetch_balance()
        usdt_total = balance['total'].get('USDT', 0)
        logger.info(f"✅ Exchange Connected. Available USDT: {usdt_total}")

        # Test Connectivity: Fetch Market Price
        ticker = exchange.fetch_ticker('BTC/USDT:USDT')
        logger.info(f"✅ Market Data Fetched. BTC Price: {ticker['last']}")

    except ccxt.AuthenticationError:
        logger.error("❌ Authentication Failed: Check Key, Secret, and Passphrase.")
        return
    except Exception as e:
        logger.error(f"❌ Connection Error: {str(e)}")
        return

    # 3. Test Discord Webhook
    try:
        payload = {
            "username": "Nexus Guardian",
            "content": "🚀 **Nexus Smoke Test:** API and Webhook connection verified for TechSolute."
        }
        response = requests.post(webhook, json=payload)
        if response.status_code == 204:
            logger.info("✅ Discord Webhook Verified.")
        else:
            logger.warning(f"⚠️ Discord returned status: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Discord Notification Failed: {e}")

if __name__ == "__main__":
    run_smoke_test()
