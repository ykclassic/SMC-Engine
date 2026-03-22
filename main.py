import time
import logging
from utils.config_loader import load_all_configs
from data.exchange_api import ExchangeInterface
from core.engine import SMCEngine
from alerts.formatter import SignalFormatter
from alerts.discord_bot import DiscordNotifier

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Nexus-SMC-Main")
    
    config = load_all_configs()
    exchange = ExchangeInterface(config)
    engine = SMCEngine(config)
    notifier = DiscordNotifier(config)
    
    symbol = "BTC/USDT"
    last_signal_time = None

    logger.info("Nexus SMC Engine Started. Monitoring markets...")

    while True:
        try:
            # 1. Fetch Data
            macro_raw = exchange.fetch_ohlcv(symbol, config['timeframes']['macro'], limit=200)
            micro_raw = exchange.fetch_ohlcv(symbol, config['timeframes']['micro'], limit=100)

            # 2. Process Market
            signal, _, _ = engine.process_market(macro_raw, micro_raw)

            # 3. Handle Signal (Only alert if it's a new timestamp)
            if signal:
                current_time = micro_raw.iloc[-1]['timestamp']
                if current_time != last_signal_time:
                    embed = SignalFormatter.format_discord_embed(signal, symbol, config)
                    notifier.send_signal(embed)
                    last_signal_time = current_time
            
            # 4. Wait for next candle (e.g., check every 60 seconds)
            time.sleep(60)

        except Exception as e:
            logger.error(f"System Error in main loop: {e}")
            time.sleep(30) # Cool-down before retry

if __name__ == "__main__":
    main()
