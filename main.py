import sys
import logging
from utils.config_loader import load_all_configs
from data.exchange_api import ExchangeInterface
from core.engine import SMCEngine
from alerts.formatter import SignalFormatter
from alerts.discord_bot import DiscordNotifier

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger("Nexus-SMC-Main")
    
    config = load_all_configs()
    exchange = ExchangeInterface(config)
    engine = SMCEngine(config)
    notifier = DiscordNotifier(config)
    
    symbol = config.get("trading", {}).get("symbol", "BTC/USDT")

    logger.info(f"Nexus SMC Engine Started. Executing single-pass scan for {symbol}...")

    try:
        # 1. Fetch Data
        macro_raw = exchange.fetch_ohlcv(symbol, config['timeframes']['macro'], limit=200)
        micro_raw = exchange.fetch_ohlcv(symbol, config['timeframes']['micro'], limit=100)

        if macro_raw is None or micro_raw is None or micro_raw.empty:
            logger.warning("Fetched empty dataset from exchange. Exiting run.")
            sys.exit(0)

        # 2. Process Market
        signal, _, _ = engine.process_market(macro_raw, micro_raw)

        # 3. Handle Signal Generation
        if signal:
            latest_row = micro_raw.iloc[-1]
            
            if 'timestamp' in micro_raw.columns:
                current_time = latest_row['timestamp']
            elif 'time' in micro_raw.columns:
                current_time = latest_row['time']
            else:
                current_time = micro_raw.index[-1]

            logger.info(f"Signal detected for {symbol} at {current_time}. Broadcasting...")
            embed = SignalFormatter.format_discord_embed(signal, symbol, config)
            notifier.send_signal(embed)
        else:
            logger.info("No valid setups detected in this cycle. Exiting cleanly.")

    except Exception as e:
        logger.error(f"System Error during execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
