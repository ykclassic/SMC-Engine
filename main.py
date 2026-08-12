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
    
    # -----------------------------
    # Configuration Enforcement
    # -----------------------------
    # Enforcing correct exchange and core fiat-paired instruments
    config.setdefault("trading", {})
    config["trading"]["exchange"] = "xt.com"
    symbols_to_scan = ["BTCUSD", "ETHUSD"]
    
    exchange = ExchangeInterface(config)
    engine = SMCEngine(config)
    notifier = DiscordNotifier(config)
    
    logger.info("Initializing Equinox Protocol... Pre-loading XT.com market structure...")

    for symbol in symbols_to_scan:
        logger.info(f"Executing multi-timeframe scan for {symbol}...")

        try:
            # 1. Fetch Data (1D, 4H, 1H, 15M)
            daily_raw = exchange.fetch_ohlcv(symbol, "1d", limit=100)
            h4_raw = exchange.fetch_ohlcv(symbol, "4h", limit=200)
            h1_raw = exchange.fetch_ohlcv(symbol, "1h", limit=200)
            m15_raw = exchange.fetch_ohlcv(symbol, "15m", limit=100)

            # Validate dataset integrity across all timeframes
            datasets = [daily_raw, h4_raw, h1_raw, m15_raw]
            if any(df is None or df.empty for df in datasets):
                logger.warning(f"Fetched incomplete dataset from XT.com for {symbol}. Skipping to next asset.")
                continue

            # 2. Process Market (4-Phase SMC Pipeline)
            signal, daily_df, h4_df, h1_df, m15_df = engine.process_market(
                daily_raw, h4_raw, h1_raw, m15_raw
            )

            # 3. Handle Signal Generation
            if signal:
                latest_row = m15_raw.iloc[-1]
                
                if 'timestamp' in m15_raw.columns:
                    current_time = latest_row['timestamp']
                elif 'time' in m15_raw.columns:
                    current_time = latest_row['time']
                else:
                    current_time = m15_raw.index[-1]

                logger.info(f"Signal detected for {symbol} at {current_time}. Broadcasting...")
                embed = SignalFormatter.format_discord_embed(signal, symbol, config)
                notifier.send_signal(embed)
            else:
                logger.info(f"No valid setups detected for {symbol} in this cycle.")

        except Exception as e:
            logger.error(f"System Error during execution for {symbol}: {e}", exc_info=True)
            # Continue to the next symbol instead of killing the entire pipeline
            continue
            
    logger.info("Execution cycle complete. Exiting cleanly.")

if __name__ == "__main__":
    main()
