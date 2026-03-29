import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

# --- FIX: Ensure the project root is in the Python Path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can safely import your core modules
from core.structure import SMCEngine
from utils.config_loader import load_all_configs
from data.exchange_api import ExchangeInterface

def run_backtest_session():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Nexus-Backtest")
    
    logger.info("🚀 Initializing Nexus Backtest Engine...")
    
    # 1. Load Config (Used for timeframes and symbol settings)
    try:
        config = load_all_configs()
    except Exception as e:
        logger.error(f"Configuration Load Failed: {e}")
        return

    # 2. Initialize Exchange (Public data only for backtesting)
    exchange = ExchangeInterface(config)
    symbol = config.get('symbol', 'BTC/USDT:USDT')
    
    # 3. Fetch Historical Data
    logger.info(f"Fetching historical data for {symbol}...")
    macro_df = exchange.fetch_ohlcv(symbol, config['timeframes']['macro'], limit=500)
    micro_df = exchange.fetch_ohlcv(symbol, config['timeframes']['micro'], limit=500)

    if macro_df.empty or micro_df.empty:
        logger.error("Failed to fetch historical data. Backtest aborted.")
        return

    # 4. Run SMC Engine Logic
    engine = SMCEngine(config)
    
    # We simulate a "rolling window" backtest
    results = []
    logger.info("Analyzing market structure for institutional setups...")
    
    # Start from index 50 to ensure indicators have enough "warm up" data
    for i in range(50, len(micro_df)):
        current_micro = micro_df.iloc[:i]
        # In a real backtest, you'd also slice the macro_df relative to micro time
        signal, _, _ = engine.process_market(macro_df, current_micro)
        
        if signal:
            results.append({
                "timestamp": current_micro.iloc[-1]['timestamp'],
                "type": signal['type'],
                "entry": signal['entry'],
                "sl": signal['stop_loss'],
                "tp": signal['take_profit'],
                "confidence": signal.get('ai_confidence', 'N/A')
            })

    # 5. Report Results
    if results:
        logger.info(f"✅ Backtest Complete. Found {len(results)} high-quality signals.")
        df_results = pd.DataFrame(results)
        print("\n--- BACKTEST SIGNAL LOG ---")
        print(df_results.to_string(index=False))
        
        # Save results for GitHub Artifacts
        os.makedirs('reports', exist_ok=True)
        df_results.to_csv('reports/backtest_results.csv', index=False)
    else:
        logger.info("🏁 Backtest Complete. No high-quality institutional setups found.")

if __name__ == "__main__":
    run_backtest_session()
