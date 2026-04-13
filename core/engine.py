import pandas as pd  # ✅ REQUIRED FIX

from core.structure import MarketStructure
from core.liquidity import LiquidityEngine
from core.zones import ZoneEngine
from core.confluence import ConfluenceEngine


class SMCEngine:
    def __init__(self, config):
        self.config = config
        self.structure = MarketStructure(config)
        self.liquidity = LiquidityEngine(config)
        self.zones = ZoneEngine(config)
        self.confluence = ConfluenceEngine(config)

    def process_market(self, macro_data: pd.DataFrame, micro_data: pd.DataFrame):
        """Runs the full SMC pipeline across two timeframes."""

        # -----------------------------
        # Process Macro (4H)
        # -----------------------------
        macro_df = self.structure.get_structure_points(macro_data)
        macro_df = self.structure.detect_bos(macro_df)
        macro_df = self.zones.detect_fvg(macro_df)
        macro_df = self.zones.find_order_blocks(macro_df)

        # -----------------------------
        # Process Micro (15m)
        # -----------------------------
        micro_df = self.structure.get_structure_points(micro_data)
        micro_df = self.structure.detect_bos(micro_df)
        micro_df = self.liquidity.identify_liquidity_pools(micro_df)
        micro_df = self.liquidity.detect_sweeps(micro_df)

        # -----------------------------
        # Confluence (SMC + AI)
        # -----------------------------
        signal = self.confluence.validate_signal(macro_df, micro_df)

        return signal, macro_df, micro_df
