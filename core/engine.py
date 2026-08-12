import pandas as pd  # ✅ REQUIRED FIX

from core.structure import MarketStructure
from core.liquidity import LiquidityEngine
from core.zones import ZoneEngine
from core.confluence import ConfluenceEngine

class SMCEngine:
    def __init__(self, config: dict):
        self.config = config
        self.structure = MarketStructure(config)
        self.liquidity = LiquidityEngine(config)
        self.zones = ZoneEngine(config)
        self.confluence = ConfluenceEngine(config)

    def process_market(
        self, 
        daily_data: pd.DataFrame, 
        h4_data: pd.DataFrame, 
        h1_data: pd.DataFrame, 
        m15_data: pd.DataFrame
    ) -> tuple:
        """
        Runs the full Multi-Timeframe Smart Money Concepts (SMC) pipeline.
        Executes across 1D, 4H, 1H, and 15M timeframes for precision execution.
        """

        # -----------------------------
        # Phase 1: Macro Context & Directional Bias (1D Timeframe)
        # Objective: Establish macro trend, market regime, and major structural boundaries.
        # -----------------------------
        daily_df = self.structure.get_structure_points(daily_data)
        daily_df = self.structure.detect_bos(daily_df)  
        # ✅ FIX: Detect FVG required to validate Daily order blocks
        daily_df = self.zones.detect_fvg(daily_df)
        daily_df = self.zones.find_order_blocks(daily_df) 

        # -----------------------------
        # Phase 2: Structural Mapping & Zones (4H Timeframe)
        # Objective: Map intermediate trend structures (BOS, CHoCH, FVG, OB).
        # -----------------------------
        h4_df = self.structure.get_structure_points(h4_data)
        h4_df = self.structure.detect_bos(h4_df)
        h4_df = self.zones.detect_fvg(h4_df)
        h4_df = self.zones.find_order_blocks(h4_df)

        # -----------------------------
        # Phase 3: Tactical Setup & Price Action (1H Timeframe)
        # Objective: Monitor intermediate momentum and look for local structural shifts.
        # -----------------------------
        h1_df = self.structure.get_structure_points(h1_data)
        # Identify local market structure shifts (CHoCH) tapping into 4H POIs
        if hasattr(self.structure, 'detect_choch'):
            h1_df = self.structure.detect_choch(h1_df)
        else:
            h1_df = self.structure.detect_bos(h1_df)

        # -----------------------------
        # Phase 4: Precision Execution & Risk Management (15M Timeframe)
        # Objective: Final entry triggers, sweep detection, and refined entry models.
        # -----------------------------
        m15_df = self.structure.get_structure_points(m15_data)
        m15_df = self.structure.detect_bos(m15_df) 
        m15_df = self.liquidity.identify_liquidity_pools(m15_df)
        m15_df = self.liquidity.detect_sweeps(m15_df)
        # ✅ FIX: Detect FVG required to validate local 15M order blocks
        m15_df = self.zones.detect_fvg(m15_df)
        m15_df = self.zones.find_order_blocks(m15_df)

        # -----------------------------
        # Confluence Validation
        # -----------------------------
        # Passing all timeframes to the confluence engine for strict top-down alignment
        signal = self.confluence.validate_signal(
            daily_df=daily_df,
            h4_df=h4_df,
            h1_df=h1_df,
            m15_df=m15_df
        )

        return signal, daily_df, h4_df, h1_df, m15_df
