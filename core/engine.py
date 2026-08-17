from __future__ import annotations

import pandas as pd

from core.confluence import ConfluenceEngine
from core.liquidity import LiquidityEngine
from core.structure import MarketStructureDetector
from core.zones import ZoneEngine


class SMCEngine:
    def __init__(self, config: dict) -> None:
        self.structure = MarketStructureDetector(config)
        self.liquidity = LiquidityEngine(config)
        self.zones = ZoneEngine(config)
        self.confluence = ConfluenceEngine(config)

    def process_market(self, daily_data: pd.DataFrame, h4_data: pd.DataFrame, h1_data: pd.DataFrame, m15_data: pd.DataFrame):
        daily = self._process_structure(daily_data, zones=True)
        h4 = self._process_structure(h4_data, zones=True)
        h1 = self._process_structure(h1_data, zones=True)
        m15 = self._process_structure(m15_data, zones=True)
        m15 = self.liquidity.identify_liquidity_pools(m15)
        m15 = self.liquidity.detect_sweeps(m15)
        signal, diagnostic = self.confluence.validate_signal(daily, h4, h1, m15)
        return signal, diagnostic, daily, h4, h1, m15

    def _process_structure(self, data: pd.DataFrame, zones: bool = True) -> pd.DataFrame:
        frame = self.structure.analyze(data)
        if zones:
            frame = self.zones.detect_fvg(frame)
            frame = self.zones.find_order_blocks(frame)
        return frame
