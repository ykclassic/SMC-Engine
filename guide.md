---

### 4. `GUIDE.md` (Technical Documentation)
This document explains the "Why" behind the logic for future maintenance.

```markdown
# Technical Implementation Guide: SMC Logic

## 1. Market Structure (`core/structure.py`)
We use a **Fractal-based approach** to identify swing points. A High is only confirmed if it is higher than the `n` candles to its left and right. This prevents "noisy" highs from triggering false Trend Changes.

## 2. Order Block Validation (`core/zones.py`)
The engine uses a "High Probability" filter for Order Blocks. An OB is only logged if:
- It is immediately followed by a **Fair Value Gap**.
- It resulted in a **Break of Structure (BOS)**.
- It remains **Unmitigated** (Price has not returned to the zone yet).

## 3. The Confluence Logic (`core/confluence.py`)
This is the "Fail-Safe." Even if a 15m chart shows a perfect Bullish Order Block, the engine will **REJECT** the signal if the 4H chart is in a Bearish Trend. We only trade with the "Higher Timeframe Flow."

## 4. Risk Calculation
Stop Losses are automatically placed 0.1% beyond the structural high/low of the Order Block. Take Profit is calculated based on a fixed **1:3 Risk/Reward ratio**, ensuring long-term profitability even with a 40% win rate.
