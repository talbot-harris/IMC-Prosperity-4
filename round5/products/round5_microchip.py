"""
MICROCHIP — Directional Divergence Trade
==========================================
SQUARE (+3,633) is strongly negatively correlated with RECTANGLE (–0.882)
and OVAL (–0.732).  TRIANGLE also declining (–2,058).

Strong diverging trends → directional play, NOT mean reversion:
  SQUARE    → max long  (+3,633)
  OVAL      → max short (–4,481)
  RECTANGLE → max short (–1,228)
  TRIANGLE  → moderate short (–2,058)
  CIRCLE    → neutral/slight long (+388)

Execution:
  - EMA crossover confirms direction before entering.
  - Aggressive take to build to target, then hold.
  - Passive MM quotes on CIRCLE (neutral) to earn spread.
  - For directional legs: also post passive quotes on the directional side
    to compound fills.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT = 10
FAST_A = 0.05
SLOW_A = 0.003

# (target_side, target_frac, mm_size)
#   target_frac: fraction of LIMIT to hold directionally
CONFIGS = {
    "MICROCHIP_SQUARE":    ( 1,  1.0, 5),
    "MICROCHIP_OVAL":      (-1,  1.0, 5),
    "MICROCHIP_RECTANGLE": (-1,  0.8, 5),
    "MICROCHIP_TRIANGLE":  (-1,  0.7, 5),
    "MICROCHIP_CIRCLE":    ( 0,  0.0, 7),  # neutral MM
}

SKEW_T = 2


def _ema(td: dict, key: str, val: float, alpha: float) -> float:
    if td.get(key) is None:
        td[key] = val
    else:
        td[key] += alpha * (val - td[key])
    return td[key]


def _trade(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    side, frac, mm_sz = CONFIGS[product]

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    if side == 1:
        # Go long; reduce target if EMA says we're in a dip against the trend
        target = int(LIMIT * frac) if drift >= -20 else int(LIMIT * frac * 0.5)
    elif side == -1:
        target = -int(LIMIT * frac) if drift <= 20 else -int(LIMIT * frac * 0.5)
    else:
        target = 0

    orders = []

    # Aggressive take to reach target
    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))
    elif pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))

    # Passive MM overlay (works in both directions; inventory skew handles it)
    if mm_sz > 0:
        inv = pos / LIMIT
        skew = int(round(inv * SKEW_T))
        bid_p = min(bb + 1 - skew, ba - 1)
        ask_p = max(ba - 1 - skew, bb + 1)
        if bid_p >= ask_p:
            bid_p = ask_p - 1
        bid_q = min(mm_sz, max(0, LIMIT - pos))
        ask_q = min(mm_sz, max(0, LIMIT + pos))
        if bid_q > 0:
            orders.append(Order(product, bid_p, bid_q))
        if ask_q > 0:
            orders.append(Order(product, ask_p, -ask_q))

    return orders


class Trader:

    def run(self, state: TradingState):
        result = {}
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product in CONFIGS:
            od = state.order_depths.get(product)
            if od is None:
                continue
            pos = state.position.get(product, 0)
            result[product] = _trade(product, od, pos, td)

        return result, 0, json.dumps(td)
