"""
TRANSLATOR — Pairs Divergence Trade
=====================================
VOID_BLUE (+1,564) and SPACE_GRAY (–1,571) are near-mirror divergences.
ASTRO_BLACK (–1,036) is also declining.

These are NOT mean-reverting pairs — they diverge directionally.
Strategy: long VOID_BLUE, short SPACE_GRAY (divergence trade).
Secondary: short ASTRO_BLACK.

Spread ratio ~0.88 on all variants (tight for pure MM), so:
  - Directional mode: aggressive takes to build target position
  - MM overlay: earn spread while in position
  - ECLIPSE_CHARCOAL / GRAPHITE_MIST: near-flat, neutral MM
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT  = 10
SKEW_T = 2
FAST_A = 0.05
SLOW_A = 0.003

# (target_side, target_frac, mm_size)
CONFIGS = {
    "TRANSLATOR_VOID_BLUE":        ( 1,  1.0, 5),
    "TRANSLATOR_SPACE_GRAY":       (-1,  1.0, 5),
    "TRANSLATOR_ASTRO_BLACK":      (-1,  0.7, 5),
    "TRANSLATOR_ECLIPSE_CHARCOAL": ( 0,  0.0, 7),
    "TRANSLATOR_GRAPHITE_MIST":    ( 0,  0.0, 7),
}


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
        target = int(LIMIT * frac) if drift >= -8 else int(LIMIT * frac * 0.5)
    elif side == -1:
        target = -int(LIMIT * frac) if drift <= 8 else -int(LIMIT * frac * 0.5)
    else:
        target = 0

    orders = []

    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))
    elif pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))

    # Passive MM overlay
    if mm_sz > 0:
        inv = pos / LIMIT
        inv_skew = int(round(inv * SKEW_T))
        bid_p = max(1, min(bb + 1 - inv_skew, ba - 1))
        ask_p = max(bb + 1, ba - 1 - inv_skew)
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
