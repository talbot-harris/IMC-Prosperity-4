"""
PANEL — Directional: Long 2X4, Short Others
============================================
PANEL_2X4 gained +2,354 consistently across all 3 days.
All other panel sizes declined (–304 to –872).
Area doesn't linearly determine price — 2X4 is structurally favored.

Strategy:
  PANEL_2X4  → max long (consistent uptrend each day)
  PANEL_4X4  → moderate short (–872 net, most negative)
  PANEL_1X4  → moderate short (–773 net)
  PANEL_2X2  → slight short (–607 net)
  PANEL_1X2  → slight short (–304 net)
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
    "PANEL_2X4": ( 1, 1.0, 5),
    "PANEL_4X4": (-1, 0.7, 5),
    "PANEL_1X4": (-1, 0.7, 5),
    "PANEL_2X2": (-1, 0.5, 5),
    "PANEL_1X2": (-1, 0.3, 5),
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

    # EMA-confirmed target
    if side == 1:
        target = int(LIMIT * frac) if drift >= -10 else int(LIMIT * frac * 0.5)
    elif side == -1:
        target = -int(LIMIT * frac) if drift <= 10 else -int(LIMIT * frac * 0.5)
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
