"""
PEBBLES — Directional Trend Following
=======================================
XL gained +6,068 and XS lost –3,962 over 3 days with consistent daily trends.
M is moderate (+702); L and S are mixed.

Strategy:
  XL  — aggressive long up to limit (consistent uptrend each day)
  XS  — aggressive short up to limit (consistent downtrend each day)
  S   — moderate short (-1934 net)
  L   — slight short (-874 net)
  M   — neutral MM (+702 net, low signal)

Execution:
  - Directional products: take market when under-positioned, then hold.
  - EMA crossover guards entry: only enter/hold if fast EMA confirms direction.
  - Passive MM quotes supplement to earn spread while holding.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT = 10

# (target_side, passive_mm_enabled, mm_size)
#   target_side: +1=long, -1=short, 0=neutral
CONFIGS = {
    "PEBBLES_XL": ( 1,  True,  5),
    "PEBBLES_M":  ( 0,  True,  7),
    "PEBBLES_L":  (-1,  True,  5),
    "PEBBLES_S":  (-1,  True,  5),
    "PEBBLES_XS": (-1,  True,  5),
}

FAST_A = 0.05
SLOW_A = 0.003
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

    cfg = CONFIGS[product]
    side, do_mm, mm_sz = cfg

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow  # >0 = uptrend in motion

    # EMA-confirmed target: only go in direction if EMA agrees
    if side == 1:
        target = LIMIT if drift >= 0 else max(0, LIMIT // 2)
    elif side == -1:
        target = -LIMIT if drift <= 0 else min(0, -LIMIT // 2)
    else:
        # Neutral: small directional lean from drift
        if drift > 10:
            target = LIMIT // 3
        elif drift < -10:
            target = -LIMIT // 3
        else:
            target = 0

    orders = []

    # Aggressive take to close gap to target
    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))
    elif pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))

    # Passive MM on top to earn spread while holding position
    if do_mm:
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
