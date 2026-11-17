"""
SLEEP_POD — Directional Long (All Variants)
============================================
All 5 variants are uptrending over 3 days:
  POLYESTER  +1,970  (strongest)
  SUEDE      +1,800
  COTTON     +1,414
  LAMB_WOOL    +808
  NYLON        +734  (weakest)

Strategy:
  Hold long positions sized proportionally to trend strength.
  Use EMA crossover to confirm direction before entering.
  Passive MM bids supplement to earn spread while accumulating.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT  = 10
SKEW_T = 2
FAST_A = 0.05
SLOW_A = 0.003

# (target_frac, mm_size)  — target_frac × LIMIT = target long position
CONFIGS = {
    "SLEEP_POD_POLYESTER":  (1.0, 5),
    "SLEEP_POD_SUEDE":      (1.0, 5),
    "SLEEP_POD_COTTON":     (0.9, 5),
    "SLEEP_POD_LAMB_WOOL":  (0.7, 5),
    "SLEEP_POD_NYLON":      (0.6, 5),
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

    frac, mm_sz = CONFIGS[product]

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    # Scale back target slightly if EMA is strongly against us
    if drift < -15:
        target = int(LIMIT * frac * 0.5)
    elif drift < -5:
        target = int(LIMIT * frac * 0.75)
    else:
        target = int(LIMIT * frac)

    orders = []

    # Aggressive take to close gap to target
    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))

    # Passive MM to earn spread while holding
    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))
    bid_p = max(1, min(bb + 1 - inv_skew, ba - 1))
    ask_p = max(bb + 1, ba - 1 - inv_skew)
    if bid_p >= ask_p:
        bid_p = ask_p - 1

    bid_q = min(mm_sz, max(0, LIMIT - pos))
    ask_q = min(mm_sz // 2, max(0, LIMIT + pos))  # smaller asks: we want to stay long

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
