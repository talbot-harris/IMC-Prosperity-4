"""
UV_VISOR — Market Making + Skip AMBER
======================================
Spread/tick ratio 1.26–1.29 for all variants.
MAGENTA (+1,532) and RED (+1,722) are uptrending.
AMBER (–2,870) is a falling-knife: ratio 1.29 looks good but symmetric MM
accumulates losing longs.  Solution: AMBER = directional short only.

Strategy:
  AMBER    → short only (no passive bids, hit bids when above EMA)
  MAGENTA  → MM + long lean
  RED      → MM + long lean
  ORANGE   → neutral MM (–660 net, slight short lean)
  YELLOW   → neutral MM (+71 net)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT  = 10
SIZE   = 7
SKEW_T = 2
EMA_A  = 0.08
FAST_A = 0.05
SLOW_A = 0.003

CONFIGS = {
    #                     mode     bid_tilt ask_tilt bm   am
    "UV_VISOR_AMBER":   ("short",  0,       0,       0.0, 0.0),
    "UV_VISOR_MAGENTA": ("mm",     1,       0,       1.2, 0.8),
    "UV_VISOR_RED":     ("mm",     1,       0,       1.2, 0.8),
    "UV_VISOR_ORANGE":  ("mm",     0,       1,       0.9, 1.1),
    "UV_VISOR_YELLOW":  ("mm",     0,       0,       1.0, 1.0),
}


def _ema(td: dict, key: str, val: float, alpha: float) -> float:
    if td.get(key) is None:
        td[key] = val
    else:
        td[key] += alpha * (val - td[key])
    return td[key]


def _amber_short(od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """Directional short on AMBER: only sell, never buy passively."""
    product = "UV_VISOR_AMBER"
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    target = -LIMIT if drift <= 5 else -LIMIT // 2

    orders = []
    if pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))

    # Passive ask only — never post bids on AMBER
    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))
    ask_p = max(bb + 1, ba - 1 - inv_skew)
    ask_q = min(SIZE, max(0, LIMIT + pos))
    if ask_q > 0:
        orders.append(Order(product, ask_p, -ask_q))
    return orders


def _mm(product: str, od: OrderDepth, pos: int, td: dict,
        bt: int, at: int, bm: float, am: float) -> List[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    ema = _ema(td, product + "_ema", mid, EMA_A)
    fair_skew = int(round(max(-1, min(1, (mid - ema) / 5.0))))

    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))

    bid_p = max(1, min(bb + 1 - inv_skew + bt - fair_skew, ba - 1))
    ask_p = max(bb + 1, ba - 1 - inv_skew + at - fair_skew)
    if bid_p >= ask_p:
        bid_p = ask_p - 1

    bid_q = int(min(SIZE * bm, max(0.0, LIMIT - pos)))
    ask_q = int(min(SIZE * am, max(0.0, LIMIT + pos)))

    if inv > 0.7:
        bid_q = int(bid_q * max(0.0, (1.0 - inv) / 0.3))
    if inv < -0.7:
        ask_q = int(ask_q * max(0.0, (1.0 + inv) / 0.3))

    orders = []
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

        for product, cfg in CONFIGS.items():
            od = state.order_depths.get(product)
            if od is None:
                continue
            pos = state.position.get(product, 0)
            mode, bt, at, bm, am = cfg

            if mode == "short":
                result[product] = _amber_short(od, pos, td)
            else:
                result[product] = _mm(product, od, pos, td, bt, at, bm, am)

        return result, 0, json.dumps(td)
