"""
GALAXY_SOUNDS — Market Making + Directional Tilt
=================================================
Spread/tick ratio ~1.26 across all 5 variants — solid MM.
BLACK_HOLES is the big winner (+3,458 net).
PLANETARY_RINGS is the laggard (–352 net).

Strategy:
  All 5 variants → passive MM (ratio > 1 means spread > tick vol)
  BLACK_HOLES    → lean long: more bids than asks, shifted up 1 tick
  SOLAR_FLAMES   → slight long lean (+823 net)
  PLANETARY_RINGS→ slight short lean (–352 net), avoid building longs
  DARK_MATTER    → neutral (+264 net)
  SOLAR_WINDS    → neutral (+248 net)

Signals used:
  - Inventory skew (standard)
  - Per-product directional tilt (static, data-driven)
  - EMA deviation for fair-value reference
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT  = 10
SIZE   = 7
SKEW_T = 2
EMA_A  = 0.08

# (bid_tilt, ask_tilt, bid_size_mult, ask_size_mult)
# bid_tilt >0 → bid 1 tick higher; ask_tilt >0 → ask 1 tick higher
# *_size_mult adjusts passive qty (1.0 = normal)
CONFIGS = {
    "GALAXY_SOUNDS_BLACK_HOLES":     ( 1, 0, 1.3, 0.7),  # strong long lean
    "GALAXY_SOUNDS_SOLAR_FLAMES":    ( 1, 0, 1.1, 0.9),  # mild long lean
    "GALAXY_SOUNDS_SOLAR_WINDS":     ( 0, 0, 1.0, 1.0),  # neutral
    "GALAXY_SOUNDS_DARK_MATTER":     ( 0, 0, 1.0, 1.0),  # neutral
    "GALAXY_SOUNDS_PLANETARY_RINGS": ( 0, 1, 0.8, 1.2),  # mild short lean
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

    ema = _ema(td, product + "_ema", mid, EMA_A)
    dev = mid - ema  # +ve = price above fair, lean short; -ve = lean long
    fair_skew = int(round(max(-1, min(1, dev / 5.0))))

    bt, at, bm, am = CONFIGS[product]

    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))

    bid_p = bb + 1 - inv_skew + bt - fair_skew
    ask_p = ba - 1 - inv_skew + at - fair_skew

    bid_p = max(1, min(bid_p, ba - 1))
    ask_p = max(bb + 1, ask_p)
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

        for product in CONFIGS:
            od = state.order_depths.get(product)
            if od is None:
                continue
            pos = state.position.get(product, 0)
            result[product] = _trade(product, od, pos, td)

        return result, 0, json.dumps(td)
