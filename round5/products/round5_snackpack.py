"""
SNACKPACK — Pure Market Making
================================
Best MM candidates in the round: spread/tick ratio 2.08–3.04.
Tick volatility is tiny (5–8) while spreads are 16–18.

STRAWBERRY and VANILLA are slight up-trends; PISTACHIO is the worst.
We carry a ~+0.2 inventory target on STRAWBERRY, –0.2 on PISTACHIO.
All other variants are neutral.

Signals used:
  - Inventory skew       (shift quotes to flatten position)
  - Directional tilt     (per-variant static bias, data-driven)
  - Volume imbalance     (widen/narrow based on side pressure)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT   = 10      # position limit per product
SIZE    = 8       # max passive qty per side
SKEW_T  = 3       # ticks of quote shift per unit of inv_ratio
VBI_SOFT = 0.15   # VBI dead-band (ignore small imbalances)

# Static tilt: (bid_extra, ask_extra) in ticks.  + = higher price.
# Positive bid_extra → quote higher bids → tend to stay long.
TILTS = {
    "SNACKPACK_STRAWBERRY": ( 1,  0),  # mild long bias  (+901 net)
    "SNACKPACK_VANILLA":    ( 0,  0),  # neutral         (+325 net)
    "SNACKPACK_RASPBERRY":  ( 0,  0),  # neutral         (+300 net)
    "SNACKPACK_CHOCOLATE":  ( 0,  1),  # mild short bias (-338 net)
    "SNACKPACK_PISTACHIO":  (-1,  1),  # short bias      (-887 net)
}

PRODUCTS = list(TILTS.keys())


def _mm(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    if ba <= bb:
        return []

    mid = (bb + ba) / 2.0

    # EMA for reference (not used for quoting here, but kept for future use)
    k = product + "_ema"
    if td.get(k) is None:
        td[k] = mid
    else:
        td[k] += 0.08 * (mid - td[k])

    # Volume-balance imbalance signal
    bv = sum(od.buy_orders.values())
    av = -sum(od.sell_orders.values())
    vbi = (bv - av) / max(1, bv + av)
    vbi_shift = 0
    if vbi > VBI_SOFT:
        vbi_shift = 1
    elif vbi < -VBI_SOFT:
        vbi_shift = -1

    # Inventory skew
    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))

    # Directional tilt
    be, ae = TILTS.get(product, (0, 0))

    bid_p = bb + 1 - inv_skew + be + vbi_shift
    ask_p = ba - 1 - inv_skew + ae + vbi_shift

    # Clamp so we never cross
    bid_p = max(1, min(bid_p, ba - 1))
    ask_p = max(bb + 1, ask_p)
    if bid_p >= ask_p:
        bid_p = ask_p - 1

    bid_q = min(SIZE, max(0, LIMIT - pos))
    ask_q = min(SIZE, max(0, LIMIT + pos))

    # Taper as we approach limits
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

        for product in PRODUCTS:
            od = state.order_depths.get(product)
            if od is None:
                continue
            pos = state.position.get(product, 0)
            result[product] = _mm(product, od, pos, td)

        return result, 0, json.dumps(td)
