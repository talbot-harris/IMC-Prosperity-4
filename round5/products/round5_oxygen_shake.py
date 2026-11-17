"""
OXYGEN_SHAKE — Market Making + Trend GARLIC + Mean Reversion CHOC/EVE
=======================================================================
Spread/tick ratio ~1.12–1.27 — solid MM base.
GARLIC has the strongest trend in the family (+3,886 over 3 days).
CHOCOLATE (AC1=–0.089) and EVENING_BREATH (AC1=–0.123) are mean-reverting.

Strategy by variant:
  GARLIC          → trend-following: hold net long, skip asks when trending up
  CHOCOLATE       → mean-reversion MM: fade moves > 1σ from EMA
  EVENING_BREATH  → mean-reversion MM: same (AC1=–0.12)
  MINT            → neutral MM (+155 net, low signal)
  MORNING_BREATH  → neutral MM (–450 net, very mild short lean)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT  = 10
SIZE   = 7
SKEW_T = 2
FAST_A = 0.05
SLOW_A = 0.003
EMA_A  = 0.10    # for mean-reversion EMA
MR_A   = 0.05    # slower EMA for MR variance tracking

MR_PRODUCTS = {"OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH"}
TREND_PRODUCT = "OXYGEN_SHAKE_GARLIC"


def _ema(td: dict, key: str, val: float, alpha: float) -> float:
    if td.get(key) is None:
        td[key] = val
    else:
        td[key] += alpha * (val - td[key])
    return td[key]


def _mm(product: str, od: OrderDepth, pos: int, td: dict,
        bid_tilt: int = 0, ask_tilt: int = 0) -> List[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    ema = _ema(td, product + "_ema", mid, EMA_A)
    fair_skew = int(round(max(-1, min(1, (mid - ema) / 4.0))))

    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))

    bid_p = max(1, min(bb + 1 - inv_skew + bid_tilt - fair_skew, ba - 1))
    ask_p = max(bb + 1, ba - 1 - inv_skew + ask_tilt - fair_skew)
    if bid_p >= ask_p:
        bid_p = ask_p - 1

    bid_q = min(SIZE, max(0, LIMIT - pos))
    ask_q = min(SIZE, max(0, LIMIT + pos))

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


def _mean_reversion(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """Fade moves > threshold from slow EMA.  Extra aggression proportional to distance."""
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    # Slow EMA = "fair value" for MR
    ema = _ema(td, product + "_mr_ema", mid, MR_A)

    # Running variance estimate via exponential smoothing
    err2 = (mid - ema) ** 2
    vk = product + "_mr_var"
    if td.get(vk) is None:
        td[vk] = err2
    else:
        td[vk] += MR_A * (err2 - td[vk])
    sigma = td[vk] ** 0.5 + 0.1  # avoid div-by-zero

    deviation = mid - ema
    z = deviation / sigma   # z-score of current deviation

    # When z is large, quote more aggressively on the reversion side
    orders = []
    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))

    if z > 1.5 and pos > -LIMIT:
        # Price too high → aggressive sell (expect reversion down)
        qty = min(SIZE + int(z * 1), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))  # hit the bid
    elif z < -1.5 and pos < LIMIT:
        # Price too low → aggressive buy (expect reversion up)
        qty = min(SIZE + int(-z * 1), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))  # lift the offer
    else:
        # Normal MM when close to EMA
        return _mm(product, od, pos, td)

    return orders


def _garlic(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """Trend-follow GARLIC: hold net long, MM with strong long lean."""
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    # Target long position confirmed by EMA; reduce if EMA turns negative
    target = LIMIT if drift >= -5 else LIMIT // 2

    orders = []
    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))

    # Passive MM with long lean (big bids, small asks)
    inv = pos / LIMIT
    inv_skew = int(round(inv * SKEW_T))
    bid_p = max(1, min(bb + 1 - inv_skew + 1, ba - 1))
    ask_p = max(bb + 1, ba - 1 - inv_skew)
    if bid_p >= ask_p:
        bid_p = ask_p - 1
    bid_q = int(min(SIZE * 1.3, max(0, LIMIT - pos)))
    ask_q = int(min(SIZE * 0.7, max(0, LIMIT + pos)))
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

        for product, od in state.order_depths.items():
            if not product.startswith("OXYGEN_SHAKE"):
                continue
            pos = state.position.get(product, 0)

            if product == TREND_PRODUCT:
                result[product] = _garlic(product, od, pos, td)
            elif product in MR_PRODUCTS:
                result[product] = _mean_reversion(product, od, pos, td)
            else:
                tilt = (0, 1) if "MORNING" in product else (0, 0)
                result[product] = _mm(product, od, pos, td, *tilt)

        return result, 0, json.dumps(td)
