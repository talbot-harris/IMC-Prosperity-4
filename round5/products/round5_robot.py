"""
ROBOT — Mean Reversion DISHES + Directional Shorts
====================================================
DISHES has the strongest mean-reversion signal in the entire dataset
(AC1 = –0.232), meaning each large move tends to partially reverse.
Tick vol = 17.8, spread = 7.4 → low MM ratio, but MR strategy exploits
the reversal structure instead.

IRONING (–2,170) and VACUUMING (–1,725) are strong downtrends.

Strategy:
  DISHES    → mean reversion: buy dips below EMA, sell spikes above EMA
  MOPPING   → slight long lean (+1,588 net)
  LAUNDRY   → slight short lean (–746 net)
  IRONING   → directional short (–2,170 net, consistent each day)
  VACUUMING → directional short (–1,725 net)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

LIMIT  = 10
SKEW_T = 2
FAST_A = 0.05
SLOW_A = 0.003
MR_EMA = 0.12    # faster EMA for mean reversion signal
MR_VAR = 0.08    # variance tracker alpha


def _ema(td: dict, key: str, val: float, alpha: float) -> float:
    if td.get(key) is None:
        td[key] = val
    else:
        td[key] += alpha * (val - td[key])
    return td[key]


def _dishes_mr(od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    Mean-reversion strategy for ROBOT_DISHES (AC1=–0.232).
    Cross the spread when z-score |deviation / sigma| > threshold.
    """
    product = "ROBOT_DISHES"
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    ema = _ema(td, product + "_mr", mid, MR_EMA)

    # Running variance
    err2 = (mid - ema) ** 2
    vk = product + "_var"
    if td.get(vk) is None:
        td[vk] = err2 + 1.0
    else:
        td[vk] += MR_VAR * (err2 - td[vk])
    sigma = max(td[vk] ** 0.5, 1.0)

    z = (mid - ema) / sigma
    inv = pos / LIMIT

    orders = []

    # z > 1.5: price spiked up → expect reversal → sell
    if z > 1.5 and pos > -LIMIT:
        qty = min(int(4 + z * 2), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))          # hit the bid
    # z < -1.5: price dipped → expect reversal → buy
    elif z < -1.5 and pos < LIMIT:
        qty = min(int(4 + (-z) * 2), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))           # lift the offer
    else:
        # Inside threshold: passive MM around EMA
        inv_skew = int(round(inv * SKEW_T))
        bid_p = max(1, min(bb + 1 - inv_skew, ba - 1))
        ask_p = max(bb + 1, ba - 1 - inv_skew)
        if bid_p >= ask_p:
            bid_p = ask_p - 1
        bid_q = min(20, max(0, LIMIT - pos))
        ask_q = min(20, max(0, LIMIT + pos))
        if bid_q > 0:
            orders.append(Order(product, bid_p, bid_q))
        if ask_q > 0:
            orders.append(Order(product, ask_p, -ask_q))

    return orders


def _directional(product: str, od: OrderDepth, pos: int, td: dict,
                  side: int, frac: float, mm_sz: int) -> List[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    if side == 1:
        target = int(LIMIT * frac) if drift >= -10 else int(LIMIT * frac * 0.5)
    else:
        target = -int(LIMIT * frac) if drift <= 10 else -int(LIMIT * frac * 0.5)

    orders = []

    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), LIMIT - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))
    elif pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), LIMIT + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))

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


# (strategy, side, frac, mm_size)
CONFIGS = {
    "ROBOT_DISHES":    ("mr",    0,  0.0, 0),
    "ROBOT_MOPPING":   ("dir",   1,  0.5, 5),
    "ROBOT_LAUNDRY":   ("dir",  -1,  0.4, 5),
    "ROBOT_IRONING":   ("dir",  -1,  1.0, 5),
    "ROBOT_VACUUMING": ("dir",  -1,  0.9, 5),
}


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
            strat, side, frac, mm_sz = cfg

            if strat == "mr":
                result[product] = _dishes_mr(od, pos, td)
            else:
                result[product] = _directional(product, od, pos, td, side, frac, mm_sz)

        return result, 0, json.dumps(td)
