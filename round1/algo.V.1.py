"""
IMC Prosperity Round 1 - Optimized Strategy

IPR: dual-EMA trend filter. Default position is +80 (long) to capture the
constant upward drift. Only flattens to 0 when fast EMA drops more than
IPR_SIGNAL_THRESHOLD below slow EMA — i.e. during confirmed strong downtrends.
Shorting is avoided because the positive drift makes it negative-EV.

ACO: market-making with position skew (unchanged from optimizer).
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Tuple
import math
import json


class Trader:
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    # IPR momentum parameters
    IPR_FAST_ALPHA       = 0.05   # fast EMA alpha (~40-period)
    IPR_SLOW_ALPHA       = 0.01   # slow EMA alpha (~200-period)
    IPR_SIGNAL_THRESHOLD = 10.0   # go flat only if fast EMA < slow EMA - 10

    # ACO market-making parameters (tuned via optimizer)
    ACO_FAIR_VALUE         = 10000
    ACO_TAKE_THRESHOLD     = 1.5
    ACO_HALF_SPREAD        = 10
    ACO_POS_SKEW           = 0.15
    ACO_POS_SPREAD_PENALTY = 0.3
    ACO_AGGRESSIVE_REST    = 1

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            if product == "INTARIAN_PEPPER_ROOT":
                result[product], td = self._ipr(od, pos, td)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self._aco(od, pos)
            else:
                result[product] = []

        return result, conversions, json.dumps(td)

    @staticmethod
    def _mid(od):
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb and ba:
            return (bb + ba) / 2
        return bb or ba or 0

    def _ipr(self, od: OrderDepth, pos: int, td: dict) -> Tuple[List[Order], dict]:
        orders = []
        limit = self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        mid = self._mid(od)
        if mid == 0:
            return orders, td

        # Update dual EMAs
        fast = td.get("ipr_fast", mid)
        slow = td.get("ipr_slow", mid)
        fast += self.IPR_FAST_ALPHA * (mid - fast)
        slow += self.IPR_SLOW_ALPHA * (mid - slow)
        td["ipr_fast"] = fast
        td["ipr_slow"] = slow

        # Default LONG; only go flat on a confirmed strong downtrend.
        # With positive drift, fast EMA naturally sits above slow EMA, so
        # the signal fires only when recent momentum is genuinely negative.
        spread = fast - slow
        if spread > -self.IPR_SIGNAL_THRESHOLD:
            target = limit          # long (default — drift is positive)
        else:
            target = 0              # flat only on clear downtrend

        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None

        if target > pos:
            room = target - pos
            # Aggressive: take from ask side
            if od.sell_orders:
                for ap in sorted(od.sell_orders):
                    if room <= 0:
                        break
                    q = min(-od.sell_orders[ap], room)
                    if q > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ap, q))
                        room -= q
            # Passive: rest at best bid + 1 for remaining
            if room > 0 and bb is not None:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bb + 1, room))

        elif target < pos:
            room = pos - target
            # Aggressive: hit the bid side
            if od.buy_orders:
                for bp in sorted(od.buy_orders, reverse=True):
                    if room <= 0:
                        break
                    q = min(od.buy_orders[bp], room)
                    if q > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bp, -q))
                        room -= q
            # Passive: rest at best ask - 1 for remaining
            if room > 0 and ba is not None:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ba - 1, -room))

        return orders, td

    def _aco(self, od: OrderDepth, pos: int) -> List[Order]:
        orders = []
        limit = self.POSITION_LIMITS["ASH_COATED_OSMIUM"]
        mid = self._mid(od)
        if mid == 0:
            return orders
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        adj_fv = self.ACO_FAIR_VALUE - pos * self.ACO_POS_SKEW
        buy_room = limit - pos
        sell_room = limit + pos

        if od.sell_orders and buy_room > 0:
            for ap in sorted(od.sell_orders):
                if buy_room <= 0:
                    break
                if ap < adj_fv - self.ACO_TAKE_THRESHOLD:
                    q = min(-od.sell_orders[ap], buy_room)
                    if q > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ap, q))
                        buy_room -= q
        if od.buy_orders and sell_room > 0:
            for bp in sorted(od.buy_orders, reverse=True):
                if sell_room <= 0:
                    break
                if bp > adj_fv + self.ACO_TAKE_THRESHOLD:
                    q = min(od.buy_orders[bp], sell_room)
                    if q > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bp, -q))
                        sell_room -= q

        half_spread = self.ACO_HALF_SPREAD + abs(pos) * self.ACO_POS_SPREAD_PENALTY
        buy_price = math.floor(adj_fv - half_spread)
        sell_price = math.ceil(adj_fv + half_spread)
        if bb is not None and buy_price <= bb:
            buy_price = bb + 1 if self.ACO_AGGRESSIVE_REST else bb
        if ba is not None and sell_price >= ba:
            sell_price = ba - 1 if self.ACO_AGGRESSIVE_REST else ba
        if buy_price >= sell_price:
            buy_price = sell_price - 1

        if buy_room > 0:
            orders.append(Order("ASH_COATED_OSMIUM", buy_price, buy_room))
        if sell_room > 0:
            orders.append(Order("ASH_COATED_OSMIUM", sell_price, -sell_room))
        return orders
