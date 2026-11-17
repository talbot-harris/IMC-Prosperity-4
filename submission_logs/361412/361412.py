"""
IMC Prosperity Round 1 - Optimized Strategy
Position limit: 80 (corrected from 50)

Parameters tuned via algorithm.py optimizer. Note: the simulator was
calibrated against a position-50 submission; bumping to 80 is a linear
scale-up in max exposure per product. Expected PnL should scale roughly
proportionally on ACO (market-making scales with inventory capacity) and
exactly linearly on IPR (buy-and-hold captures +1/1000 ticks * 80 units
instead of * 50 units, roughly 49508 * 80/50 = ~79,200 expected).

Rough sim-implied PnL at limit 80 (linear extrapolation):
  IPR: ~79,200
  ACO: ~127,000
  Total: ~206,000
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import math


class Trader:
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    # Tuned parameters (from optimizer)
    IPR_BUY_PREMIUM        = 50
    IPR_RESTING_OFFSET     = 1
    ACO_FAIR_VALUE         = 10000
    ACO_TAKE_THRESHOLD     = 1.5
    ACO_HALF_SPREAD        = 10
    ACO_POS_SKEW           = 0.15
    ACO_POS_SPREAD_PENALTY = 0.3
    ACO_AGGRESSIVE_REST    = 1

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = state.traderData if hasattr(state, "traderData") and state.traderData else ""
        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._ipr(od, pos)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self._aco(od, pos)
            else:
                result[product] = []
        return result, conversions, trader_data

    @staticmethod
    def _mid(od):
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb and ba: return (bb + ba) / 2
        return bb or ba or 0

    def _ipr(self, od, pos):
        orders = []
        limit = self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        mid = self._mid(od)
        if mid == 0: return orders
        buy_room = limit - pos
        bb = max(od.buy_orders) if od.buy_orders else None
        if od.sell_orders and buy_room > 0:
            for ap in sorted(od.sell_orders):
                if buy_room <= 0: break
                if ap <= mid + self.IPR_BUY_PREMIUM:
                    v = -od.sell_orders[ap]
                    q = min(v, buy_room)
                    if q > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ap, q))
                        buy_room -= q
        if buy_room > 0 and bb is not None:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bb + self.IPR_RESTING_OFFSET, buy_room))
        return orders

    def _aco(self, od, pos):
        orders = []
        limit = self.POSITION_LIMITS["ASH_COATED_OSMIUM"]
        mid = self._mid(od)
        if mid == 0: return orders
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        adj_fv = self.ACO_FAIR_VALUE - pos * self.ACO_POS_SKEW
        buy_room = limit - pos
        sell_room = limit + pos

        if od.sell_orders and buy_room > 0:
            for ap in sorted(od.sell_orders):
                if buy_room <= 0: break
                if ap < adj_fv - self.ACO_TAKE_THRESHOLD:
                    v = -od.sell_orders[ap]
                    q = min(v, buy_room)
                    if q > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ap, q))
                        buy_room -= q
        if od.buy_orders and sell_room > 0:
            for bp in sorted(od.buy_orders, reverse=True):
                if sell_room <= 0: break
                if bp > adj_fv + self.ACO_TAKE_THRESHOLD:
                    v = od.buy_orders[bp]
                    q = min(v, sell_room)
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