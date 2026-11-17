from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import math
import json
 
 
class Trader:
    """
    Round 1 Trading Algorithm — IMC Prosperity
 
    Products & Strategies:
    ──────────────────────
    INTARIAN_PEPPER_ROOT (IPR)
      - Near-perfect linear uptrend: +1 per 1000 ticks (~+1000/day)
      - Detrended residuals std ≈ 2.4 (extremely predictable)
      - Strategy: Trend-following. Stay maximally long. Buy aggressively.
 
    ASH_COATED_OSMIUM (ACO)
      - Mean-reverts around 10000, oscillation amplitude ~17, cycle ~7500 ticks
      - Strategy: Market-making. Buy below 10000, sell above 10000.
        Position-aware skew to manage inventory risk.
    """
 
    # Position limits — adjust if the challenge specifies different values
    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 50,
        "ASH_COATED_OSMIUM": 50,
    }
 
    # Tuned parameters (backtested on days -2, -1, 0)
    IPR_BUY_PREMIUM = 5       # willing to buy up to mid + this (trend covers it)
    ACO_FAIR_VALUE = 10000     # mean-reversion anchor
    ACO_POSITION_SKEW = 0.1   # shift fair value per unit of position
    ACO_TAKE_THRESHOLD = 0.5  # min edge to take liquidity
    ACO_BASE_HALF_SPREAD = 4  # base half-spread for resting orders
    ACO_POS_SPREAD_PENALTY = 0.1  # widen spread per unit of position
 
    def __init__(self):
        pass
 
    def run(self, state: TradingState):
        result = {}
        conversions = 0
 
        # Persist state across ticks if needed
        trader_data = state.traderData if hasattr(state, 'traderData') and state.traderData else ""
 
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)
 
            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_ipr(order_depth, position, limit, state.timestamp)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_aco(order_depth, position, limit, state.timestamp)
            else:
                result[product] = []
 
        return result, conversions, trader_data
 
    # ─── Helpers ────────────────────────────────────────────────────────
 
    @staticmethod
    def best_bid(od: OrderDepth):
        return max(od.buy_orders.keys()) if od.buy_orders else None
 
    @staticmethod
    def best_ask(od: OrderDepth):
        return min(od.sell_orders.keys()) if od.sell_orders else None
 
    @staticmethod
    def mid_price(od: OrderDepth):
        bb = max(od.buy_orders.keys()) if od.buy_orders else None
        ba = min(od.sell_orders.keys()) if od.sell_orders else None
        if bb and ba:
            return (bb + ba) / 2
        return bb or ba or 0
 
    # ─── INTARIAN_PEPPER_ROOT ───────────────────────────────────────────
 
    def trade_ipr(self, od: OrderDepth, position: int, limit: int, timestamp: int) -> List[Order]:
        """
        Strong uptrend → always buy, never sell voluntarily.
        Expected profit from holding = ~1 per 1000 ticks per unit.
        """
        orders: List[Order] = []
        mid = self.mid_price(od)
        if mid == 0:
            return orders
 
        buy_room = limit - position
        bb = self.best_bid(od)
 
        # === TAKE LIQUIDITY: buy all asks within premium of mid ===
        if od.sell_orders and buy_room > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if buy_room <= 0:
                    break
                if ask_price <= mid + self.IPR_BUY_PREMIUM:
                    volume = -od.sell_orders[ask_price]  # sell_orders has negative qty
                    qty = min(volume, buy_room)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, qty))
                        buy_room -= qty
 
        # === MAKE LIQUIDITY: place aggressive resting buy ===
        if buy_room > 0 and bb is not None:
            # Bid aggressively at best_bid + 1 to get filled
            orders.append(Order("INTARIAN_PEPPER_ROOT", bb + 1, buy_room))
 
        return orders
 
    # ─── ASH_COATED_OSMIUM ─────────────────────────────────────────────
 
    def trade_aco(self, od: OrderDepth, position: int, limit: int, timestamp: int) -> List[Order]:
        """
        Mean-reversion market-making around fair value = 10000.
        Uses position-aware skew to manage inventory.
        """
        orders: List[Order] = []
        mid = self.mid_price(od)
        if mid == 0:
            return orders
 
        bb = self.best_bid(od)
        ba = self.best_ask(od)
 
        # Adjusted fair value: lean against current position
        adj_fv = self.ACO_FAIR_VALUE - position * self.ACO_POSITION_SKEW
 
        buy_room = limit - position
        sell_room = limit + position
 
        # === TAKE LIQUIDITY: cross spread when edge > threshold ===
 
        # Buy cheap asks
        if od.sell_orders and buy_room > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if buy_room <= 0:
                    break
                if ask_price < adj_fv - self.ACO_TAKE_THRESHOLD:
                    volume = -od.sell_orders[ask_price]
                    qty = min(volume, buy_room)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask_price, qty))
                        buy_room -= qty
 
        # Sell expensive bids
        if od.buy_orders and sell_room > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if sell_room <= 0:
                    break
                if bid_price > adj_fv + self.ACO_TAKE_THRESHOLD:
                    volume = od.buy_orders[bid_price]
                    qty = min(volume, sell_room)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid_price, -qty))
                        sell_room -= qty
 
        # === MAKE LIQUIDITY: place resting orders ===
        half_spread = self.ACO_BASE_HALF_SPREAD + abs(position) * self.ACO_POS_SPREAD_PENALTY
 
        buy_price = math.floor(adj_fv - half_spread)
        sell_price = math.ceil(adj_fv + half_spread)
 
        # Stay competitive but don't cross the book
        if bb is not None and buy_price <= bb:
            buy_price = bb + 1
        if ba is not None and sell_price >= ba:
            sell_price = ba - 1
 
        # Prevent self-crossing
        if buy_price >= sell_price:
            buy_price = sell_price - 1
 
        if buy_room > 0:
            orders.append(Order("ASH_COATED_OSMIUM", buy_price, buy_room))
        if sell_room > 0:
            orders.append(Order("ASH_COATED_OSMIUM", sell_price, -sell_room))
 
        return orders