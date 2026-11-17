"""
IMC Prosperity Round 4 — VFE-only algorithm

Product: VELVETFRUIT_EXTRACT (abbrev: VFE)
Position limit: 200

Why this is standalone:
  Every prior version coupled VFE to the option / HP logic and used a slow
  EMA as the fair-value anchor. On a trending day the EMA lags the mid by a
  few ticks; whichever side of the book is closer to fair fills more often
  → on a downtrend, our bid is the closer side, we accumulate longs, then
  eat the rest of the move. Day-3 submission 500913 lost 3,534 on VFE this
  way — PnL tracked the mid 1:1 implying a stuck +100 long.

Design:
  1. NO fair-value EMA. Anchor every quote to the live (bb, ba). The EMA
     of a trending mid is the bug, not the cure.
  2. Inventory skew shifts both quotes in the SAME direction against
     position. When long, both bid and ask shift down → bid less likely to
     fill (good), ask more likely to fill (good). At full position the
     skew is 5 ticks, big enough to actually flip the fill bias on a
     5-tick spread.
  3. Linear size taper on the side that *adds* to inventory: full size
     when flat, zero size at the throttle threshold.
  4. Aggressive unwind once past the throttle: cross the book at bb (or
     ba) to dump inventory — willingly pays 1-tick spread to escape a
     stuck position before the next adverse move arrives.
  5. Larger passive size (30, up from 6) because (a) every fill at bb+1 /
     ba-1 captures ~3 ticks of spread on the observed 5-tick book, and
     (b) the inventory taper means we self-throttle before pinning.

Other products: this file emits no orders for HP or any VEV strike. Combine
with separate per-product algos later.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:

    PRODUCT       = "VELVETFRUIT_EXTRACT"
    LIMIT         = 200

    SIZE          = 30      # passive quote size when flat
    SKEW_TICKS    = 5       # at |inv|=1, shift both quotes 5 ticks against pos
    THROTTLE_FRAC = 0.65    # past this fraction of limit, kill add-side and unwind
    AGGR_SIZE     = 25      # max single-tick aggressive unwind size

    def run(self, state: TradingState):
        result = {p: [] for p in state.order_depths}

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        if self.PRODUCT not in state.order_depths:
            return result, 0, json.dumps(td)

        od  = state.order_depths[self.PRODUCT]
        pos = state.position.get(self.PRODUCT, 0)

        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return result, 0, json.dumps(td)

        inv  = pos / self.LIMIT                          # in [-1, 1]
        skew = int(round(inv * self.SKEW_TICKS))         # ticks to shift quotes

        orders: List[Order] = []

        # Aggressive unwind path: past the throttle, cross the book.
        if inv > self.THROTTLE_FRAC:
            # Long-pinned: dump at bb.
            avail = od.buy_orders.get(bb, 0)
            target_excess = pos - int(self.THROTTLE_FRAC * self.LIMIT)
            qty = min(self.AGGR_SIZE, avail, target_excess, self.LIMIT + pos)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.SIZE, self.LIMIT + pos - qty)

        elif inv < -self.THROTTLE_FRAC:
            # Short-pinned: lift ba.
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.THROTTLE_FRAC * self.LIMIT)
            qty = min(self.AGGR_SIZE, avail, target_excess, self.LIMIT - pos)
            if qty > 0:
                orders.append(Order(self.PRODUCT, ba, qty))
            ask_qty = 0
            bid_qty = min(self.SIZE, self.LIMIT - pos - qty)

        else:
            # Linear size taper on the side that adds to inventory.
            # add_frac = signed inventory in the adding direction.
            # bid adds long; ask adds short.
            bid_add_frac = max(0.0,  inv) / self.THROTTLE_FRAC
            ask_add_frac = max(0.0, -inv) / self.THROTTLE_FRAC
            bid_qty = int(self.SIZE * (1.0 - bid_add_frac))
            ask_qty = int(self.SIZE * (1.0 - ask_add_frac))
            bid_qty = min(bid_qty, self.LIMIT - pos)
            ask_qty = min(ask_qty, self.LIMIT + pos)

        # Passive quotes. Default = capture spread inside (bb+1, ba-1).
        # Skew shifts both in the same direction against current position.
        our_bid = bb + 1 - skew
        our_ask = ba - 1 - skew

        # Safety: never lift offers passively, never slam bids passively
        # (the aggressive unwind path above handles intentional crossing).
        our_bid = min(our_bid, ba - 1)
        our_ask = max(our_ask, bb + 1)
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if bid_qty > 0:
            orders.append(Order(self.PRODUCT, our_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order(self.PRODUCT, our_ask, -ask_qty))

        result[self.PRODUCT] = orders

        # Snapshot for post-mortem.
        td["vfe"] = {
            "ts": state.timestamp,
            "pos": pos, "bb": bb, "ba": ba,
            "skew": skew, "bid_qty": bid_qty, "ask_qty": ask_qty,
        }
        return result, 0, json.dumps(td)