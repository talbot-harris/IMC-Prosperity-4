"""
IMC Prosperity Round 4 — HP-only algorithm

Product: HYDROGEL_PACK (abbrev: HP)
Position limit: 200

Empirical day-3 stats (from 501054 log):
  - mid range:          9993 to 10061 (~10σ daily range)
  - per-tick std:       2.17
  - spread histogram:   16  → 92.5% of ticks
                        17  →  3.9%
                        7-9 →  3.6% (tight regime; 3rd level inside book)
  - L1 vol (each side): mean 12.4
  - V.4 result:         -263 — got short on the +9 close move and ate it,
                        same EMA-lag adverse-selection pathology as VFE.

Why HP over VEV options for this round of fixes:
  - 16-tick spread vs VFE's 5 → ~3× edge per fill at bb+1 / ba-1
  - simple mean-revert-ish underlying, no BS / IV / TTE machinery needed
  - V.4 was sized at 8 — leaving most of the 200 limit on the table

Design (mirrors the working VFE structure with HP-tuned constants):
  1. NO fair-value EMA. Anchor to live (bb, ba). Same root cause as VFE:
     a lagging EMA biases the closer side to fill more often, which on a
     trending tick accumulates the wrong inventory.
  2. Quote at bb+1 - skew (bid) and ba-1 - skew (ask). bb+1 / ba-1
     captures 14 ticks of spread per round-trip on the 16-tick book —
     strictly more profitable per fill than V.4's mid±4 quoting.
  3. SKEW_TICKS = 8: at full position the skew shifts both quotes by
     half the typical spread, large enough to actually flip the fill
     bias when we're loaded.
  4. Linear size taper on the side that adds inventory; full size when
     flat → 0 at the throttle. Self-throttles before pinning.
  5. Past the throttle, aggressively cross at bb (or ba) to dump
     inventory. Pays one tick of spread to avoid carrying directional
     risk into the next adverse move.
  6. SIZE = 40 (vs V.4's 8). HP spread is ~3× VFE's, std is ~2× VFE's →
     edge-per-fill is ~1.5× VFE's, so size scales up roughly accordingly.

Other products: this file emits no orders for VFE or any VEV strike.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:

    PRODUCT       = "HYDROGEL_PACK"
    LIMIT         = 200

    SIZE          = 40      # passive quote size when flat
    SKEW_TICKS    = 8       # at |inv|=1, shift both quotes 8 ticks against pos
    THROTTLE_FRAC = 0.60    # past this fraction of limit, kill add-side and unwind
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
            avail = od.buy_orders.get(bb, 0)
            target_excess = pos - int(self.THROTTLE_FRAC * self.LIMIT)
            qty = min(self.AGGR_SIZE, avail, target_excess, self.LIMIT + pos)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.SIZE, self.LIMIT + pos - qty)

        elif inv < -self.THROTTLE_FRAC:
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.THROTTLE_FRAC * self.LIMIT)
            qty = min(self.AGGR_SIZE, avail, target_excess, self.LIMIT - pos)
            if qty > 0:
                orders.append(Order(self.PRODUCT, ba, qty))
            ask_qty = 0
            bid_qty = min(self.SIZE, self.LIMIT - pos - qty)

        else:
            # Linear size taper on the side that adds to inventory.
            bid_add_frac = max(0.0,  inv) / self.THROTTLE_FRAC
            ask_add_frac = max(0.0, -inv) / self.THROTTLE_FRAC
            bid_qty = int(self.SIZE * (1.0 - bid_add_frac))
            ask_qty = int(self.SIZE * (1.0 - ask_add_frac))
            bid_qty = min(bid_qty, self.LIMIT - pos)
            ask_qty = min(ask_qty, self.LIMIT + pos)

        # Passive quotes: capture spread inside (bb+1, ba-1) shifted by skew.
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

        td["hp"] = {
            "ts": state.timestamp,
            "pos": pos, "bb": bb, "ba": ba, "spread": ba - bb,
            "skew": skew, "bid_qty": bid_qty, "ask_qty": ask_qty,
        }
        return result, 0, json.dumps(td)