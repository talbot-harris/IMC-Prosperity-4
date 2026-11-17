"""
IMC Prosperity Round 4 — HP-only algorithm (v3, VBI-aware)

Product: HYDROGEL_PACK (abbrev: HP)
Position limit: 200

What changed from v2 (round.4.algo.HP.py prior):
  - Added an order-flow imbalance (VBI) gate on top of the v2 inventory MM.
  - hp_signal_tester showed VBI is the dominant signal for HP:
      vbi_tot Q1 = -14.95 / Q5 = +15.35, win5 = 95% at horizon=30.
      vbi_L1  Q1 = -14.90 / Q5 = +15.29, win5 = 96% at horizon=30.
    Smoothed versions collapse to Q5 ≈ +4 → the alpha lives in the untreated
    tails, so we threshold raw VBI (no EMA, no MR average).
  - When the book is balanced (|vbi| < VBI_SOFT) we behave exactly like v2.
  - When VBI tilts past VBI_SOFT we shrink the adverse-selected side,
    grow the aligned side, and skew the quote midpoint toward the flow.
  - Stays passive: with a 16-tick spread and a +15-tick mean fwd move at
    Q5, crossing the book is roughly break-even after costs.
  - Skipped vol_50 / range_200 / dist_ma_200: their high quintile spreads
    are coming from this day's monotonic up-drift, not a portable signal.
    slope_5/10/20 had |r30|<0.02 — pure noise.

Day-3 (501140) replay against current quoting model:
  +2000 simulated PnL, 65 fills, 71-74% win rate already — but only
  3.6 avg qty per fill, so we are leaving ~3/4 of book size unused. The
  VBI gate raises participation precisely on the 95%-win ticks.

Other products: this file emits no orders for VFE or any VEV strike.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class Trader:

    PRODUCT       = "HYDROGEL_PACK"
    LIMIT         = 200

    # Base passive market making (unchanged from v2)
    SIZE          = 40
    SKEW_TICKS    = 8
    THROTTLE_FRAC = 0.60
    AGGR_SIZE     = 25

    # VBI flow gating. Day-3 distribution: 964/1000 ticks have vbi=0; the
    # non-zero tail tops out at |0.50| on L1 and |0.12| on full book. So the
    # gate has to fire on small absolute values — 0.05 already isolates the
    # ~36 informative ticks where the 95% Q5 win-rate edge lives.
    VBI_SOFT      = 0.05   # any non-zero imbalance starts asymmetric quoting
    VBI_HARD      = 0.20   # fully kills the adverse side
    VBI_BOOST     = 0.75   # max extra size multiplier on the aligned side
    VBI_PRICE_TICKS = 3    # max ticks of midpoint shift toward flow
    # No aggressive take: with a 16-tick spread and a +15-tick mean fwd move
    # at Q5, crossing the book is ~break-even after costs. Stay passive.

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
        skew = int(round(inv * self.SKEW_TICKS))

        # L1 VBI: untreated raw signal — tails carry the alpha. L1 has
        # wider dynamic range than full-book here (max |0.50| vs |0.12|)
        # and identical Q5 win rate in the signal tester (96% vs 95%).
        bv1 = od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        vbi = (bv1 - av1) / max(1, bv1 + av1)            # in [-1, 1]

        # Continuous strength in [0,1] between SOFT and HARD thresholds.
        denom = max(1e-9, self.VBI_HARD - self.VBI_SOFT)
        vbi_strength = max(0.0, min(1.0, (abs(vbi) - self.VBI_SOFT) / denom))

        bid_flow_mult = 1.0
        ask_flow_mult = 1.0
        flow_shift    = 0
        if vbi > self.VBI_SOFT:
            # Buyers dominate → asks fill into a rising mid (adverse).
            ask_flow_mult = max(0.0, 1.0 - vbi_strength)
            bid_flow_mult = 1.0 + self.VBI_BOOST * vbi_strength
            flow_shift    = int(round(self.VBI_PRICE_TICKS * vbi_strength))
        elif vbi < -self.VBI_SOFT:
            bid_flow_mult = max(0.0, 1.0 - vbi_strength)
            ask_flow_mult = 1.0 + self.VBI_BOOST * vbi_strength
            flow_shift    = -int(round(self.VBI_PRICE_TICKS * vbi_strength))

        orders: List[Order] = []

        # Aggressive unwind (risk, untouched). VBI does not override this.
        if inv > self.THROTTLE_FRAC:
            avail = od.buy_orders.get(bb, 0)
            target_excess = pos - int(self.THROTTLE_FRAC * self.LIMIT)
            qty = min(self.AGGR_SIZE, avail, target_excess, self.LIMIT + pos)
            if qty > 0:
                orders.append(Order(self.PRODUCT, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.SIZE, self.LIMIT + pos - qty)
            flow_shift = 0          # don't fight the unwind

        elif inv < -self.THROTTLE_FRAC:
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.THROTTLE_FRAC * self.LIMIT)
            qty = min(self.AGGR_SIZE, avail, target_excess, self.LIMIT - pos)
            if qty > 0:
                orders.append(Order(self.PRODUCT, ba, qty))
            ask_qty = 0
            bid_qty = min(self.SIZE, self.LIMIT - pos - qty)
            flow_shift = 0

        else:
            # Linear inventory taper × VBI flow multiplier on each side.
            bid_add_frac = max(0.0,  inv) / self.THROTTLE_FRAC
            ask_add_frac = max(0.0, -inv) / self.THROTTLE_FRAC
            bid_qty = int(self.SIZE * (1.0 - bid_add_frac) * bid_flow_mult)
            ask_qty = int(self.SIZE * (1.0 - ask_add_frac) * ask_flow_mult)
            bid_qty = min(bid_qty, self.LIMIT - pos)
            ask_qty = min(ask_qty, self.LIMIT + pos)

        # Passive quotes: capture spread inside (bb+1, ba-1), shifted by both
        # the inventory skew and the VBI flow direction.
        our_bid = bb + 1 - skew + flow_shift
        our_ask = ba - 1 - skew + flow_shift

        # Safety: passive bid never crosses ba; passive ask never crosses bb.
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
            "vbi": round(vbi, 3),
            "skew": skew, "flow_shift": flow_shift,
            "bid_qty": bid_qty, "ask_qty": ask_qty,
        }
        return result, 0, json.dumps(td)
