"""
IMC Prosperity Round 4 Algorithm — V.3

Fixes vs V.2 (which scored -66,539):
  1. TICKS_PER_DAY = 100_000  (Round 4 days are 1000 ticks: ts 0..99900 step 100)
     V.2 used 1_000_000, making T change by only 0.1 over the whole day instead of 1.0.
  2. Market-inferred TTE bootstrap on first tick.
     Backtester starts on Day 3 fresh (blank traderData).  With day-counting, algo
     thought current_day=1 → T=4.0, but actual TTE=2.0.  We now bisect T from
     near-ATM option prices, so T≈2.0 is found immediately on tick 1.
  3. VFE aggressive takes disabled (VFE_TAKE_THRESH very large).
     VFE fell 42 pts on day 3; EMA lagged badly, causing aggressive buys all day
     → accumulated max-long 200 while price kept falling.  Now passive only.
  4. VFE EMA faster (0.003 → 0.015) to reduce fair-value lag on trends.
  5. VFE passive size cut (20 → 8) to limit directional exposure.
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional
import math
import json


# ── Black-Scholes ──────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    poly = t * (0.319381530
          + t * (-0.356563782
          + t * (1.781477937
          + t * (-1.821255978
          + t *   1.330274429))))
    p = 1.0 - d * poly
    return p if x >= 0 else 1.0 - p


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)


def _implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or T <= 0.0 or S <= 0.0:
        return None
    lo, hi = 0.001, 5.0
    for _ in range(50):
        m = (lo + hi) / 2.0
        if _bs_call(S, K, T, m) < price:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2.0


def _infer_tte(price: float, S: float, K: float, sigma: float,
               lo: float = 0.01, hi: float = 6.0) -> Optional[float]:
    """Bisect for T s.t. BS(S, K, T, sigma) = price."""
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or S <= 0.0 or sigma <= 0.0:
        return None
    if _bs_call(S, K, hi, sigma) < price:
        return hi
    if _bs_call(S, K, lo, sigma) > price:
        return lo
    for _ in range(50):
        m = (lo + hi) / 2.0
        if _bs_call(S, K, m, sigma) < price:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2.0


# ──────────────────────────────────────────────────────────────────────────────

class Trader:

    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":        200,
        "VELVETFRUIT_EXTRACT":  200,
        **{f"VEV_{s}": 300 for s in [4000, 4500, 5000, 5100, 5200,
                                      5300, 5400, 5500, 6000, 6500]},
    }

    # ── Timing ────────────────────────────────────────────────────────────────
    # Round 4 timestamps: 0..99900 step 100 per day = 1000 ticks/day.
    # TICKS_PER_DAY = 100_000.  (Round 3 had 10000 ticks/day = 1_000_000.)
    # TTE at Round 4 start = 4.0 days (competition-confirmed).
    # Backtester evaluates Day 3 only (traderData blank at start).
    # => Use market-inferred T on first tick instead of day-counting.
    TICKS_PER_DAY = 100_000
    TTE_START     = 4.0    # TTE at Day 1 tick 0; Day 3 tick 0 should be 2.0

    # ── HYDROGEL_PACK: symmetric market-making ────────────────────────────────
    HP_EMA_ALPHA   = 0.01
    HP_HALF_SPREAD = 5
    HP_SIZE        = 10
    HP_TAKE_THRESH = 5
    HP_TAKE_SIZE   = 8
    HP_POS_SKEW    = 2.0

    # ── VELVETFRUIT_EXTRACT: passive market-making only ───────────────────────
    # Aggressive takes are OFF: EMA lags on trending markets and causes
    # directional accumulation that wipes out all spread income.
    VFE_EMA_ALPHA   = 0.015   # faster EMA to reduce fair-value lag
    VFE_TAKE_THRESH = 999     # effectively disabled
    VFE_HALF_SPREAD = 2.0
    VFE_PASSIVE_SZ  = 8
    VFE_POS_SKEW    = 2.0     # stronger lean to stay near flat

    # ── VEV Options: BS mean-reversion ───────────────────────────────────────
    VEV_IV_PRIOR      = 0.017
    VEV_IV_EMA_ALPHA  = 0.05    # faster IV calibration
    VEV_VFE_MR_ALPHA  = 0.005
    VEV_EDGE_TAKE     = 6.0
    VEV_AGG_MAX_SIZE  = 20
    VEV_PASSIVE_SIZE  = 15
    # Near-ATM strikes used for calibration
    VEV_CALIB_STRIKES = (5000, 5100, 5200, 5300)

    # ── Counterparty flow ─────────────────────────────────────────────────────
    MARK_BOTS  = {"Mark 01", "Mark 14", "Mark 22", "Mark 38",
                  "Mark 49", "Mark 55", "Mark 67"}
    FLOW_ALPHA = 0.1
    FLOW_SCALE = 0.3

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result      = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp

        # ── Day-boundary detection ────────────────────────────────────────────
        last_ts = td.get("last_ts", -1)
        if ts == 0 and last_ts > 0:
            # New day: T_calib decrements by 1 day
            if "T_calib" in td:
                td["T_calib"] = td["T_calib"] - 1.0
                td["T_calib_ts"] = 0
        td["last_ts"] = ts

        # ── Seed persistent state ─────────────────────────────────────────────
        if "hp_ema"     not in td: td["hp_ema"]     = None
        if "vfe_ema"    not in td: td["vfe_ema"]     = None
        if "vfe_mr_ema" not in td: td["vfe_mr_ema"]  = None
        if "vev_iv_ema" not in td: td["vev_iv_ema"]  = self.VEV_IV_PRIOR
        if "flow"       not in td: td["flow"]         = {}

        # ── Counterparty flow ─────────────────────────────────────────────────
        flow_ema: Dict[str, float] = td["flow"]
        for product, trades in state.market_trades.items():
            net = 0
            for trade in trades:
                qty = trade.quantity
                if trade.buyer in self.MARK_BOTS:
                    net += qty
                elif trade.seller in self.MARK_BOTS:
                    net -= qty
            prev = flow_ema.get(product, 0.0)
            flow_ema[product] = prev + self.FLOW_ALPHA * (net - prev)
        td["flow"] = flow_ema

        # ── Price discovery: HP, VFE ──────────────────────────────────────────
        vfe_mid = td.get("last_vfe_mid", 5248.0)

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            mid = self._mid(od)
            flow_bias = flow_ema.get(product, 0.0) * self.FLOW_SCALE

            if product == "HYDROGEL_PACK":
                if mid > 0:
                    if td["hp_ema"] is None:
                        td["hp_ema"] = mid
                    else:
                        td["hp_ema"] += self.HP_EMA_ALPHA * (mid - td["hp_ema"])
                fair = (td["hp_ema"] or mid) + flow_bias
                result[product] = self._hp_mm(od, pos, fair)

            elif product == "VELVETFRUIT_EXTRACT":
                if mid > 0:
                    if td["vfe_ema"] is None:
                        td["vfe_ema"]    = mid
                        td["vfe_mr_ema"] = mid
                    else:
                        td["vfe_ema"]    += self.VFE_EMA_ALPHA    * (mid - td["vfe_ema"])
                        td["vfe_mr_ema"] += self.VEV_VFE_MR_ALPHA * (mid - td["vfe_mr_ema"])
                    vfe_mid = mid
                    td["last_vfe_mid"] = mid
                fair = (td["vfe_ema"] or mid) + flow_bias
                result[product] = self._vfe_mm(od, pos, fair)

            elif product.startswith("VEV_"):
                result[product] = []
            else:
                result[product] = []

        # ── Bootstrap T and IV from market on first tick ──────────────────────
        # Backtester may start fresh on any day with blank traderData.
        # We infer the current TTE from near-ATM option prices rather than
        # relying on day-counting, which fails when traderData is reset.
        iv_ema = td["vev_iv_ema"]

        if "T_calib" not in td and vfe_mid > 0:
            calib_data = []
            for k in self.VEV_CALIB_STRIKES:
                prod = f"VEV_{k}"
                if prod in state.order_depths:
                    c_mid = self._mid(state.order_depths[prod])
                    if c_mid > 0:
                        calib_data.append((float(k), c_mid))

            if calib_data:
                T_est    = 2.0   # initial guess
                sig_est  = self.VEV_IV_PRIOR
                for _ in range(6):
                    ivs = [_implied_vol(c, vfe_mid, k, T_est) for k, c in calib_data]
                    ivs = [x for x in ivs if x is not None]
                    if ivs:
                        sig_est = max(0.004, min(0.15, sum(ivs) / len(ivs)))
                    ttes = [_infer_tte(c, vfe_mid, k, sig_est) for k, c in calib_data]
                    ttes = [x for x in ttes if x is not None]
                    if ttes:
                        T_est = max(0.01, min(self.TTE_START, sum(ttes) / len(ttes)))

                td["T_calib"]    = T_est
                td["T_calib_ts"] = ts
                td["vev_iv_ema"] = sig_est
                iv_ema           = sig_est
            else:
                # No option data yet; default to TTE=2.0 (Day 3 assumption)
                td["T_calib"]    = 2.0
                td["T_calib_ts"] = ts

        # ── Compute current TTE from calibration anchor ───────────────────────
        if "T_calib" in td:
            elapsed = (ts - td["T_calib_ts"]) / self.TICKS_PER_DAY
            T = max(0.0, td["T_calib"] - elapsed)
        else:
            T = max(0.0, self.TTE_START - ts / self.TICKS_PER_DAY)

        # ── IV calibration (near-ATM, using corrected T) ──────────────────────
        if T > 0 and vfe_mid > 0:
            iv_samples = []
            for k in self.VEV_CALIB_STRIKES:
                prod = f"VEV_{k}"
                if prod in state.order_depths:
                    c_mid = self._mid(state.order_depths[prod])
                    if c_mid > 0:
                        iv_c = _implied_vol(c_mid, vfe_mid, float(k), T)
                        if iv_c is not None:
                            iv_samples.append(iv_c)
            if iv_samples:
                raw_iv = sum(iv_samples) / len(iv_samples)
                raw_iv = max(0.004, min(0.15, raw_iv))
                iv_ema += self.VEV_IV_EMA_ALPHA * (raw_iv - iv_ema)

        td["vev_iv_ema"] = iv_ema

        # ── VEV options ───────────────────────────────────────────────────────
        vfe_mr = td["vfe_mr_ema"] or vfe_mid
        vfe_flow_bias = flow_ema.get("VELVETFRUIT_EXTRACT", 0.0) * self.FLOW_SCALE

        for product, od in state.order_depths.items():
            if not product.startswith("VEV_"):
                continue
            try:
                strike = int(product[4:])
            except ValueError:
                continue
            pos = state.position.get(product, 0)
            result[product] = self._vev_option(
                product, od, pos,
                vfe_mr + vfe_flow_bias,
                vfe_mid,
                strike, T, iv_ema
            )

        return result, conversions, json.dumps(td)

    # ─────────────────────────────────────── Utilities ────────────────────────

    @staticmethod
    def _mid(od: OrderDepth) -> float:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb or ba or 0)

    # ─────────────────────────────────────── Strategies ───────────────────────

    def _hp_mm(self, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        orders    = []
        limit     = self.POSITION_LIMITS["HYDROGEL_PACK"]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return orders

        mid       = (bb + ba) / 2.0
        dev       = mid - fair
        buy_room  = limit - pos
        sell_room = limit + pos

        if self.HP_TAKE_THRESH > 0:
            if dev < -self.HP_TAKE_THRESH and buy_room > 0:
                avail = -od.sell_orders.get(ba, 0)
                q = min(self.HP_TAKE_SIZE, buy_room, avail)
                if q > 0:
                    orders.append(Order("HYDROGEL_PACK", ba, q))
                    buy_room  -= q
                    sell_room += q
            elif dev > self.HP_TAKE_THRESH and sell_room > 0:
                avail = od.buy_orders.get(bb, 0)
                q = min(self.HP_TAKE_SIZE, sell_room, avail)
                if q > 0:
                    orders.append(Order("HYDROGEL_PACK", bb, -q))
                    sell_room -= q
                    buy_room  += q

        skew      = int((pos / limit) * self.HP_POS_SKEW)
        bid_price = math.floor(fair - self.HP_HALF_SPREAD - skew)
        ask_price = math.ceil( fair + self.HP_HALF_SPREAD - skew)
        bid_price = min(bid_price, bb)
        ask_price = max(ask_price, ba)
        if bid_price >= ask_price:
            bid_price = ask_price - 1

        if buy_room  > 0:
            orders.append(Order("HYDROGEL_PACK", bid_price,
                                min(self.HP_SIZE, buy_room)))
        if sell_room > 0:
            orders.append(Order("HYDROGEL_PACK", ask_price,
                                -min(self.HP_SIZE, sell_room)))
        return orders

    def _vfe_mm(self, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        orders    = []
        limit     = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        buy_room  = limit - pos
        sell_room = limit + pos

        if bb is None or ba is None:
            return orders

        # Aggressive takes disabled: EMA lags on trending days causing losses.
        # Only passive market-making.

        skew      = (pos / limit) * self.VFE_POS_SKEW
        bid_price = math.floor(fair - self.VFE_HALF_SPREAD - skew)
        ask_price = math.ceil( fair + self.VFE_HALF_SPREAD - skew)
        if ba is not None and bid_price >= ba: bid_price = ba - 1
        if bb is not None and ask_price <= bb: ask_price = bb + 1
        if bid_price >= ask_price:             bid_price = ask_price - 1

        if buy_room  > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", bid_price,
                                min(self.VFE_PASSIVE_SZ, buy_room)))
        if sell_room > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", ask_price,
                                -min(self.VFE_PASSIVE_SZ, sell_room)))
        return orders

    def _vev_option(
        self,
        product: str, od: OrderDepth, pos: int,
        vfe_mr: float,
        vfe_raw: float,
        K: int, T: float,
        iv_ema: float,
    ) -> List[Order]:
        orders    = []
        limit     = self.POSITION_LIMITS.get(product, 300)
        buy_room  = limit - pos
        sell_room = limit + pos

        if vfe_mr <= 0 or T <= 0:
            return orders

        fair = _bs_call(vfe_mr, K, T, iv_ema)

        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return orders

        market_mid    = (bb + ba) / 2.0
        market_spread = ba - bb

        # Aggressive takes when clearly mispriced
        if buy_room > 0 and (fair - ba) > self.VEV_EDGE_TAKE:
            q = min(-od.sell_orders[ba], self.VEV_AGG_MAX_SIZE, buy_room)
            if q > 0:
                orders.append(Order(product, ba, q))
                buy_room -= q

        if sell_room > 0 and (bb - fair) > self.VEV_EDGE_TAKE:
            q = min(od.buy_orders[bb], self.VEV_AGG_MAX_SIZE, sell_room)
            if q > 0:
                orders.append(Order(product, bb, -q))
                sell_room -= q

        # Passive quotes inside spread
        if market_spread < 2:
            return orders

        signal      = fair - market_mid
        signal_skew = -round(max(-2, min(2, signal / self.VEV_EDGE_TAKE)))
        pos_skew    = round((pos / limit) * (market_spread / 4.0))
        skew        = signal_skew + pos_skew

        our_bid = bb + 1 - skew
        our_ask = ba - 1 - skew

        our_bid = max(bb + 1, min(our_bid, ba - 2))
        our_ask = max(bb + 2, min(our_ask, ba - 1))
        if our_bid >= our_ask:
            our_ask = our_bid + 1
        if our_ask >= ba:
            return orders

        if buy_room  > 0:
            orders.append(Order(product, our_bid,
                                min(self.VEV_PASSIVE_SIZE, buy_room)))
        if sell_room > 0:
            orders.append(Order(product, our_ask,
                                -min(self.VEV_PASSIVE_SIZE, sell_room)))

        return orders