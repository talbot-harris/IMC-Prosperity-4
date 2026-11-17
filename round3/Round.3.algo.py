"""
IMC Prosperity Round 3 — Strategy V.4

V.3 failure analysis (log 376498):
  HP  (-2515): Market-making with slow EMA anchor (9990) keeps accumulating
               positions on the wrong side of random oscillations. When HP
               swings 30+ units, our 15-unit passive quotes pile up inventory
               in the trend direction instead of fading it.
  VFE (+1703): Works. Keep it.
  VEV  (+41) : Spread-making earns tiny amounts. No real alpha — we earn
               1-2 units per fill but have no view on direction.

V.4 strategy:

  HP:  Passive MEAN-REVERSION only. Quote only the side that bets on
       reversion to the long-run anchor (~9990). When price > EMA, only
       post asks (short, expecting fall). When price < EMA, only post bids
       (long, expecting rise). Never quote the trend direction. Tiny size (5
       units) limits inventory risk on any individual swing.

  VFE: Keep V.3 market-making (it works).

  VEV: DUAL MEAN-REVERSION (VFE + IV).
       Fair value = BS(EMA_VFE, K, T, EMA_IV) where:
         EMA_VFE  — slow EMA of VFE price (alpha=0.01, half-life ~70 ticks).
                    When VFE falls below EMA, calls are priced HIGHER than
                    the raw-price BS would suggest (VFE will revert up).
         EMA_IV   — slow EMA of calibrated implied vol (alpha=0.005).
                    When current IV spikes above EMA_IV, options are priced
                    above our fair value → SELL. When IV dips → BUY.
       These two signals combine: e.g. VFE below mean AND IV low → strong
       BUY for OTM calls. VFE above mean AND IV high → strong SELL.

       Execution:
         Aggressive: lift/hit when |market_mid - fair| > edge_take (4 units).
                     Capped at 30 units per aggressive take.
         Passive:    quote inside market spread (bb+1 / ba-1) with position
                     skew, same as V.3 (limits size to VEV_PASSIVE_SIZE).
       Skip K ≤ 4500 (delta≈1, spread≈0, no edge).

IPR / ACO unchanged from V.1.
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json


# ─── Black-Scholes helpers ────────────────────────────────────────────────────

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
    """Black-Scholes European call. T in days, sigma in daily log-return vol."""
    if T <= 0.0:             return max(S - K, 0.0)
    if sigma <= 0.0 or S <= 0.0: return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)


def _implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    """Bisection for IV. Returns None if price ≤ intrinsic or T ≤ 0."""
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


# ─────────────────────────────────────────────────────────────────────────────

class Trader:

    POSITION_LIMITS: Dict[str, int] = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM":    80,
        "HYDROGEL_PACK":        200,
        "VELVETFRUIT_EXTRACT":  200,
        **{f"VEV_{s}": 300 for s in [4000, 4500, 5000, 5100, 5200,
                                      5300, 5400, 5500, 6000, 6500]},
    }

    # ── Round timing ──────────────────────────────────────────────────────────
    TICKS_PER_DAY = 100_000
    TTE_START     = 5.0

    # ── IPR (V.1, unchanged) ──────────────────────────────────────────────────
    IPR_FAST_ALPHA       = 0.05
    IPR_SLOW_ALPHA       = 0.01
    IPR_SIGNAL_THRESHOLD = 10.0

    # ── ACO (V.1, unchanged) ──────────────────────────────────────────────────
    ACO_FAIR_VALUE         = 10_000
    ACO_TAKE_THRESHOLD     = 1.5
    ACO_HALF_SPREAD        = 10
    ACO_POS_SKEW           = 0.15
    ACO_POS_SPREAD_PENALTY = 0.3
    ACO_AGGRESSIVE_REST    = 1

    # ── HYDROGEL_PACK: passive mean-reversion ─────────────────────────────────
    # Quote ONLY on the side that profits from reversion to HP_EMA_ANCHOR.
    # Tiny size so no single swing creates catastrophic inventory.
    HP_EMA_ALPHA   = 0.001    # very slow — anchors near historical 9990
    HP_FAIR_INIT   = 9990.0
    HP_PASSIVE_SIZE = 5       # units per one-sided quote

    # ── VELVETFRUIT_EXTRACT: spread market-making (V.3, working) ─────────────
    VFE_MM_EMA_ALPHA    = 0.005   # for market-making fair value
    VFE_MM_FAIR_INIT    = 5260.0
    VFE_MM_TAKE_THRESH  = 2.5
    VFE_MM_HALF_SPREAD  = 1.0
    VFE_MM_PASSIVE_SIZE = 15
    VFE_MM_POS_SKEW_SCL = 1.0

    # ── VFE slow EMA: for options mean-reversion pricing ─────────────────────
    VFE_MR_EMA_ALPHA = 0.01     # half-life ~70 ticks — captures local mean
    VFE_MR_FAIR_INIT = 5260.0

    # ── VEV Options: dual mean-reversion (VFE + IV) ───────────────────────────
    # IV calibration
    VEV_IV_PRIOR   = 0.014      # empirically correct prior (from log analysis)
    VEV_IV_MIN     = 0.005
    VEV_IV_MAX     = 0.10
    VEV_IV_EMA_ALPHA = 0.005    # half-life ~140 ticks — stable mean for IV MR

    # Strikes for IV calibration (near-ATM, highest IV sensitivity)
    VEV_CALIB_STRIKES = (5200, 5300)

    # Strikes to skip (delta ≈ 1, spread ≈ 0)
    VEV_SKIP_STRIKES  = {4000, 4500}

    # Execution thresholds
    VEV_EDGE_TAKE      = 4.0    # aggressive take when |fair − market_mid| > this
    VEV_AGG_MAX_SIZE   = 30     # cap on aggressive take per tick
    VEV_PASSIVE_SIZE   = 10     # passive quote size (inside market spread)

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result      = {}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        # ── Seed state on first tick ──────────────────────────────────────────
        if "hp_ema"     not in td: td["hp_ema"]     = self.HP_FAIR_INIT
        if "vfe_mm_ema" not in td: td["vfe_mm_ema"] = self.VFE_MM_FAIR_INIT
        if "vfe_mr_ema" not in td: td["vfe_mr_ema"] = self.VFE_MR_FAIR_INIT
        if "vev_iv_ema" not in td: td["vev_iv_ema"] = self.VEV_IV_PRIOR

        # ── Time to expiry ────────────────────────────────────────────────────
        ts = state.timestamp
        T  = max(0.0, self.TTE_START - ts / self.TICKS_PER_DAY)

        # ── First pass: HP, VFE, IPR, ACO ────────────────────────────────────
        vfe_mid = td.get("last_vfe_mid", self.VFE_MR_FAIR_INIT)

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            mid = self._mid(od)

            if product == "INTARIAN_PEPPER_ROOT":
                result[product], td = self._ipr(od, pos, td)

            elif product == "ASH_COATED_OSMIUM":
                result[product] = self._aco(od, pos)

            elif product == "HYDROGEL_PACK":
                if mid > 0:
                    td["hp_ema"] += self.HP_EMA_ALPHA * (mid - td["hp_ema"])
                result[product] = self._hp_mean_revert(od, pos, td["hp_ema"])

            elif product == "VELVETFRUIT_EXTRACT":
                if mid > 0:
                    td["vfe_mm_ema"] += self.VFE_MM_EMA_ALPHA * (mid - td["vfe_mm_ema"])
                    td["vfe_mr_ema"] += self.VFE_MR_EMA_ALPHA * (mid - td["vfe_mr_ema"])
                    vfe_mid = mid
                    td["last_vfe_mid"] = vfe_mid
                result[product] = self._vfe_market_make(od, pos, td["vfe_mm_ema"])

            elif product.startswith("VEV_"):
                result[product] = []   # filled in second pass

            else:
                result[product] = []

        # ── IV calibration and EMA update ─────────────────────────────────────
        iv_ema = td["vev_iv_ema"]
        iv_samples = []
        for k in self.VEV_CALIB_STRIKES:
            prod = f"VEV_{k}"
            if prod in state.order_depths and vfe_mid > 0 and T > 0:
                c_mid = self._mid(state.order_depths[prod])
                if c_mid > 0:
                    iv_c = _implied_vol(c_mid, vfe_mid, float(k), T)
                    if iv_c is not None:
                        iv_samples.append(iv_c)

        if iv_samples:
            raw_iv = sum(iv_samples) / len(iv_samples)
            raw_iv = max(self.VEV_IV_MIN, min(self.VEV_IV_MAX, raw_iv))
            # EMA-smooth with slow alpha → this is our "mean IV" for MR trading
            iv_ema += self.VEV_IV_EMA_ALPHA * (raw_iv - iv_ema)

        td["vev_iv_ema"] = iv_ema

        # ── Second pass: VEV options with dual MR fair value ─────────────────
        # fair = BS(EMA_VFE, K, T, EMA_IV)
        # Deviation from market → alpha signal (VFE and IV both mean-reverting)
        vfe_mr = td["vfe_mr_ema"]   # mean-reverting VFE fair value

        for product, od in state.order_depths.items():
            if not product.startswith("VEV_"):
                continue
            try:
                strike = int(product[4:])
            except ValueError:
                continue
            if strike in self.VEV_SKIP_STRIKES:
                result[product] = []
                continue
            pos = state.position.get(product, 0)
            result[product] = self._vev_option(
                product, od, pos, vfe_mr, vfe_mid, strike, T, iv_ema
            )

        return result, conversions, json.dumps(td)

    # ──────────────────────────── Utilities ──────────────────────────────────

    @staticmethod
    def _mid(od: OrderDepth) -> float:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb or ba or 0)

    # ──────────────────────────── Strategies ─────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, td: dict) -> Tuple[List[Order], dict]:
        """Dual-EMA momentum: long by default, flat on confirmed downtrend."""
        orders = []
        limit  = self.POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        mid    = self._mid(od)
        if mid == 0:
            return orders, td

        fast  = td.get("ipr_fast", mid)
        slow  = td.get("ipr_slow", mid)
        fast += self.IPR_FAST_ALPHA * (mid - fast)
        slow += self.IPR_SLOW_ALPHA * (mid - slow)
        td["ipr_fast"] = fast
        td["ipr_slow"] = slow

        target = limit if (fast - slow) > -self.IPR_SIGNAL_THRESHOLD else 0
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None

        if target > pos:
            room = target - pos
            for ap in sorted(od.sell_orders):
                if room <= 0: break
                q = min(-od.sell_orders[ap], room)
                if q > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ap, q))
                    room -= q
            if room > 0 and bb is not None:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bb + 1, room))
        elif target < pos:
            room = pos - target
            for bp in sorted(od.buy_orders, reverse=True):
                if room <= 0: break
                q = min(od.buy_orders[bp], room)
                if q > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bp, -q))
                    room -= q
            if room > 0 and ba is not None:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ba - 1, -room))

        return orders, td

    def _aco(self, od: OrderDepth, pos: int) -> List[Order]:
        """Position-skewed market-making (ASH_COATED_OSMIUM)."""
        orders    = []
        limit     = self.POSITION_LIMITS["ASH_COATED_OSMIUM"]
        if not self._mid(od):
            return orders
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None

        adj_fv    = self.ACO_FAIR_VALUE - pos * self.ACO_POS_SKEW
        buy_room  = limit - pos
        sell_room = limit + pos

        if od.sell_orders and buy_room > 0:
            for ap in sorted(od.sell_orders):
                if buy_room <= 0: break
                if ap < adj_fv - self.ACO_TAKE_THRESHOLD:
                    q = min(-od.sell_orders[ap], buy_room)
                    if q > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ap, q))
                        buy_room -= q

        if od.buy_orders and sell_room > 0:
            for bp in sorted(od.buy_orders, reverse=True):
                if sell_room <= 0: break
                if bp > adj_fv + self.ACO_TAKE_THRESHOLD:
                    q = min(od.buy_orders[bp], sell_room)
                    if q > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bp, -q))
                        sell_room -= q

        hs         = self.ACO_HALF_SPREAD + abs(pos) * self.ACO_POS_SPREAD_PENALTY
        buy_price  = math.floor(adj_fv - hs)
        sell_price = math.ceil(adj_fv + hs)
        if bb is not None and buy_price  <= bb: buy_price  = bb + (1 if self.ACO_AGGRESSIVE_REST else 0)
        if ba is not None and sell_price >= ba: sell_price = ba - (1 if self.ACO_AGGRESSIVE_REST else 0)
        if buy_price >= sell_price: buy_price = sell_price - 1

        if buy_room  > 0: orders.append(Order("ASH_COATED_OSMIUM", buy_price,   buy_room))
        if sell_room > 0: orders.append(Order("ASH_COATED_OSMIUM", sell_price, -sell_room))
        return orders

    def _hp_mean_revert(
        self, od: OrderDepth, pos: int, ema: float,
    ) -> List[Order]:
        """
        HYDROGEL_PACK: one-sided passive mean-reversion.

        Only quotes the direction that profits from reversion toward EMA.
        Never takes aggressively — we wait for the market to come to us.
        Tiny size (HP_PASSIVE_SIZE) limits inventory on any single swing.
        """
        orders = []
        limit  = self.POSITION_LIMITS["HYDROGEL_PACK"]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None

        mid = self._mid(od)
        if mid == 0 or bb is None or ba is None:
            return orders

        buy_room  = limit - pos
        sell_room = limit + pos

        if mid > ema:
            # Price is above fair → expect it to fall → passive SELL only
            # Quote at ba-1 to improve market ask, earning spread on the way down
            ask_price = ba - 1
            if ask_price > bb and sell_room > 0:
                orders.append(Order("HYDROGEL_PACK", ask_price,
                                    -min(self.HP_PASSIVE_SIZE, sell_room)))
        else:
            # Price is below fair → expect it to rise → passive BUY only
            bid_price = bb + 1
            if bid_price < ba and buy_room > 0:
                orders.append(Order("HYDROGEL_PACK", bid_price,
                                    min(self.HP_PASSIVE_SIZE, buy_room)))

        return orders

    def _vfe_market_make(
        self, od: OrderDepth, pos: int, fair: float,
    ) -> List[Order]:
        """
        VELVETFRUIT_EXTRACT: inside-the-spread market-making (V.3 approach,
        working at +1703). Uses EMA fair value with position skewing.
        """
        orders    = []
        limit     = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        buy_room  = limit - pos
        sell_room = limit + pos

        if bb is None or ba is None:
            return orders

        # Aggressive takes
        if od.sell_orders and buy_room > 0:
            for ap in sorted(od.sell_orders):
                if ap >= fair - self.VFE_MM_TAKE_THRESH: break
                q = min(-od.sell_orders[ap], buy_room)
                if q > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", ap, q))
                    buy_room -= q

        if od.buy_orders and sell_room > 0:
            for bp in sorted(od.buy_orders, reverse=True):
                if bp <= fair + self.VFE_MM_TAKE_THRESH: break
                q = min(od.buy_orders[bp], sell_room)
                if q > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", bp, -q))
                    sell_room -= q

        # Passive quotes with position skew
        skew      = (pos / limit) * self.VFE_MM_POS_SKEW_SCL
        bid_price = math.floor(fair - self.VFE_MM_HALF_SPREAD - skew)
        ask_price = math.ceil (fair + self.VFE_MM_HALF_SPREAD - skew)
        if ba is not None and bid_price >= ba: bid_price = ba - 1
        if bb is not None and ask_price <= bb: ask_price = bb + 1
        if bid_price >= ask_price:             bid_price = ask_price - 1

        if buy_room  > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", bid_price,
                                min(self.VFE_MM_PASSIVE_SIZE, buy_room)))
        if sell_room > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", ask_price,
                                -min(self.VFE_MM_PASSIVE_SIZE, sell_room)))

        return orders

    def _vev_option(
        self,
        product: str, od: OrderDepth, pos: int,
        vfe_mr: float,     # mean-reverting VFE fair value (EMA)
        vfe_raw: float,    # actual VFE mid price (for delta sense-check)
        K: int, T: float,
        iv_ema: float,     # mean-reverting IV fair level (EMA)
    ) -> List[Order]:
        """
        VEV call option: dual mean-reversion market-making.

        fair = BS(EMA_VFE, K, T, EMA_IV)

        Signal sources:
          VFE mean reversion: vfe_mr ≠ vfe_raw →
            VFE below mean (vfe_raw < vfe_mr): calls underpriced → BUY
            VFE above mean (vfe_raw > vfe_mr): calls overpriced  → SELL

          IV mean reversion: iv_ema ≠ current_market_IV →
            Embedded naturally: market prices at current_IV, our fair uses
            iv_ema. If current IV > iv_ema, market > fair → SELL.
            If current IV < iv_ema, market < fair → BUY.

        Execution:
          Aggressive: take up to VEV_AGG_MAX_SIZE when |mid−fair|>edge_take.
          Passive:    quote bb+1 / ba-1 inside market spread.
        """
        orders    = []
        limit     = self.POSITION_LIMITS.get(product, 300)
        buy_room  = limit - pos
        sell_room = limit + pos

        if vfe_mr <= 0 or T <= 0:
            return orders

        # Our fair value uses MEAN-REVERTING estimates for both S and sigma
        fair = _bs_call(vfe_mr, K, T, iv_ema)

        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return orders

        market_mid = (bb + ba) / 2.0

        # ── Aggressive takes (mean-reversion alpha) ───────────────────────────
        if buy_room > 0 and (fair - ba) > self.VEV_EDGE_TAKE:
            # Market ask is below our fair → buy (underpriced by MR model)
            q = min(-od.sell_orders[ba],
                    min(self.VEV_AGG_MAX_SIZE, buy_room))
            if q > 0:
                orders.append(Order(product, ba, q))
                buy_room -= q

        if sell_room > 0 and (bb - fair) > self.VEV_EDGE_TAKE:
            # Market bid is above our fair → sell (overpriced by MR model)
            q = min(od.buy_orders[bb],
                    min(self.VEV_AGG_MAX_SIZE, sell_room))
            if q > 0:
                orders.append(Order(product, bb, -q))
                sell_room -= q

        # ── Passive quotes inside market spread ───────────────────────────────
        market_spread = ba - bb
        if market_spread < 2:
            return orders

        # Skew based on position AND signal direction
        pos_skew     = round((pos / limit) * (market_spread / 4.0))
        # Additional signal skew: if we think it's underpriced, lean bid;
        # if overpriced, lean ask.
        signal       = fair - market_mid          # + → buy signal, − → sell
        signal_skew  = -round(max(-1, min(1, signal / self.VEV_EDGE_TAKE)))
        skew         = pos_skew + signal_skew

        our_bid = bb + 1 - skew
        our_ask = ba - 1 - skew

        our_bid = max(bb + 1, min(our_bid, ba - 1))
        our_ask = max(bb + 1, min(our_ask, ba - 1))
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
