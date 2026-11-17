"""
IMC Prosperity Round 4 Algorithm — V.4

Root cause from V.3 submission 500649 (-52k on day 3 alone):

  V.3 fed `vfe_mr_ema` (alpha=0.005, half-life ~138 ticks) as the underlying S
  into the BS pricer for every VEV strike. VFE drifted 5295.5 -> 5253.5 over
  day 3 and the slow EMA lagged ~30 pts behind. So BS thought S=5290 while the
  market priced S=5260. Every call looked "cheap to the ask" by 30+ pts vs a
  6-pt threshold — algo lifted offers on every strike, pinned +300 on six
  strikes + +200 VFE by ts ~22k, then bled 80k ticks of unhedged delta.

Fixes:

  1. BS underlying = raw VFE mid (or 1-tick EMA, alpha=0.5). The slow
     mean-reversion EMA is removed from the option pricing path entirely.
     If we want a mean-reversion *signal* on VFE, that belongs in a separate
     directional bet, not embedded in the option fair value.

  2. Aggregate-delta governor across VFE + all VEV strikes.
     book_delta = vfe_pos + sum(vev_pos[k] * N(d1_k))
     Hard cap at +/- DELTA_CAP. Refuse any order (passive or aggressive) that
     would push |book_delta| above the cap. When stale model meets full book,
     the "edge" is fake and adding loses money.

  3. Inventory throttle: aggressive takes disabled when |pos|/limit > 0.5
     for HP, VFE, and VEVs. The signal is suspect once we're already loaded.

  4. HP aggressive takes disabled (same EMA-lag pathology that bit VFE).
     HP EMA bumped to 0.05 (was 0.01) so passive fair value tracks better.

  5. VEV sizes cut and position skew strengthened.

  6. Flow bias removed from VEV underlying. It only applies to HP/VFE fair.

  7. traderData snapshot per tick (T, IV, vfe_mid, book_delta, positions)
     so post-mortem on the next submission is not blind. V.3 sandboxLog was
     empty.
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json


# Black-Scholes ────────────────────────────────────────────────────────────────

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


def _bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 1.0 if S > K else 0.0
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return _ncdf(d1)


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

    # Round 4: 1000 ticks/day, ts 0..99900 step 100 → 100_000 tick units/day.
    # TTE = 4.0 days at round start; backtester may run any single day with
    # blank traderData → infer T from near-ATM option prices on tick 1.
    TICKS_PER_DAY = 100_000
    TTE_START     = 4.0

    # Aggregate-delta governor (VFE-equivalent units).
    # VFE delta = 1 per share; VEV delta = N(d1) per option contract.
    # Day 3 V.3 ran around +1500 effective delta when it pinned. 400 is tight.
    DELTA_CAP        = 400.0
    INV_THROTTLE_FRC = 0.5   # disable aggressive takes once |pos|/limit > this

    # HP: passive MM, no aggressive takes (EMA lag pathology bit V.3).
    HP_EMA_ALPHA   = 0.05
    HP_HALF_SPREAD = 4
    HP_SIZE        = 8
    HP_POS_SKEW    = 3.0

    # VFE: passive only, faster EMA than V.3, smaller size.
    VFE_EMA_ALPHA   = 0.05
    VFE_HALF_SPREAD = 2.0
    VFE_PASSIVE_SZ  = 6
    VFE_POS_SKEW    = 3.0

    # Underlying for BS: very fast EMA of raw VFE mid (effectively raw mid
    # with one tick of noise filtering). NOT the slow mean-reversion EMA.
    VEV_S_EMA_ALPHA = 0.5

    # VEV options.
    VEV_IV_PRIOR      = 0.017
    VEV_IV_EMA_ALPHA  = 0.05
    VEV_EDGE_TAKE     = 8.0    # higher bar than V.3's 6.0 — fewer false takes
    VEV_AGG_MAX_SIZE  = 10     # cut from 20
    VEV_PASSIVE_SIZE  = 8      # cut from 15
    VEV_CALIB_STRIKES = (5000, 5100, 5200, 5300)

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

        # Day boundary: ts wraps to 0 → calibration anchor decrements 1 day.
        last_ts = td.get("last_ts", -1)
        if ts == 0 and last_ts > 0:
            if "T_calib" in td:
                td["T_calib"]    = max(0.0, td["T_calib"] - 1.0)
                td["T_calib_ts"] = 0
        td["last_ts"] = ts

        td.setdefault("hp_ema",     None)
        td.setdefault("vfe_ema",    None)
        td.setdefault("vev_s_ema",  None)
        td.setdefault("vev_iv_ema", self.VEV_IV_PRIOR)
        td.setdefault("flow",       {})

        # Counterparty flow EMA per product.
        flow_ema: Dict[str, float] = td["flow"]
        for product, trades in state.market_trades.items():
            net = 0
            for trade in trades:
                if trade.buyer in self.MARK_BOTS:
                    net += trade.quantity
                elif trade.seller in self.MARK_BOTS:
                    net -= trade.quantity
            prev = flow_ema.get(product, 0.0)
            flow_ema[product] = prev + self.FLOW_ALPHA * (net - prev)
        td["flow"] = flow_ema

        # First pass: update price EMAs and capture mids.
        mids: Dict[str, float] = {}
        for product, od in state.order_depths.items():
            mid = self._mid(od)
            if mid > 0:
                mids[product] = mid

        vfe_mid = mids.get("VELVETFRUIT_EXTRACT", td.get("last_vfe_mid", 0.0))
        if vfe_mid > 0:
            td["last_vfe_mid"] = vfe_mid
            if td["vev_s_ema"] is None:
                td["vev_s_ema"] = vfe_mid
            else:
                td["vev_s_ema"] += self.VEV_S_EMA_ALPHA * (vfe_mid - td["vev_s_ema"])

        hp_mid = mids.get("HYDROGEL_PACK", 0.0)
        if hp_mid > 0:
            if td["hp_ema"] is None:
                td["hp_ema"] = hp_mid
            else:
                td["hp_ema"] += self.HP_EMA_ALPHA * (hp_mid - td["hp_ema"])

        if vfe_mid > 0:
            if td["vfe_ema"] is None:
                td["vfe_ema"] = vfe_mid
            else:
                td["vfe_ema"] += self.VFE_EMA_ALPHA * (vfe_mid - td["vfe_ema"])

        # Bootstrap T from market on tick 1 (V.3 logic, kept intact).
        S_bs = td["vev_s_ema"] or vfe_mid
        if "T_calib" not in td and S_bs > 0:
            calib = []
            for k in self.VEV_CALIB_STRIKES:
                prod = f"VEV_{k}"
                if prod in mids:
                    calib.append((float(k), mids[prod]))
            if calib:
                T_est, sig_est = 2.0, self.VEV_IV_PRIOR
                for _ in range(6):
                    ivs = [_implied_vol(c, S_bs, k, T_est) for k, c in calib]
                    ivs = [x for x in ivs if x is not None]
                    if ivs:
                        sig_est = max(0.004, min(0.15, sum(ivs) / len(ivs)))
                    ttes = [_infer_tte(c, S_bs, k, sig_est) for k, c in calib]
                    ttes = [x for x in ttes if x is not None]
                    if ttes:
                        T_est = max(0.01, min(self.TTE_START, sum(ttes) / len(ttes)))
                td["T_calib"]    = T_est
                td["T_calib_ts"] = ts
                td["vev_iv_ema"] = sig_est
            else:
                td["T_calib"]    = 2.0
                td["T_calib_ts"] = ts

        if "T_calib" in td:
            elapsed = (ts - td["T_calib_ts"]) / self.TICKS_PER_DAY
            T = max(0.0, td["T_calib"] - elapsed)
        else:
            T = max(0.0, self.TTE_START - ts / self.TICKS_PER_DAY)

        # IV calibration vs corrected T (raw VFE mid as underlying).
        iv_ema = td["vev_iv_ema"]
        if T > 0 and S_bs > 0:
            iv_samples = []
            for k in self.VEV_CALIB_STRIKES:
                prod = f"VEV_{k}"
                if prod in mids:
                    iv_c = _implied_vol(mids[prod], S_bs, float(k), T)
                    if iv_c is not None:
                        iv_samples.append(iv_c)
            if iv_samples:
                raw_iv = max(0.004, min(0.15, sum(iv_samples) / len(iv_samples)))
                iv_ema += self.VEV_IV_EMA_ALPHA * (raw_iv - iv_ema)
        td["vev_iv_ema"] = iv_ema

        # Aggregate book delta (VFE-equivalent units).
        vfe_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        book_delta = float(vfe_pos)
        for k in self.POSITION_LIMITS:
            if not k.startswith("VEV_"):
                continue
            try:
                strike = int(k[4:])
            except ValueError:
                continue
            p = state.position.get(k, 0)
            if p == 0 or S_bs <= 0 or T <= 0:
                continue
            book_delta += p * _bs_call_delta(S_bs, float(strike), T, iv_ema)

        # Helper closures over the governor.
        def buy_allowed(extra: float) -> bool:
            return book_delta + extra <= self.DELTA_CAP

        def sell_allowed(extra: float) -> bool:
            return book_delta - extra >= -self.DELTA_CAP

        # HP: passive MM only. No aggressive takes.
        if "HYDROGEL_PACK" in state.order_depths:
            od  = state.order_depths["HYDROGEL_PACK"]
            pos = state.position.get("HYDROGEL_PACK", 0)
            fair = (td["hp_ema"] or self._mid(od)) + \
                   flow_ema.get("HYDROGEL_PACK", 0.0) * self.FLOW_SCALE
            result["HYDROGEL_PACK"] = self._hp_passive(od, pos, fair)

        # VFE: passive MM only, governor-aware.
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            od  = state.order_depths["VELVETFRUIT_EXTRACT"]
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            fair = (td["vfe_ema"] or self._mid(od)) + \
                   flow_ema.get("VELVETFRUIT_EXTRACT", 0.0) * self.FLOW_SCALE
            result["VELVETFRUIT_EXTRACT"] = self._vfe_passive(
                od, pos, fair, buy_allowed, sell_allowed
            )

        # VEV options: BS fair on raw VFE mid, governor-aware.
        for product, od in state.order_depths.items():
            if not product.startswith("VEV_"):
                continue
            try:
                strike = int(product[4:])
            except ValueError:
                result[product] = []
                continue
            pos = state.position.get(product, 0)
            result[product] = self._vev_option(
                product, od, pos, S_bs, strike, T, iv_ema,
                buy_allowed, sell_allowed,
            )

        # Snapshot for post-mortem (visible in logs as printed traderData).
        td["debug"] = {
            "ts": ts,
            "T":  round(T, 4),
            "iv": round(iv_ema, 5),
            "S":  round(S_bs, 2),
            "vfe_mid": round(vfe_mid, 2),
            "book_delta": round(book_delta, 1),
            "vfe_pos": vfe_pos,
        }

        return result, conversions, json.dumps(td)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _mid(od: OrderDepth) -> float:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb or ba or 0)

    # ── Strategies ────────────────────────────────────────────────────────────

    def _hp_passive(self, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        orders = []
        limit  = self.POSITION_LIMITS["HYDROGEL_PACK"]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return orders

        buy_room  = limit - pos
        sell_room = limit + pos

        skew      = (pos / limit) * self.HP_POS_SKEW
        bid_price = math.floor(fair - self.HP_HALF_SPREAD - skew)
        ask_price = math.ceil( fair + self.HP_HALF_SPREAD - skew)
        if bid_price >= ba: bid_price = ba - 1
        if ask_price <= bb: ask_price = bb + 1
        if bid_price >= ask_price: bid_price = ask_price - 1

        if buy_room > 0:
            orders.append(Order("HYDROGEL_PACK", bid_price,
                                min(self.HP_SIZE, buy_room)))
        if sell_room > 0:
            orders.append(Order("HYDROGEL_PACK", ask_price,
                                -min(self.HP_SIZE, sell_room)))
        return orders

    def _vfe_passive(self, od: OrderDepth, pos: int, fair: float,
                     buy_allowed, sell_allowed) -> List[Order]:
        orders = []
        limit  = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return orders

        buy_room  = limit - pos
        sell_room = limit + pos

        skew      = (pos / limit) * self.VFE_POS_SKEW
        bid_price = math.floor(fair - self.VFE_HALF_SPREAD - skew)
        ask_price = math.ceil( fair + self.VFE_HALF_SPREAD - skew)
        if bid_price >= ba: bid_price = ba - 1
        if ask_price <= bb: ask_price = bb + 1
        if bid_price >= ask_price: bid_price = ask_price - 1

        bid_qty = min(self.VFE_PASSIVE_SZ, buy_room)
        ask_qty = min(self.VFE_PASSIVE_SZ, sell_room)

        # Governor: refuse the side that would push delta past the cap.
        if bid_qty > 0 and not buy_allowed(bid_qty):
            bid_qty = 0
        if ask_qty > 0 and not sell_allowed(ask_qty):
            ask_qty = 0

        if bid_qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", ask_price, -ask_qty))
        return orders

    def _vev_option(
        self, product: str, od: OrderDepth, pos: int,
        S: float, K: int, T: float, iv: float,
        buy_allowed, sell_allowed,
    ) -> List[Order]:
        orders = []
        limit  = self.POSITION_LIMITS.get(product, 300)
        if S <= 0 or T <= 0:
            return orders

        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None:
            return orders

        fair  = _bs_call(S, float(K), T, iv)
        delta = _bs_call_delta(S, float(K), T, iv)
        market_mid    = (bb + ba) / 2.0
        market_spread = ba - bb

        buy_room  = limit - pos
        sell_room = limit + pos
        loaded_long  = pos >  self.INV_THROTTLE_FRC * limit
        loaded_short = pos < -self.INV_THROTTLE_FRC * limit

        # Aggressive takes — only when (a) edge exceeds threshold, (b) not
        # already loaded that direction, and (c) governor permits the
        # added delta.
        if buy_room > 0 and not loaded_long and (fair - ba) > self.VEV_EDGE_TAKE:
            avail = -od.sell_orders[ba]
            q = min(avail, self.VEV_AGG_MAX_SIZE, buy_room)
            while q > 0 and not buy_allowed(q * delta):
                q -= 1
            if q > 0:
                orders.append(Order(product, ba, q))
                buy_room -= q

        if sell_room > 0 and not loaded_short and (bb - fair) > self.VEV_EDGE_TAKE:
            avail = od.buy_orders[bb]
            q = min(avail, self.VEV_AGG_MAX_SIZE, sell_room)
            while q > 0 and not sell_allowed(q * delta):
                q -= 1
            if q > 0:
                orders.append(Order(product, bb, -q))
                sell_room -= q

        # Passive quotes inside the spread.
        if market_spread < 2:
            return orders

        signal      = fair - market_mid
        signal_skew = -round(max(-2, min(2, signal / self.VEV_EDGE_TAKE)))
        pos_skew    = round((pos / limit) * (market_spread / 2.0))
        skew        = signal_skew + pos_skew

        our_bid = bb + 1 - skew
        our_ask = ba - 1 - skew
        our_bid = max(bb + 1, min(our_bid, ba - 2))
        our_ask = max(bb + 2, min(our_ask, ba - 1))
        if our_bid >= our_ask:
            our_ask = our_bid + 1
        if our_ask >= ba:
            return orders

        bid_qty = min(self.VEV_PASSIVE_SIZE, buy_room)
        ask_qty = min(self.VEV_PASSIVE_SIZE, sell_room)

        # Governor on passive side too — a passive bid that fills also
        # adds delta. Trim the qty down until it fits.
        while bid_qty > 0 and not buy_allowed(bid_qty * delta):
            bid_qty -= 1
        while ask_qty > 0 and not sell_allowed(ask_qty * delta):
            ask_qty -= 1

        # If we're already past the inventory throttle on a side, kill that
        # side entirely (let the other side bleed inventory off).
        if loaded_long:
            bid_qty = 0
        if loaded_short:
            ask_qty = 0

        if bid_qty > 0:
            orders.append(Order(product, our_bid, bid_qty))
        if ask_qty > 0:
            orders.append(Order(product, our_ask, -ask_qty))

        return orders
