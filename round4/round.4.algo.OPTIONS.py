"""
IMC Prosperity Round 4 — VEV options-only algorithm

Tradable products (calls on VFE, position limit 300 each):
  K ∈ {4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500}

Day-3 data (S = VFE mid; range 5244..5300, std 15.7, T_at_start = 2.0):

    K     spread_med  vol1_med  IV_signal     trade-bucket
    4000  21          11        none (intrinsic) DEEP-ITM-MM
    4500  16           9        weak             DEEP-ITM-MM
    5000   6          15        clean smile      ATM-MM   ← primary
    5100   4          18        clean smile      ATM-MM   ← primary
    5200   3          23        clean smile      ATM-MM   ← primary
    5300   2          20        clean smile      smile-only (no MM, spread<3)
    5400   1          21        clean smile      smile-only
    5500   1          22        zero             dead
    6000   1          22        zero             dead
    6500   1          15        zero             dead

The smile across {5000..5400} is a shallow downward parabola in
log-moneyness (coef ~ -0.56). Daily IV oscillates in 0.019..0.020.
After fitting iv(x)=a+b*x+c*x², residuals are ~1e-4 — small, but the
fitted value is a much better fair anchor than a single ATM IV.

Strategy menu (each independently gated):

  (1) SMILE-FIT FAIR VALUE PER STRIKE (the headline change vs V.4)
      Per tick, fit a quadratic IV(log(K/S)) on calibration strikes
      {5000,5100,5200,5300,5400} and evaluate fair_K = BS(S, K, T, smile_iv(K))
      for every traded strike — including the deep-ITM 4000/4500 where
      smile_iv extrapolates cleanly because intrinsic dominates.

  (2) ATM PASSIVE MARKET MAKING
      For K ∈ {5000, 5100, 5200}: quote bb+1 / ba-1 around fair_K with
      size scaled to spread. Inventory skew + smile-residual skew shift
      the quotes in the cheap/rich direction.

  (3) DEEP-ITM PASSIVE MM WITH VERTICAL OFFSET
      For K ∈ {4000, 4500}: quote inside the wide spread (bb+2 / ba-2)
      with small size. These adds are ~1.0 delta each → governed
      strictly by the global delta cap.

  (4) SMILE-RESIDUAL AGGRESSIVE TAKES
      When |market_mid - fair_K| > EDGE_TAKE_PRICE for K ∈ {5000..5300},
      lift/hit in mean-reversion direction. Throttled by per-strike
      inventory and global delta cap.

  (5) GLOBAL DELTA + VEGA GOVERNORS (V.4 lesson, expanded)
      book_delta = Σ pos_K * N(d1_K)   (in VFE-eq units)
      book_vega  = Σ pos_K * vega_K
      Refuse any order (passive or aggressive) that would push either
      governor past its cap. Without ability to hedge in VFE itself,
      this is the only thing standing between us and a stale-IV blowup.

  (6) SKIP DEAD STRIKES
      5500/6000/6500: emit nothing. No spread, no signal, just noise.

  (7) PIN PROTECTION NEAR EXPIRY
      If T < 0.25 days, halve sizes near ATM (gamma scales as 1/√T).

  (8) TTE INFERENCE FROM TICK-1 PRICES (ported from V.4)
      Backtester / single-day eval starts with blank traderData; algo
      bootstraps T by inverting BS on near-ATM strikes against an
      IV prior, then iterating with the smile fit.

This file emits NO orders for HYDROGEL_PACK or VELVETFRUIT_EXTRACT.
Combine with round.4.algo.HP.py / round.4.algo.VFE.py via a top-level
dispatcher when ready to submit a full algo.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json


# ── Black–Scholes ────────────────────────────────────────────────────────────

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = math.exp(-0.5 * x * x) / SQRT_2PI
    poly = t * (0.319381530
          + t * (-0.356563782
          + t * (1.781477937
          + t * (-1.821255978
          + t *   1.330274429))))
    p = 1.0 - d * poly
    return p if x >= 0 else 1.0 - p


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)


def _bs_d1(S: float, K: float, T: float, sigma: float) -> float:
    sv = sigma * math.sqrt(T)
    return (math.log(S / K) + 0.5 * sigma * sigma * T) / sv


def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 1.0 if S > K else 0.0
    return _ncdf(_bs_d1(S, K, T, sigma))


def _bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 0.0
    return S * math.sqrt(T) * _npdf(_bs_d1(S, K, T, sigma))


def _implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or T <= 0.0 or S <= 0.0:
        return None
    lo, hi = 1e-4, 5.0
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


def _fit_quadratic(xs: List[float], ys: List[float]
                  ) -> Optional[Tuple[float, float, float]]:
    """Closed-form least-squares quadratic on (x, y); returns (a, b, c) for
    y = a + b*x + c*x^2. None if degenerate."""
    n = len(xs)
    if n < 3:
        return None
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x*x  for x in xs)
    s3 = sum(x**3 for x in xs)
    s4 = sum(x**4 for x in xs)
    t0 = sum(ys)
    t1 = sum(x*y          for x, y in zip(xs, ys))
    t2 = sum(x*x*y        for x, y in zip(xs, ys))

    # 3x3 system: [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] @ [a,b,c] = [t0,t1,t2]
    m = [[s0, s1, s2, t0],
         [s1, s2, s3, t1],
         [s2, s3, s4, t2]]
    # Gauss elimination
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12:
            return None
        m[i], m[p] = m[p], m[i]
        for j in range(i+1, 3):
            f = m[j][i] / m[i][i]
            for k in range(i, 4):
                m[j][k] -= f * m[i][k]
    c = m[2][3] / m[2][2]
    b = (m[1][3] - m[1][2]*c) / m[1][1]
    a = (m[0][3] - m[0][1]*b - m[0][2]*c) / m[0][0]
    return a, b, c


# ──────────────────────────────────────────────────────────────────────────────

class Trader:

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    STRIKES_ALL  = [4000, 4500, 5000, 5100, 5200, 5300,
                    5400, 5500, 6000, 6500]
    STRIKES_MM   = [4000, 4500, 5000, 5100, 5200, 5300]   # we quote here
    STRIKES_FIT  = [5000, 5100, 5200, 5300, 5400]         # smile fit input
    STRIKES_DEAD = {5500, 6000, 6500}                     # never quote

    POSITION_LIMITS = {f"VEV_{k}": 300 for k in STRIKES_ALL}

    TICKS_PER_DAY = 100_000
    TTE_START     = 4.0

    # IV defaults / bounds
    IV_PRIOR = 0.0195
    IV_MIN   = 0.005
    IV_MAX   = 0.10

    # Smile EMA: blend new fit into stored coefficients to dampen tick noise
    SMILE_EMA_ALPHA = 0.20

    # Underlying EMA: short, just denoise raw mid (V.4 lesson — not slow!)
    S_EMA_ALPHA = 0.5

    # ── Trading thresholds ───────────────────────────────────────────────────
    # V1 of this algo (submission 514293) made +423 on day 3 with only 45
    # fills total across all strikes — all clustered on ~9 Mark-bot sweep
    # events. Tuning: size up, threshold down, governors looser. The big
    # safety nets (delta cap + vega cap + per-strike throttle + the
    # "block-only-if-worsening" governor logic) stay in place.

    EDGE_TAKE_PRICE      = 1.0   # was 1.5 — fire more often on smile drift
    EDGE_TAKE_IV_RESID   = 0.0005  # was 0.0008

    # Per-strike inventory throttle: kill the add-side of passive at this
    # fraction of the per-strike limit. Looser than V1 (0.50 → 0.65).
    PER_STRIKE_THROTTLE  = 0.65

    # Aggregate governor caps.
    # VEGA_CAP scale: per-contract ATM vega ≈ S*sqrt(T)*φ(d1) ≈ 5250*1.22*0.21
    # ≈ 1350 at T=1.5. 200,000 allows ~150 ATM-equivalent contracts net.
    # DELTA_CAP 300 → ~150 ITM-equivalent or ~600 ATM-equivalent net.
    DELTA_CAP            = 300.0      # was 200 — VFE-equivalent units
    VEGA_CAP             = 200_000.0  # was 80k

    # Sizes — bumped substantially. Mark bots sweep with depth ~20-30 per
    # strike per event. Sized to capture most of a sweep when flat.
    PASSIVE_SIZE_ATM     = 25        # was 8  (5000/5100/5200/5300)
    PASSIVE_SIZE_ITM     = 15        # was 4  (4000/4500)
    AGGRO_SIZE_MAX       = 15        # was 6

    # Pin-risk damper: if T < this many days, halve ATM passive size.
    PIN_T_THRESHOLD      = 0.25

    # Minimum book spread we will quote inside.
    # 2 → join the book on tight strikes (5300 with spread=2). Quoting at
    # bb / ba doesn't capture spread edge per fill but does let us
    # participate in sweep flow and accumulate small theta-positive
    # carry. Per-strike inventory throttle keeps risk bounded.
    MIN_PASSIVE_SPREAD   = 2         # was 3
    MIN_ITM_PASSIVE_SPREAD = 5

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result      = {p: [] for p in state.order_depths}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp

        # Day boundary: ts wraps to 0 → calibration anchor decrements 1 day.
        last_ts = td.get("last_ts", -1)
        if ts == 0 and last_ts > 0 and "T_calib" in td:
            td["T_calib"]    = max(0.0, td["T_calib"] - 1.0)
            td["T_calib_ts"] = 0
        td["last_ts"] = ts

        td.setdefault("s_ema",   None)
        td.setdefault("smile",   None)         # (a, b, c) tuple
        td.setdefault("iv_atm",  self.IV_PRIOR)

        # ── Mids ────────────────────────────────────────────────────────────
        mids: Dict[str, Tuple[float, int, int]] = {}   # product → (mid, bb, ba)
        for product, od in state.order_depths.items():
            bb = max(od.buy_orders)  if od.buy_orders  else None
            ba = min(od.sell_orders) if od.sell_orders else None
            if bb is not None and ba is not None:
                mids[product] = ((bb + ba) / 2.0, bb, ba)

        if self.UNDERLYING not in mids:
            return result, conversions, json.dumps(td)

        vfe_mid, _, _ = mids[self.UNDERLYING]
        if td["s_ema"] is None:
            td["s_ema"] = vfe_mid
        else:
            td["s_ema"] += self.S_EMA_ALPHA * (vfe_mid - td["s_ema"])
        S = td["s_ema"]

        # ── TTE bootstrap on first tick (when traderData is blank) ──────────
        if "T_calib" not in td:
            T_est, sig_est = 2.0, self.IV_PRIOR
            for _ in range(6):
                samples = []
                for k in (5000, 5100, 5200, 5300):
                    prod = f"VEV_{k}"
                    if prod in mids:
                        samples.append((float(k), mids[prod][0]))
                if not samples:
                    break
                ivs = [_implied_vol(c, S, k, T_est) for k, c in samples]
                ivs = [x for x in ivs if x is not None]
                if ivs:
                    sig_est = max(self.IV_MIN, min(self.IV_MAX,
                                                   sum(ivs) / len(ivs)))
                ttes = [_infer_tte(c, S, k, sig_est) for k, c in samples]
                ttes = [x for x in ttes if x is not None]
                if ttes:
                    T_est = max(0.01, min(self.TTE_START,
                                          sum(ttes) / len(ttes)))
            td["T_calib"]    = T_est
            td["T_calib_ts"] = ts
            td["iv_atm"]     = sig_est

        elapsed = (ts - td["T_calib_ts"]) / self.TICKS_PER_DAY
        T = max(0.0, td["T_calib"] - elapsed)

        # ── Smile fit ───────────────────────────────────────────────────────
        # Compute IV per fit strike, fit y = a + b*x + c*x², EMA the
        # coefficients to denoise.
        smile_xs: List[float] = []
        smile_ys: List[float] = []
        for k in self.STRIKES_FIT:
            prod = f"VEV_{k}"
            if prod not in mids:
                continue
            mid_k = mids[prod][0]
            v = _implied_vol(mid_k, S, float(k), T)
            if v is None or v < self.IV_MIN or v > self.IV_MAX:
                continue
            smile_xs.append(math.log(float(k) / S))
            smile_ys.append(v)

        smile_now = _fit_quadratic(smile_xs, smile_ys) if len(smile_xs) >= 3 \
                    else None

        if smile_now is not None:
            if td["smile"] is None:
                td["smile"] = list(smile_now)
            else:
                a_s, b_s, c_s = td["smile"]
                a_n, b_n, c_n = smile_now
                alpha = self.SMILE_EMA_ALPHA
                td["smile"] = [a_s + alpha*(a_n - a_s),
                               b_s + alpha*(b_n - b_s),
                               c_s + alpha*(c_n - c_s)]

        smile = td["smile"]
        if smile is not None:
            td["iv_atm"] = max(self.IV_MIN, min(self.IV_MAX, smile[0]))

        def smile_iv(K: float) -> float:
            if smile is None or S <= 0:
                return td["iv_atm"]
            x = math.log(K / S)
            iv = smile[0] + smile[1]*x + smile[2]*x*x
            return max(self.IV_MIN, min(self.IV_MAX, iv))

        # ── Aggregate book delta + vega ─────────────────────────────────────
        book_delta = 0.0
        book_vega  = 0.0
        for k in self.STRIKES_ALL:
            prod = f"VEV_{k}"
            p = state.position.get(prod, 0)
            if p == 0 or T <= 0:
                continue
            iv_k = smile_iv(float(k))
            book_delta += p * _bs_delta(S, float(k), T, iv_k)
            book_vega  += p * _bs_vega( S, float(k), T, iv_k)

        # Governors. When a cap is already breached, allow the side that
        # REDUCES the breach (so we can unwind). Only block sides that
        # would push us further into excess. Buy adds delta and vega;
        # sell subtracts both.
        def can_buy(extra_delta: float, extra_vega: float) -> bool:
            new_d = book_delta + extra_delta
            new_v = book_vega  + extra_vega
            if new_d > self.DELTA_CAP and new_d > book_delta:
                return False
            if abs(new_v) > self.VEGA_CAP and abs(new_v) > abs(book_vega):
                return False
            return True

        def can_sell(extra_delta: float, extra_vega: float) -> bool:
            new_d = book_delta - extra_delta
            new_v = book_vega  - extra_vega
            if new_d < -self.DELTA_CAP and new_d < book_delta:
                return False
            if abs(new_v) > self.VEGA_CAP and abs(new_v) > abs(book_vega):
                return False
            return True

        # ── Per-strike trading ──────────────────────────────────────────────
        per_strike_dbg = {}
        for k in self.STRIKES_MM:
            prod = f"VEV_{k}"
            if prod not in state.order_depths:
                continue
            od = state.order_depths[prod]
            pos = state.position.get(prod, 0)
            if prod not in mids:
                continue
            mid_k, bb, ba = mids[prod]

            iv_k    = smile_iv(float(k))
            fair    = _bs_call(S, float(k), T, iv_k)
            delta_k = _bs_delta(S, float(k), T, iv_k)
            vega_k  = _bs_vega( S, float(k), T, iv_k)

            ords, dbg = self._trade_strike(
                prod, k, od, pos, bb, ba, mid_k,
                fair, delta_k, vega_k, T,
                can_buy, can_sell,
            )
            result[prod] = ords
            per_strike_dbg[k] = dbg

        # Dead strikes: emit nothing (already initialized to []).
        # (Loop below could explicitly clear, but result was init'd that way.)

        td["debug"] = {
            "ts": ts,
            "T":  round(T, 4),
            "S":  round(S, 2),
            "vfe_mid": round(vfe_mid, 2),
            "iv_atm":  round(td["iv_atm"], 5),
            "smile":   None if smile is None else
                       [round(smile[0], 5), round(smile[1], 5),
                        round(smile[2], 5)],
            "book_delta": round(book_delta, 1),
            "book_vega":  round(book_vega, 0),
        }

        return result, conversions, json.dumps(td)

    # ── Per-strike strategy ──────────────────────────────────────────────────

    def _trade_strike(
        self, prod: str, K: int, od: OrderDepth, pos: int,
        bb: int, ba: int, mid: float,
        fair: float, delta: float, vega: float, T: float,
        can_buy, can_sell,
    ) -> Tuple[List[Order], dict]:

        orders: List[Order] = []
        limit = self.POSITION_LIMITS[prod]
        spread = ba - bb

        is_itm = K <= 4500
        is_atm = 5000 <= K <= 5300

        # Inventory throttle.
        loaded_long  = pos >  self.PER_STRIKE_THROTTLE * limit
        loaded_short = pos < -self.PER_STRIKE_THROTTLE * limit

        # ── (4) Smile-residual aggressive takes ─────────────────────────────
        # ATM strikes: edge measured against smile-fitted fair.
        # ITM strikes (4000/4500): edge measured against intrinsic + small
        # extrinsic. Smile extrapolation is uncertain at deep ITM, so we use
        # a wider price floor (4 ticks) for ITM takes.
        if T > 0:
            edge_buy  = fair - ba
            edge_sell = bb - fair
            iv_resid_thr = self.EDGE_TAKE_IV_RESID * vega
            price_floor = 4.0 if is_itm else self.EDGE_TAKE_PRICE
            edge_thr = max(price_floor, iv_resid_thr)

            if not loaded_long and edge_buy > edge_thr:
                avail = -od.sell_orders[ba]
                buy_room = limit - pos
                q = min(avail, self.AGGRO_SIZE_MAX, buy_room)
                while q > 0 and not can_buy(q * delta, q * vega):
                    q -= 1
                if q > 0:
                    orders.append(Order(prod, ba, q))
                    pos += q  # local update for passive sizing below

            if not loaded_short and edge_sell > edge_thr:
                avail = od.buy_orders[bb]
                sell_room = limit + pos
                q = min(avail, self.AGGRO_SIZE_MAX, sell_room)
                while q > 0 and not can_sell(q * delta, q * vega):
                    q -= 1
                if q > 0:
                    orders.append(Order(prod, bb, -q))
                    pos -= q

        # ── (2)/(3) Passive market making ───────────────────────────────────
        min_spread = self.MIN_ITM_PASSIVE_SPREAD if is_itm \
                     else self.MIN_PASSIVE_SPREAD
        if spread < min_spread:
            return orders, {"pos": pos, "fair": round(fair, 2),
                            "spread": spread, "skipped": "spread<min"}

        # Inventory skew (in ticks): shifts both quotes against position.
        inv = pos / limit

        # Smile-residual skew: cheap on smile → push quotes UP (lean long);
        # rich on smile → push quotes DOWN (lean short). Magnitude capped.
        signal = fair - mid
        sig_skew_ticks = int(round(max(-2, min(2, signal / 1.0))))

        if spread <= 2:
            # Tight book (5300 with spread=2 most of day-3): no room to
            # quote inside, so JOIN the book at bb / ba. We give up the
            # 1-tick edge but participate in sweep flow. Inventory throttle
            # + signal skew shave size on the unwanted side.
            inv_skew_ticks = int(round(inv * 1.0))
            skew = sig_skew_ticks - inv_skew_ticks
            our_bid = bb
            our_ask = ba
            # Skew nudges one side off the touch by 1 tick to bias fills.
            if skew > 0:        # lean long: bid harder
                our_bid = bb
                our_ask = ba + 1
            elif skew < 0:      # lean short: ask harder
                our_bid = bb - 1
                our_ask = ba
        else:
            # Wide-enough book: quote inside (bb+edge_in / ba-edge_in).
            edge_in = 2 if is_itm else 1
            base_bid = bb + edge_in
            base_ask = ba - edge_in
            inv_skew_ticks = int(round(inv * (spread / 2.0)))
            skew = sig_skew_ticks - inv_skew_ticks
            our_bid = base_bid + skew
            our_ask = base_ask + skew
            our_bid = max(bb + 1, min(our_bid, ba - 2))
            our_ask = max(bb + 2, min(our_ask, ba - 1))
            if our_bid >= our_ask:
                our_ask = our_bid + 1
            if our_ask >= ba:
                return orders, {"pos": pos, "fair": round(fair, 2),
                                "spread": spread, "skipped": "no inside room"}

        # Sizing.
        base_size = self.PASSIVE_SIZE_ITM if is_itm else self.PASSIVE_SIZE_ATM
        # Pin damper near expiry on ATM only.
        if is_atm and T < self.PIN_T_THRESHOLD:
            base_size = max(2, base_size // 2)

        bid_qty = min(base_size, limit - pos)
        ask_qty = min(base_size, limit + pos)

        # Linear taper on the side that adds to inventory.
        thr = self.PER_STRIKE_THROTTLE
        bid_taper = max(0.0,  inv) / thr   # 0 when flat, 1 when at throttle
        ask_taper = max(0.0, -inv) / thr
        bid_qty = int(bid_qty * (1.0 - min(1.0, bid_taper)))
        ask_qty = int(ask_qty * (1.0 - min(1.0, ask_taper)))

        # Hard kill of add-side past throttle.
        if loaded_long:
            bid_qty = 0
        if loaded_short:
            ask_qty = 0

        # Governor on each side (passive bid that fills also adds delta).
        while bid_qty > 0 and not can_buy(bid_qty * delta, bid_qty * vega):
            bid_qty -= 1
        while ask_qty > 0 and not can_sell(ask_qty * delta, ask_qty * vega):
            ask_qty -= 1

        if bid_qty > 0:
            orders.append(Order(prod, our_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order(prod, our_ask, -ask_qty))

        return orders, {
            "pos": pos, "fair": round(fair, 2),
            "spread": spread,
            "bid": our_bid, "bid_qty": bid_qty,
            "ask": our_ask, "ask_qty": ask_qty,
            "skew": skew, "sig": round(signal, 2),
        }
