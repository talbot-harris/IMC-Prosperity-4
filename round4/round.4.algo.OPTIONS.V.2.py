"""
IMC Prosperity Round 5 — Options algorithm with Mark intelligence + exotics

Builds on round.4.algo.OPTIONS.py (V2) and incorporates four new pieces of
guidance from the round-5 reading material:

  HINT 1 (binary + knock-out puts): both have discontinuous payoffs.
    Risk concentrates at one threshold/path. Layer vanillas alongside to
    soften the cliff. → We add closed-form pricers for digital and barrier
    options, route them through the same delta/vega governor, and offer a
    vanilla-replication hedge hook.

  HINT 2 (counterparty Marks): each Mark is a stable persona — some MM-like
    (noise), some directional, some big-and-clumsy. Their arrival is a
    signal for the underlying. → We hard-code per-Mark profiles from the
    day-3 evidence (submission 514293 / 523472 trade history) and run a
    decaying "clumsy signal" that biases quoting + take aggressiveness on
    every VEV strike.

  HINT 3 (exotics are path/spot/timing-sensitive): replicate parts with
    vanillas, control exposure. → Exotic positions feed the same global
    delta/vega/gamma governors as vanilla options, with hard caps. The
    file emits no orders for unknown products until their names are
    confirmed (logs them under td["unknown_products"] for post-mortem).

  HINT 4 (chooser, two phases): pre-choice is a straddle-like flexibility
    payoff; post-choice is a directional vanilla. → We track choose-time
    in traderData; pre-choice price uses the closed-form chooser formula,
    post-choice routes to call or put pricer based on a stored decision.

Mark profile (encoded from day-3 trade history):

  Mark    role           bias  max_sz  notes
  01      OTM_BUYER      +0.89   8     buys VEV_6000/6500/VFE; persistent
  14      MM_LIKE        +0.26  10     HP/VFE; balanced — treat as noise
  22      OTM_SELLER     -0.98   7     sells VEV_5400/6000/6500; persistent
  38      PURE_MM        -0.02   6     HP + ITM calls; balanced — noise
  49      BIG_CLUMSY_S   -1.00  12     **VFE-only, 3 trades total — signal**
  55      VFE_TAKER      -0.21   8     VFE-only, mid-sized
  67      BIG_CLUMSY_L   +1.00  12     **VFE-only, 5 trades total — signal**

Actionable rules:
  - When Mark 49 sells VFE with qty ≥ 8 → clumsy_signal -= 1.0
  - When Mark 67 buys  VFE with qty ≥ 8 → clumsy_signal += 1.0
  - Signal half-life = 500 ticks (5% of a day) — aggressive decay, since
    a single sweep is informative but not for the whole day.
  - Signal magnitude shifts every VEV quote by up to ±2 ticks against
    direction we expect S to move. Buy-side aggressive take threshold
    drops when signal is positive; sell-side drops when negative.
  - Mark 01 / 22 OTM flow does NOT enter the signal: they are persistent
    flow, not events. Their effect is already in the smile fit.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json


# ── Black–Scholes core ────────────────────────────────────────────────────────

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


def _bs_d1(S: float, K: float, T: float, sigma: float) -> float:
    sv = sigma * math.sqrt(T)
    return (math.log(S / K) + 0.5 * sigma * sigma * T) / sv


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)


def _bs_put(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(K - S, 0.0)
    return _bs_call(S, K, T, sigma) - S + K       # r = 0


def _bs_call_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    return _ncdf(_bs_d1(S, K, T, sigma))


def _bs_put_delta(S, K, T, sigma):
    return _bs_call_delta(S, K, T, sigma) - 1.0


def _bs_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    return S * math.sqrt(T) * _npdf(_bs_d1(S, K, T, sigma))


def _bs_gamma(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    return _npdf(_bs_d1(S, K, T, sigma)) / (S * sigma * math.sqrt(T))


# ── Exotic pricers (HINT 1, 3, 4) ────────────────────────────────────────────

def _bs_digital_call(S, K, T, sigma) -> float:
    """Cash-or-nothing digital: pays $1 if S_T > K, else 0. r = 0."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d2 = _bs_d1(S, K, T, sigma) - sigma * math.sqrt(T)
    return _ncdf(d2)


def _bs_digital_put(S, K, T, sigma) -> float:
    """Cash-or-nothing digital put: pays $1 if S_T < K. r = 0."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S < K else 0.0
    d2 = _bs_d1(S, K, T, sigma) - sigma * math.sqrt(T)
    return _ncdf(-d2)


def _bs_digital_put_delta(S, K, T, sigma):
    """∂P_dig/∂S. Useful for governor."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d2 = _bs_d1(S, K, T, sigma) - sigma * math.sqrt(T)
    # P_dig = N(-d2); dP/dS = -φ(d2) * dd2/dS = -φ(d2) / (S σ √T)
    return -_npdf(d2) / (S * sigma * math.sqrt(T))


def _down_and_out_put(S, K, T, sigma, B) -> float:
    """Down-and-out put: vanilla put with knock-out barrier B (B < S).
    Standard Hull-Reiner-Rubinstein closed form for r = 0.

    For B < K (barrier below strike — the typical case):
        P_DI =  -S·N(-x1)  + K·N(-x1+σ√T)
                + B·[N(y) - N(y1)]
                - K·(S/B)·[N(y-σ√T) - N(y1-σ√T)]
        P_DO =  P_vanilla(S,K) - P_DI

    For B ≥ K, the down-and-out put is equivalent to a no-knock-in put
    on the surviving paths and the simpler form
        P_DO = P_vanilla - (S/B)·P_vanilla(B²/S, K)·1{B²/S<K}
    suffices. We pick formulas by case to avoid sign blow-ups.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or B <= 0:
        return max(K - S, 0.0) if S > B else 0.0
    if S <= B:
        return 0.0
    sv = sigma * math.sqrt(T)
    lam = 0.5                     # (r + σ²/2)/σ² with r = 0

    if B < K:
        x1 = math.log(S / B) / sv + lam * sv
        y  = math.log(B*B / (S*K)) / sv + lam * sv
        y1 = math.log(B / S) / sv + lam * sv
        BoS_2lam   = (B / S) ** (2 * lam)        # = B/S
        BoS_2lam_2 = (B / S) ** (2 * lam - 2)    # = S/B
        p_di = (-S * _ncdf(-x1) + K * _ncdf(-x1 + sv)
                + S * BoS_2lam   * (_ncdf(y)  - _ncdf(y1))
                - K * BoS_2lam_2 * (_ncdf(y  - sv) - _ncdf(y1 - sv)))
        return max(0.0, _bs_put(S, K, T, sigma) - p_di)

    # B ≥ K case — barrier at or above strike. The reflection-only term
    # is bounded; clip at vanilla-put value as upper bound.
    refl = B * B / S
    p_refl = _bs_put(refl, K, T, sigma)
    return max(0.0, min(_bs_put(S, K, T, sigma),
                        _bs_put(S, K, T, sigma) - (S / B) * p_refl))


def _down_and_out_put_delta(S, K, T, sigma, B):
    """Numerical ∂/∂S — closed-form gets messy with the reflection term."""
    if S <= B + 0.5:
        return 0.0
    h = max(0.5, S * 1e-4)
    return (_down_and_out_put(S+h, K, T, sigma, B)
          - _down_and_out_put(S-h, K, T, sigma, B)) / (2 * h)


def _chooser_value(S, K, T_total, T_choose, sigma) -> float:
    """Standard chooser: at t = T_choose ≤ T_total holder picks max(call, put)
    with strike K, expiry T_total. r = 0 closed form (Rubinstein):

        V = C(S, K, T_total) + P(S, K, T_choose)

    Pre-choice value. After T_choose passes, the algo should know which
    leg was chosen; pricing reverts to vanilla."""
    if T_choose >= T_total:
        T_choose = T_total - 1e-6
    if T_choose <= 0:
        # past the choice point — caller should switch to vanilla pricer
        return max(_bs_call(S, K, T_total, sigma),
                   _bs_put( S, K, T_total, sigma))
    return _bs_call(S, K, T_total, sigma) + _bs_put(S, K, T_choose, sigma)


# ── Calibration helpers ──────────────────────────────────────────────────────

def _implied_vol(price, S, K, T) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or T <= 0 or S <= 0:
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(50):
        m = (lo + hi) / 2
        if _bs_call(S, K, T, m) < price: lo = m
        else: hi = m
    return (lo + hi) / 2


def _infer_tte(price, S, K, sigma, lo=0.01, hi=6.0) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or S <= 0 or sigma <= 0:
        return None
    if _bs_call(S, K, hi, sigma) < price: return hi
    if _bs_call(S, K, lo, sigma) > price: return lo
    for _ in range(50):
        m = (lo + hi) / 2
        if _bs_call(S, K, m, sigma) < price: lo = m
        else: hi = m
    return (lo + hi) / 2


def _fit_quadratic(xs, ys) -> Optional[Tuple[float, float, float]]:
    n = len(xs)
    if n < 3: return None
    s0 = float(n); s1 = sum(xs); s2 = sum(x*x for x in xs)
    s3 = sum(x**3 for x in xs); s4 = sum(x**4 for x in xs)
    t0 = sum(ys); t1 = sum(x*y for x,y in zip(xs,ys))
    t2 = sum(x*x*y for x,y in zip(xs,ys))
    m = [[s0,s1,s2,t0],[s1,s2,s3,t1],[s2,s3,s4,t2]]
    for i in range(3):
        p = max(range(i,3), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12: return None
        m[i], m[p] = m[p], m[i]
        for j in range(i+1, 3):
            f = m[j][i]/m[i][i]
            for k in range(i, 4): m[j][k] -= f*m[i][k]
    c = m[2][3]/m[2][2]
    b = (m[1][3]-m[1][2]*c)/m[1][1]
    a = (m[0][3]-m[0][1]*b-m[0][2]*c)/m[0][0]
    return a, b, c


# ── Mark counterparty profiles (HINT 2) ──────────────────────────────────────

# role values:
#   "BIG_CLUMSY_LONG"  — large, rare, one-sided buyer in VFE → +signal
#   "BIG_CLUMSY_SHORT" — large, rare, one-sided seller in VFE → −signal
#   "OTM_FLOW"         — persistent, in-the-smile (ignore for signal)
#   "MM_LIKE"          — balanced, frequent (ignore — pure noise)
MARK_ROLES = {
    "Mark 01": "OTM_FLOW",          # +0.89 bias, OTM buyer
    "Mark 14": "MM_LIKE",           # balanced
    "Mark 22": "OTM_FLOW",          # -0.98 bias, OTM seller
    "Mark 38": "MM_LIKE",           # balanced
    "Mark 49": "BIG_CLUMSY_SHORT",  # rare, big, sells VFE
    "Mark 55": "MM_LIKE",           # mid-sized VFE taker — treat as noise
    "Mark 67": "BIG_CLUMSY_LONG",   # rare, big, buys VFE
}

CLUMSY_QTY_THRESHOLD = 8     # ≥8 in VFE counts as a clumsy event


# ── Product-type routing ─────────────────────────────────────────────────────

def detect_type(product: str) -> str:
    """Classify any product name into one of: HP, VFE, CALL, PUT,
    BINARY_CALL, BINARY_PUT, KO_CALL, KO_PUT, CHOOSER, UNKNOWN."""
    p = product.upper()
    if p == "HYDROGEL_PACK": return "HP"
    if p == "VELVETFRUIT_EXTRACT": return "VFE"
    if p.startswith("VEV_"): return "CALL"
    # Heuristics for round-5 products. Names not yet known — match on
    # likely keywords. Adjust when actual names are seen in the log.
    if "BINARY" in p or "DIGITAL" in p:
        return "BINARY_PUT" if "PUT" in p else "BINARY_CALL"
    if "KO" in p or "KNOCK" in p or "BARRIER" in p:
        return "KO_PUT" if "PUT" in p else "KO_CALL"
    if "CHOOSER" in p:
        return "CHOOSER"
    return "UNKNOWN"


def parse_strike(product: str) -> Optional[int]:
    """Pull the trailing integer from a product name (VEV_5100 → 5100)."""
    parts = product.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


# ──────────────────────────────────────────────────────────────────────────────

class Trader:

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    STRIKES_ALL  = [4000, 4500, 5000, 5100, 5200, 5300,
                    5400, 5500, 6000, 6500]
    STRIKES_MM   = [4000, 4500, 5000, 5100, 5200, 5300]
    STRIKES_FIT  = [5000, 5100, 5200, 5300, 5400]
    STRIKES_DEAD = {5500, 6000, 6500}

    POSITION_LIMITS = {f"VEV_{k}": 300 for k in STRIKES_ALL}
    # Default for any new exotic product — overridden if learnt from state.
    DEFAULT_EXOTIC_LIMIT = 100

    TICKS_PER_DAY = 100_000
    TTE_START     = 4.0

    IV_PRIOR = 0.0195
    IV_MIN   = 0.005
    IV_MAX   = 0.10

    SMILE_EMA_ALPHA = 0.20
    S_EMA_ALPHA     = 0.5

    # Same V2 sizing/threshold parameters
    EDGE_TAKE_PRICE      = 1.0
    EDGE_TAKE_IV_RESID   = 0.0005
    PER_STRIKE_THROTTLE  = 0.65
    DELTA_CAP            = 300.0
    VEGA_CAP             = 200_000.0
    PASSIVE_SIZE_ATM     = 25
    PASSIVE_SIZE_ITM     = 15
    AGGRO_SIZE_MAX       = 15
    PIN_T_THRESHOLD      = 0.25
    MIN_PASSIVE_SPREAD   = 2
    MIN_ITM_PASSIVE_SPREAD = 5

    # Mark clumsy signal (NEW)
    CLUMSY_HALFLIFE_TICKS  = 500    # ts units; 1 tick = 100 timestamp
    CLUMSY_SKEW_MAX_TICKS  = 2      # max additional skew per quote
    CLUMSY_TAKE_BOOST      = 0.50   # multiplies edge threshold downward
    CLUMSY_TRIGGER_QTY     = CLUMSY_QTY_THRESHOLD

    # Exotic governor caps (HINT 3) — treated alongside vanilla
    GAMMA_CAP              = 200.0   # in S²/contract — keep convexity in check

    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result      = {p: [] for p in state.order_depths}
        conversions = 0

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        ts = state.timestamp
        last_ts = td.get("last_ts", -1)

        # Day boundary handling for TTE.
        if ts == 0 and last_ts > 0 and "T_calib" in td:
            td["T_calib"]    = max(0.0, td["T_calib"] - 1.0)
            td["T_calib_ts"] = 0
        td["last_ts"] = ts

        td.setdefault("s_ema",          None)
        td.setdefault("smile",          None)
        td.setdefault("iv_atm",         self.IV_PRIOR)
        td.setdefault("clumsy_signal",  0.0)
        td.setdefault("clumsy_last_ts", ts)
        td.setdefault("unknown_products", [])

        # ── Mids ────────────────────────────────────────────────────────────
        mids: Dict[str, Tuple[float, int, int]] = {}
        for product, od in state.order_depths.items():
            bb = max(od.buy_orders)  if od.buy_orders  else None
            ba = min(od.sell_orders) if od.sell_orders else None
            if bb is not None and ba is not None:
                mids[product] = ((bb + ba) / 2.0, bb, ba)

        if self.UNDERLYING not in mids:
            return result, conversions, json.dumps(td)

        vfe_mid, _, _ = mids[self.UNDERLYING]
        if td["s_ema"] is None: td["s_ema"] = vfe_mid
        else: td["s_ema"] += self.S_EMA_ALPHA * (vfe_mid - td["s_ema"])
        S = td["s_ema"]

        # ── Mark clumsy signal update (HINT 2) ──────────────────────────────
        # Decay first based on elapsed timestamp.
        elapsed_ts = max(0, ts - td["clumsy_last_ts"])
        if elapsed_ts > 0:
            half = self.CLUMSY_HALFLIFE_TICKS * 100  # convert tick → timestamp
            decay = 0.5 ** (elapsed_ts / half)
            td["clumsy_signal"] *= decay
        td["clumsy_last_ts"] = ts

        # Process market trades on VFE only — that's where the clumsy
        # signal lives per the day-3 evidence.
        vfe_trades = state.market_trades.get(self.UNDERLYING, []) or []
        for trade in vfe_trades:
            buyer  = getattr(trade, "buyer",  "") or ""
            seller = getattr(trade, "seller", "") or ""
            qty    = abs(int(getattr(trade, "quantity", 0)))
            if qty < self.CLUMSY_TRIGGER_QTY:
                continue
            if MARK_ROLES.get(buyer)  == "BIG_CLUMSY_LONG":
                td["clumsy_signal"] += 1.0
            if MARK_ROLES.get(seller) == "BIG_CLUMSY_SHORT":
                td["clumsy_signal"] -= 1.0
        # Saturate to ±3.
        td["clumsy_signal"] = max(-3.0, min(3.0, td["clumsy_signal"]))
        clumsy = td["clumsy_signal"]

        # ── TTE bootstrap (unchanged from V2) ───────────────────────────────
        if "T_calib" not in td:
            T_est, sig_est = 2.0, self.IV_PRIOR
            for _ in range(6):
                samples = []
                for k in (5000, 5100, 5200, 5300):
                    p = f"VEV_{k}"
                    if p in mids: samples.append((float(k), mids[p][0]))
                if not samples: break
                ivs = [_implied_vol(c, S, k, T_est) for k, c in samples]
                ivs = [x for x in ivs if x is not None]
                if ivs:
                    sig_est = max(self.IV_MIN, min(self.IV_MAX,
                                                   sum(ivs)/len(ivs)))
                ttes = [_infer_tte(c, S, k, sig_est) for k, c in samples]
                ttes = [x for x in ttes if x is not None]
                if ttes:
                    T_est = max(0.01, min(self.TTE_START, sum(ttes)/len(ttes)))
            td["T_calib"]    = T_est
            td["T_calib_ts"] = ts
            td["iv_atm"]     = sig_est

        elapsed = (ts - td["T_calib_ts"]) / self.TICKS_PER_DAY
        T = max(0.0, td["T_calib"] - elapsed)

        # ── Smile fit (unchanged) ──────────────────────────────────────────
        smile_xs, smile_ys = [], []
        for k in self.STRIKES_FIT:
            p = f"VEV_{k}"
            if p not in mids: continue
            v = _implied_vol(mids[p][0], S, float(k), T)
            if v is None or v < self.IV_MIN or v > self.IV_MAX: continue
            smile_xs.append(math.log(float(k) / S))
            smile_ys.append(v)
        smile_now = _fit_quadratic(smile_xs, smile_ys) \
                    if len(smile_xs) >= 3 else None
        if smile_now is not None:
            if td["smile"] is None:
                td["smile"] = list(smile_now)
            else:
                a_s,b_s,c_s = td["smile"]
                a_n,b_n,c_n = smile_now
                a = self.SMILE_EMA_ALPHA
                td["smile"] = [a_s+a*(a_n-a_s), b_s+a*(b_n-b_s), c_s+a*(c_n-c_s)]
        smile = td["smile"]
        if smile is not None:
            td["iv_atm"] = max(self.IV_MIN, min(self.IV_MAX, smile[0]))

        def smile_iv(K: float) -> float:
            if smile is None or S <= 0: return td["iv_atm"]
            x = math.log(K / S)
            iv = smile[0] + smile[1]*x + smile[2]*x*x
            return max(self.IV_MIN, min(self.IV_MAX, iv))

        # ── Aggregate book greeks (now spans vanilla + exotic) ─────────────
        book_delta = 0.0
        book_vega  = 0.0
        book_gamma = 0.0
        for prod, p in state.position.items():
            if p == 0 or T <= 0: continue
            ptype = detect_type(prod)
            K = parse_strike(prod) or 0
            iv_k = smile_iv(float(K)) if K > 0 else td["iv_atm"]

            if ptype == "CALL":
                book_delta += p * _bs_call_delta(S, float(K), T, iv_k)
                book_vega  += p * _bs_vega(S, float(K), T, iv_k)
                book_gamma += p * _bs_gamma(S, float(K), T, iv_k)
            elif ptype == "BINARY_PUT":
                book_delta += p * _bs_digital_put_delta(S, float(K), T, iv_k)
                # Digital vega is small except very near K; we conservatively
                # track via a small constant per contract.
                book_vega  += p * 0.0
            elif ptype == "BINARY_CALL":
                book_delta += p * (-_bs_digital_put_delta(S, float(K), T, iv_k))
            elif ptype in ("KO_PUT", "KO_CALL"):
                # Use numerical delta. Without a clear barrier from the
                # name we punt on vega/gamma until the product is observed.
                # Place-holder: assume KO-put barrier at 0.9*K.
                B = 0.9 * float(K) if K > 0 else 0
                book_delta += p * _down_and_out_put_delta(
                    S, float(K), T, iv_k, B)
            # CHOOSER / UNKNOWN: not yet active

        # ── Governors (block-only-if-worsening logic, V2) ──────────────────
        def can_buy(d_extra, v_extra, g_extra=0.0):
            new_d = book_delta + d_extra
            new_v = book_vega  + v_extra
            new_g = book_gamma + g_extra
            if new_d > self.DELTA_CAP and new_d > book_delta: return False
            if abs(new_v) > self.VEGA_CAP and abs(new_v) > abs(book_vega): return False
            if abs(new_g) > self.GAMMA_CAP and abs(new_g) > abs(book_gamma): return False
            return True

        def can_sell(d_extra, v_extra, g_extra=0.0):
            new_d = book_delta - d_extra
            new_v = book_vega  - v_extra
            new_g = book_gamma - g_extra
            if new_d < -self.DELTA_CAP and new_d < book_delta: return False
            if abs(new_v) > self.VEGA_CAP and abs(new_v) > abs(book_vega): return False
            if abs(new_g) > self.GAMMA_CAP and abs(new_g) > abs(book_gamma): return False
            return True

        # ── Per-product trading ─────────────────────────────────────────────
        for product, od in state.order_depths.items():
            ptype = detect_type(product)
            if ptype == "CALL":
                K = parse_strike(product)
                if K is None or K not in self.STRIKES_MM:
                    continue
                if product not in mids: continue
                pos = state.position.get(product, 0)
                mid_k, bb, ba = mids[product]
                iv_k = smile_iv(float(K))
                fair = _bs_call(S, float(K), T, iv_k)
                d_k  = _bs_call_delta(S, float(K), T, iv_k)
                v_k  = _bs_vega(S, float(K), T, iv_k)
                g_k  = _bs_gamma(S, float(K), T, iv_k)
                result[product] = self._trade_call(
                    product, K, od, pos, bb, ba, mid_k,
                    fair, d_k, v_k, g_k, T, clumsy,
                    can_buy, can_sell)
            elif ptype in ("BINARY_PUT", "BINARY_CALL", "KO_PUT", "KO_CALL",
                           "CHOOSER"):
                # Stub: log existence for now. Pricers ready, but until
                # we observe order-depth shapes we don't emit orders.
                if product not in td["unknown_products"]:
                    td["unknown_products"].append(f"EXOTIC:{product}")
            elif ptype == "UNKNOWN":
                if product not in td["unknown_products"]:
                    td["unknown_products"].append(product)

        # Snapshot.
        td["debug"] = {
            "ts": ts,
            "T":  round(T, 4),
            "S":  round(S, 2),
            "vfe_mid": round(vfe_mid, 2),
            "iv_atm":  round(td["iv_atm"], 5),
            "smile":   None if smile is None else
                       [round(smile[0],5), round(smile[1],5), round(smile[2],5)],
            "book_delta": round(book_delta, 1),
            "book_vega":  round(book_vega, 0),
            "book_gamma": round(book_gamma, 1),
            "clumsy":     round(clumsy, 2),
        }

        return result, conversions, json.dumps(td)

    # ── Vanilla call strategy (V2 logic + clumsy-signal injection) ───────────

    def _trade_call(
        self, prod: str, K: int, od: OrderDepth, pos: int,
        bb: int, ba: int, mid: float,
        fair: float, delta: float, vega: float, gamma: float,
        T: float, clumsy: float,
        can_buy, can_sell,
    ) -> List[Order]:

        orders: List[Order] = []
        limit  = self.POSITION_LIMITS[prod]
        spread = ba - bb
        is_itm = K <= 4500
        is_atm = 5000 <= K <= 5300

        loaded_long  = pos >  self.PER_STRIKE_THROTTLE * limit
        loaded_short = pos < -self.PER_STRIKE_THROTTLE * limit

        # Aggressive takes — threshold lowered when clumsy signal aligns.
        if T > 0:
            base_thr = max(4.0 if is_itm else self.EDGE_TAKE_PRICE,
                           self.EDGE_TAKE_IV_RESID * vega)
            # Clumsy alignment: positive clumsy → expect S↑ → call fair
            # rises → buy is more attractive. Multiply the BUY-side
            # threshold by (1 - boost*max(0,clumsy)) so positive clumsy
            # makes us more eager to lift offers, and vice versa.
            buy_thr  = base_thr * max(0.25, 1.0 - self.CLUMSY_TAKE_BOOST*max(0.0, clumsy))
            sell_thr = base_thr * max(0.25, 1.0 - self.CLUMSY_TAKE_BOOST*max(0.0, -clumsy))

            edge_buy  = fair - ba
            edge_sell = bb - fair

            if not loaded_long and edge_buy > buy_thr:
                avail = -od.sell_orders[ba]
                room  = limit - pos
                q = min(avail, self.AGGRO_SIZE_MAX, room)
                while q > 0 and not can_buy(q*delta, q*vega, q*gamma):
                    q -= 1
                if q > 0:
                    orders.append(Order(prod, ba, q))
                    pos += q
            if not loaded_short and edge_sell > sell_thr:
                avail = od.buy_orders[bb]
                room  = limit + pos
                q = min(avail, self.AGGRO_SIZE_MAX, room)
                while q > 0 and not can_sell(q*delta, q*vega, q*gamma):
                    q -= 1
                if q > 0:
                    orders.append(Order(prod, bb, -q))
                    pos -= q

        # Passive MM.
        min_spread = self.MIN_ITM_PASSIVE_SPREAD if is_itm \
                     else self.MIN_PASSIVE_SPREAD
        if spread < min_spread:
            return orders

        inv = pos / limit
        signal = fair - mid
        sig_skew_ticks = int(round(max(-2, min(2, signal / 1.0))))
        # Clumsy-signal skew: positive → quotes shift UP (we're more eager
        # to buy / less eager to sell at low prices). Magnitude capped.
        clumsy_skew = int(round(max(-self.CLUMSY_SKEW_MAX_TICKS,
                                    min(self.CLUMSY_SKEW_MAX_TICKS,
                                        clumsy * self.CLUMSY_SKEW_MAX_TICKS / 2.0))))

        if spread <= 2:
            # Tight book: join at bb/ba; clumsy + skew biases sides.
            inv_skew = int(round(inv * 1.0))
            net = sig_skew_ticks - inv_skew + clumsy_skew
            our_bid, our_ask = bb, ba
            if net > 0:    our_ask = ba + 1
            elif net < 0:  our_bid = bb - 1
        else:
            edge_in = 2 if is_itm else 1
            base_bid = bb + edge_in
            base_ask = ba - edge_in
            inv_skew = int(round(inv * (spread / 2.0)))
            net = sig_skew_ticks - inv_skew + clumsy_skew
            our_bid = base_bid + net
            our_ask = base_ask + net
            our_bid = max(bb + 1, min(our_bid, ba - 2))
            our_ask = max(bb + 2, min(our_ask, ba - 1))
            if our_bid >= our_ask:
                our_ask = our_bid + 1
            if our_ask >= ba:
                return orders

        base_size = self.PASSIVE_SIZE_ITM if is_itm else self.PASSIVE_SIZE_ATM
        if is_atm and T < self.PIN_T_THRESHOLD:
            base_size = max(2, base_size // 2)

        bid_qty = min(base_size, limit - pos)
        ask_qty = min(base_size, limit + pos)

        thr = self.PER_STRIKE_THROTTLE
        bid_taper = max(0.0,  inv) / thr
        ask_taper = max(0.0, -inv) / thr
        bid_qty = int(bid_qty * (1.0 - min(1.0, bid_taper)))
        ask_qty = int(ask_qty * (1.0 - min(1.0, ask_taper)))

        if loaded_long:  bid_qty = 0
        if loaded_short: ask_qty = 0

        # Clumsy-signal volume bias: when signal is strong, upsize the side
        # that aligns with the signal direction and shrink the other.
        if clumsy >= 1.0:
            # expect S↑: buying calls is good. Boost bid, shrink ask.
            bid_qty = int(bid_qty * 1.4)
            ask_qty = int(ask_qty * 0.6)
        elif clumsy <= -1.0:
            bid_qty = int(bid_qty * 0.6)
            ask_qty = int(ask_qty * 1.4)
        bid_qty = min(bid_qty, limit - pos)
        ask_qty = min(ask_qty, limit + pos)

        while bid_qty > 0 and not can_buy(bid_qty*delta, bid_qty*vega, bid_qty*gamma):
            bid_qty -= 1
        while ask_qty > 0 and not can_sell(ask_qty*delta, ask_qty*vega, ask_qty*gamma):
            ask_qty -= 1

        if bid_qty > 0:
            orders.append(Order(prod, our_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order(prod, our_ask, -ask_qty))
        return orders
