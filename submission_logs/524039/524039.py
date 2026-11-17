"""
IMC Prosperity Round 4 — FINAL unified algorithm

Combines three previously-separate per-product algos into a single Trader:

  HYDROGEL_PACK        — VBI-gated passive MM (from round.4.algo.HP.py v3)
  VELVETFRUIT_EXTRACT  — passive MM + cross-product delta hedge (new)
  VEV_K (options)      — smile-fit + Mark intelligence + exotic scaffolding
                         (from round.4.algo.OPTIONS.V.2.py / round-5 prep)

Two structural fixes vs the prior options-only file:

  1. STAGNATION KILLER. The options book filled only on Mark sweeps —
     ~9 events per day per strike. Two new mechanisms break that:

       a. Anti-stagnation tick-march: per-strike `last_fill_ts` is tracked.
          When `ts - last_fill_ts > STAGNATION_NOFILL_TICKS`, the passive
          quote marches 1 tick toward the touch every STAGNATION_STEP_TICKS,
          capped by quoting AT bb / ba once it gets there. Resets on any
          fill (detected via SUBMISSION presence in state.market_trades).

       b. Quote-at-touch fallback for any spread ≥ 2: previously skipped
          tight 5300/5400; now we always quote, joining bb/ba when there's
          no inside room. Accepts adverse selection — pays for in
          per-strike inventory throttle and global governors.

  2. CROSS-PRODUCT DELTA HEDGE. The options book's net delta now feeds
     directly into VFE's quote-skew: when the options book is long-delta,
     VFE quotes shift down (favoring sells) to flatten the portfolio.
     This is the under-the-hood "vanilla replication" the round-5 hint
     keeps pointing at.

Aggressiveness pass on every product:

  HP   SIZE        40 → 80   |  VBI_BOOST 0.75 → 1.5
  VFE  SIZE        30 → 70   |  AGGR_SIZE  25 → 50  |  added VBI gate
  VEV  ATM passive 25 → 50   |  ITM passive 15 → 30
       AGGRO_SIZE  15 → 35   |  EDGE_TAKE_PRICE 1.0 → 0.4
       IV_RESID    0.0005 → 0.0002
       ITM passive moved from bb+2/ba-2 to bb+1/ba-1
       PER_STRIKE_THROTTLE 0.65 → 0.85
       Clumsy skew cap 2 → 4 ticks  |  take boost 0.50 → 0.80
  Governors  DELTA 300 → 500 | VEGA 200k → 500k | GAMMA 200 → 600

"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json


# ── Black–Scholes core ───────────────────────────────────────────────────────

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

def _npdf(x): return math.exp(-0.5*x*x) / SQRT_2PI

def _bs_d1(S, K, T, sigma):
    return (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma * math.sqrt(T))

def _bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)

def _bs_put(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return max(K - S, 0.0)
    return _bs_call(S, K, T, sigma) - S + K

def _bs_call_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 1.0 if S > K else 0.0
    return _ncdf(_bs_d1(S, K, T, sigma))

def _bs_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    return S * math.sqrt(T) * _npdf(_bs_d1(S, K, T, sigma))

def _bs_gamma(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    return _npdf(_bs_d1(S, K, T, sigma)) / (S * sigma * math.sqrt(T))

# Exotic pricers (scaffolding — round-5 ready) -------------------------------

def _bs_digital_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 1.0 if S > K else 0.0
    return _ncdf(_bs_d1(S, K, T, sigma) - sigma * math.sqrt(T))

def _bs_digital_put(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 1.0 if S < K else 0.0
    return _ncdf(-(_bs_d1(S, K, T, sigma) - sigma * math.sqrt(T)))

def _bs_digital_put_delta(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.0
    d2 = _bs_d1(S, K, T, sigma) - sigma * math.sqrt(T)
    return -_npdf(d2) / (S * sigma * math.sqrt(T))

def _down_and_out_put(S, K, T, sigma, B):
    if T <= 0 or sigma <= 0 or S <= 0 or B <= 0:
        return max(K - S, 0.0) if S > B else 0.0
    if S <= B: return 0.0
    sv = sigma * math.sqrt(T)
    lam = 0.5
    if B < K:
        x1 = math.log(S/B)/sv + lam*sv
        y  = math.log(B*B/(S*K))/sv + lam*sv
        y1 = math.log(B/S)/sv + lam*sv
        BoS_2lam   = (B/S)**(2*lam)
        BoS_2lam_2 = (B/S)**(2*lam-2)
        p_di = (-S * _ncdf(-x1) + K * _ncdf(-x1+sv)
                + S * BoS_2lam   * (_ncdf(y) - _ncdf(y1))
                - K * BoS_2lam_2 * (_ncdf(y-sv) - _ncdf(y1-sv)))
        return max(0.0, _bs_put(S, K, T, sigma) - p_di)
    refl = B*B/S
    return max(0.0, min(_bs_put(S, K, T, sigma),
                        _bs_put(S, K, T, sigma) - (S/B)*_bs_put(refl, K, T, sigma)))

def _chooser_value(S, K, T_total, T_choose, sigma):
    if T_choose >= T_total: T_choose = T_total - 1e-6
    if T_choose <= 0:
        return max(_bs_call(S, K, T_total, sigma),
                   _bs_put(S, K, T_total, sigma))
    return _bs_call(S, K, T_total, sigma) + _bs_put(S, K, T_choose, sigma)

# Calibration helpers --------------------------------------------------------

def _implied_vol(price, S, K, T):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or T <= 0 or S <= 0: return None
    lo, hi = 1e-4, 5.0
    for _ in range(50):
        m = (lo + hi) / 2
        if _bs_call(S, K, T, m) < price: lo = m
        else: hi = m
    return (lo + hi) / 2

def _infer_tte(price, S, K, sigma, lo=0.01, hi=6.0):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or S <= 0 or sigma <= 0: return None
    if _bs_call(S, K, hi, sigma) < price: return hi
    if _bs_call(S, K, lo, sigma) > price: return lo
    for _ in range(50):
        m = (lo + hi) / 2
        if _bs_call(S, K, m, sigma) < price: lo = m
        else: hi = m
    return (lo + hi) / 2

def _fit_quadratic(xs, ys):
    n = len(xs)
    if n < 3: return None
    s0=float(n); s1=sum(xs); s2=sum(x*x for x in xs)
    s3=sum(x**3 for x in xs); s4=sum(x**4 for x in xs)
    t0=sum(ys); t1=sum(x*y for x,y in zip(xs,ys)); t2=sum(x*x*y for x,y in zip(xs,ys))
    m=[[s0,s1,s2,t0],[s1,s2,s3,t1],[s2,s3,s4,t2]]
    for i in range(3):
        p = max(range(i,3), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-12: return None
        m[i], m[p] = m[p], m[i]
        for j in range(i+1,3):
            f = m[j][i]/m[i][i]
            for k in range(i,4): m[j][k] -= f*m[i][k]
    c = m[2][3]/m[2][2]
    b = (m[1][3]-m[1][2]*c)/m[1][1]
    a = (m[0][3]-m[0][1]*b-m[0][2]*c)/m[0][0]
    return a, b, c


# ── Mark counterparty profiles ───────────────────────────────────────────────

MARK_ROLES = {
    "Mark 01": "OTM_FLOW",
    "Mark 14": "MM_LIKE",
    "Mark 22": "OTM_FLOW",
    "Mark 38": "MM_LIKE",
    "Mark 49": "BIG_CLUMSY_SHORT",   # rare, big, sells VFE → expect S↓
    "Mark 55": "MM_LIKE",
    "Mark 67": "BIG_CLUMSY_LONG",    # rare, big, buys VFE → expect S↑
}
CLUMSY_QTY_THRESHOLD = 8


# ── Product-type routing (round-5 ready) ─────────────────────────────────────

def detect_type(product: str) -> str:
    p = product.upper()
    if p == "HYDROGEL_PACK": return "HP"
    if p == "VELVETFRUIT_EXTRACT": return "VFE"
    if p.startswith("VEV_"): return "CALL"
    if "BINARY" in p or "DIGITAL" in p:
        return "BINARY_PUT" if "PUT" in p else "BINARY_CALL"
    if "KO" in p or "KNOCK" in p or "BARRIER" in p:
        return "KO_PUT" if "PUT" in p else "KO_CALL"
    if "CHOOSER" in p: return "CHOOSER"
    return "UNKNOWN"

def parse_strike(product: str) -> Optional[int]:
    parts = product.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit(): return int(parts[1])
    return None


# ─────────────────────────────────────────────────────────────────────────────

class Trader:

    UNDERLYING = "VELVETFRUIT_EXTRACT"
    HP_PRODUCT = "HYDROGEL_PACK"

    STRIKES_ALL  = [4000, 4500, 5000, 5100, 5200, 5300,
                    5400, 5500, 6000, 6500]
    STRIKES_MM   = [4000, 4500, 5000, 5100, 5200, 5300, 5400]   # added 5400
    STRIKES_FIT  = [5000, 5100, 5200, 5300, 5400]
    STRIKES_DEAD = {5500, 6000, 6500}

    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        **{f"VEV_{k}": 300 for k in STRIKES_ALL},
    }

    TICKS_PER_DAY = 100_000
    TTE_START     = 4.0

    # ── HP parameters (more aggressive) ──────────────────────────────────────
    HP_SIZE          = 80          # was 40
    HP_SKEW_TICKS    = 8
    HP_THROTTLE_FRAC = 0.70        # was 0.60
    HP_AGGR_SIZE     = 50          # was 25
    HP_VBI_SOFT      = 0.04        # was 0.05 — fire on smaller imbalance
    HP_VBI_HARD      = 0.18        # was 0.20
    HP_VBI_BOOST     = 1.5         # was 0.75
    HP_VBI_PRICE_TKS = 4           # was 3

    # ── VFE parameters (more aggressive + VBI + delta hedge) ────────────────
    VFE_SIZE          = 70         # was 30
    VFE_SKEW_TICKS    = 6
    VFE_THROTTLE_FRAC = 0.75
    VFE_AGGR_SIZE     = 50         # was 25
    VFE_VBI_SOFT      = 0.04
    VFE_VBI_HARD      = 0.18
    VFE_VBI_BOOST     = 1.0
    VFE_VBI_PRICE_TKS = 2
    # Cross-product delta hedge: how many ticks of VFE quote shift per 100
    # units of net options book_delta. At book_delta=+300 (cap), shift = 6.
    VFE_HEDGE_TICKS_PER_100D = 2.0
    VFE_HEDGE_VOLUME_FRAC    = 0.5  # at full options-book delta, 50% size on flatten side

    # ── VEV options parameters (much more aggressive) ────────────────────────
    IV_PRIOR = 0.0195
    IV_MIN   = 0.005
    IV_MAX   = 0.10
    SMILE_EMA_ALPHA = 0.20
    S_EMA_ALPHA     = 0.5

    EDGE_TAKE_PRICE      = 0.4    # was 1.0 — fire takes far more often
    EDGE_TAKE_IV_RESID   = 0.0002 # was 0.0005
    PER_STRIKE_THROTTLE  = 0.85   # was 0.65
    DELTA_CAP            = 500.0  # was 300
    VEGA_CAP             = 500_000.0  # was 200k
    GAMMA_CAP            = 600.0  # was 200

    PASSIVE_SIZE_ATM     = 50     # was 25
    PASSIVE_SIZE_ITM     = 30     # was 15
    AGGRO_SIZE_MAX       = 35     # was 15
    PIN_T_THRESHOLD      = 0.25
    MIN_PASSIVE_SPREAD   = 2
    MIN_ITM_PASSIVE_SPREAD = 3    # was 5 — quote 4500 even on tighter days

    # Mark clumsy signal
    CLUMSY_HALFLIFE_TICKS = 500
    CLUMSY_SKEW_MAX_TICKS = 4     # was 2 — bigger skew per |signal|
    CLUMSY_TAKE_BOOST     = 0.80  # was 0.50

    # Anti-stagnation: when no fill on a strike for STAGNATION_NOFILL_TICKS
    # ticks (timestamp units), march our quote 1 tick closer to the touch
    # every STAGNATION_STEP_TICKS thereafter (capped at the touch itself).
    STAGNATION_NOFILL_TICKS = 2000
    STAGNATION_STEP_TICKS   = 800
    STAGNATION_MAX_MARCH    = 4   # max ticks of march toward the touch

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
        if ts == 0 and last_ts > 0 and "T_calib" in td:
            td["T_calib"]    = max(0.0, td["T_calib"] - 1.0)
            td["T_calib_ts"] = 0
        td["last_ts"] = ts

        td.setdefault("s_ema",         None)
        td.setdefault("smile",         None)
        td.setdefault("iv_atm",        self.IV_PRIOR)
        td.setdefault("clumsy_signal", 0.0)
        td.setdefault("clumsy_last_ts", ts)
        td.setdefault("last_fill_ts",  {})    # per-strike anti-stagnation
        td.setdefault("unknown",       [])

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

        # ── Detect own fills (anti-stagnation reset) + Mark trades ──────────
        for prod, trades in (state.market_trades or {}).items():
            if not trades: continue
            for tr in trades:
                buyer  = getattr(tr, "buyer",  "") or ""
                seller = getattr(tr, "seller", "") or ""
                qty    = abs(int(getattr(tr, "quantity", 0)))
                if buyer == "SUBMISSION" or seller == "SUBMISSION":
                    td["last_fill_ts"][prod] = ts

        # ── Mark clumsy signal update ───────────────────────────────────────
        elapsed_ts = max(0, ts - td["clumsy_last_ts"])
        if elapsed_ts > 0:
            half = self.CLUMSY_HALFLIFE_TICKS * 100
            td["clumsy_signal"] *= 0.5 ** (elapsed_ts / half)
        td["clumsy_last_ts"] = ts
        for tr in (state.market_trades or {}).get(self.UNDERLYING, []):
            buyer  = getattr(tr, "buyer",  "") or ""
            seller = getattr(tr, "seller", "") or ""
            qty    = abs(int(getattr(tr, "quantity", 0)))
            if qty < CLUMSY_QTY_THRESHOLD: continue
            if MARK_ROLES.get(buyer)  == "BIG_CLUMSY_LONG":  td["clumsy_signal"] += 1.0
            if MARK_ROLES.get(seller) == "BIG_CLUMSY_SHORT": td["clumsy_signal"] -= 1.0
        td["clumsy_signal"] = max(-3.0, min(3.0, td["clumsy_signal"]))
        clumsy = td["clumsy_signal"]

        # ── TTE bootstrap ───────────────────────────────────────────────────
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
                    sig_est = max(self.IV_MIN, min(self.IV_MAX, sum(ivs)/len(ivs)))
                ttes = [_infer_tte(c, S, k, sig_est) for k, c in samples]
                ttes = [x for x in ttes if x is not None]
                if ttes:
                    T_est = max(0.01, min(self.TTE_START, sum(ttes)/len(ttes)))
            td["T_calib"]    = T_est
            td["T_calib_ts"] = ts
            td["iv_atm"]     = sig_est

        elapsed = (ts - td["T_calib_ts"]) / self.TICKS_PER_DAY
        T = max(0.0, td["T_calib"] - elapsed)

        # ── Smile fit ──────────────────────────────────────────────────────
        smile_xs, smile_ys = [], []
        for k in self.STRIKES_FIT:
            p = f"VEV_{k}"
            if p not in mids: continue
            v = _implied_vol(mids[p][0], S, float(k), T)
            if v is None or v < self.IV_MIN or v > self.IV_MAX: continue
            smile_xs.append(math.log(float(k)/S))
            smile_ys.append(v)
        smile_now = _fit_quadratic(smile_xs, smile_ys) if len(smile_xs) >= 3 else None
        if smile_now is not None:
            if td["smile"] is None:
                td["smile"] = list(smile_now)
            else:
                a_s,b_s,c_s = td["smile"]; a_n,b_n,c_n = smile_now
                a = self.SMILE_EMA_ALPHA
                td["smile"] = [a_s+a*(a_n-a_s), b_s+a*(b_n-b_s), c_s+a*(c_n-c_s)]
        smile = td["smile"]
        if smile is not None:
            td["iv_atm"] = max(self.IV_MIN, min(self.IV_MAX, smile[0]))

        def smile_iv(K):
            if smile is None or S <= 0: return td["iv_atm"]
            x = math.log(K/S)
            return max(self.IV_MIN, min(self.IV_MAX,
                                        smile[0] + smile[1]*x + smile[2]*x*x))

        # ── Aggregate book greeks ──────────────────────────────────────────
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
            elif ptype == "BINARY_CALL":
                book_delta += p * (-_bs_digital_put_delta(S, float(K), T, iv_k))
            # KO/CHOOSER: dormant until products appear

        # VFE position contributes 1.0 delta per share.
        vfe_pos = state.position.get(self.UNDERLYING, 0)
        portfolio_delta = book_delta + float(vfe_pos)

        def can_buy(d, v, g=0.0):
            new_d = portfolio_delta + d
            new_v = book_vega + v
            new_g = book_gamma + g
            if new_d > self.DELTA_CAP and new_d > portfolio_delta: return False
            if abs(new_v) > self.VEGA_CAP and abs(new_v) > abs(book_vega): return False
            if abs(new_g) > self.GAMMA_CAP and abs(new_g) > abs(book_gamma): return False
            return True

        def can_sell(d, v, g=0.0):
            new_d = portfolio_delta - d
            new_v = book_vega - v
            new_g = book_gamma - g
            if new_d < -self.DELTA_CAP and new_d < portfolio_delta: return False
            if abs(new_v) > self.VEGA_CAP and abs(new_v) > abs(book_vega): return False
            if abs(new_g) > self.GAMMA_CAP and abs(new_g) > abs(book_gamma): return False
            return True

        # ── Per-product trading ─────────────────────────────────────────────

        # HP
        if self.HP_PRODUCT in state.order_depths:
            result[self.HP_PRODUCT] = self._trade_hp(
                state.order_depths[self.HP_PRODUCT],
                state.position.get(self.HP_PRODUCT, 0))

        # VFE — uses options book_delta as a hedge bias
        if self.UNDERLYING in state.order_depths:
            result[self.UNDERLYING] = self._trade_vfe(
                state.order_depths[self.UNDERLYING],
                vfe_pos, book_delta, clumsy)

        # VEV options
        for product, od in state.order_depths.items():
            ptype = detect_type(product)
            if ptype != "CALL": continue
            K = parse_strike(product)
            if K is None or K not in self.STRIKES_MM: continue
            if product not in mids: continue
            pos = state.position.get(product, 0)
            mid_k, bb, ba = mids[product]
            iv_k = smile_iv(float(K))
            fair = _bs_call(S, float(K), T, iv_k)
            d_k  = _bs_call_delta(S, float(K), T, iv_k)
            v_k  = _bs_vega(S, float(K), T, iv_k)
            g_k  = _bs_gamma(S, float(K), T, iv_k)
            last_fill = td["last_fill_ts"].get(product, ts)
            stagnation_ticks = max(0, ts - last_fill)
            result[product] = self._trade_call(
                product, K, od, pos, bb, ba, mid_k,
                fair, d_k, v_k, g_k, T, clumsy, stagnation_ticks,
                can_buy, can_sell)

        # Detect any unknown products (Round-5 readiness)
        for product in state.order_depths:
            if detect_type(product) == "UNKNOWN" and product not in td["unknown"]:
                td["unknown"].append(product)

        # Snapshot
        td["debug"] = {
            "ts": ts, "T": round(T, 4), "S": round(S, 2),
            "vfe_mid": round(vfe_mid, 2),
            "iv_atm": round(td["iv_atm"], 5),
            "smile": None if smile is None else
                     [round(smile[0],5), round(smile[1],5), round(smile[2],5)],
            "book_delta": round(book_delta, 1),
            "vfe_pos": vfe_pos,
            "port_delta": round(portfolio_delta, 1),
            "book_vega": round(book_vega, 0),
            "book_gamma": round(book_gamma, 1),
            "clumsy": round(clumsy, 2),
        }
        return result, conversions, json.dumps(td)

    # ── HP strategy ──────────────────────────────────────────────────────────

    def _trade_hp(self, od: OrderDepth, pos: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[self.HP_PRODUCT]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None: return orders

        inv  = pos / limit
        skew = int(round(inv * self.HP_SKEW_TICKS))

        bv1 = od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        vbi = (bv1 - av1) / max(1, bv1 + av1)
        denom = max(1e-9, self.HP_VBI_HARD - self.HP_VBI_SOFT)
        strength = max(0.0, min(1.0, (abs(vbi) - self.HP_VBI_SOFT) / denom))

        bid_mult = ask_mult = 1.0
        flow_shift = 0
        if vbi > self.HP_VBI_SOFT:
            ask_mult = max(0.0, 1.0 - strength)
            bid_mult = 1.0 + self.HP_VBI_BOOST * strength
            flow_shift = int(round(self.HP_VBI_PRICE_TKS * strength))
        elif vbi < -self.HP_VBI_SOFT:
            bid_mult = max(0.0, 1.0 - strength)
            ask_mult = 1.0 + self.HP_VBI_BOOST * strength
            flow_shift = -int(round(self.HP_VBI_PRICE_TKS * strength))

        if inv > self.HP_THROTTLE_FRAC:
            avail = od.buy_orders.get(bb, 0)
            target_excess = pos - int(self.HP_THROTTLE_FRAC * limit)
            qty = min(self.HP_AGGR_SIZE, avail, target_excess, limit + pos)
            if qty > 0:
                orders.append(Order(self.HP_PRODUCT, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.HP_SIZE, limit + pos - qty)
            flow_shift = 0
        elif inv < -self.HP_THROTTLE_FRAC:
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.HP_THROTTLE_FRAC * limit)
            qty = min(self.HP_AGGR_SIZE, avail, target_excess, limit - pos)
            if qty > 0:
                orders.append(Order(self.HP_PRODUCT, ba, qty))
            ask_qty = 0
            bid_qty = min(self.HP_SIZE, limit - pos - qty)
            flow_shift = 0
        else:
            bid_add_frac = max(0.0,  inv) / self.HP_THROTTLE_FRAC
            ask_add_frac = max(0.0, -inv) / self.HP_THROTTLE_FRAC
            bid_qty = int(self.HP_SIZE * (1.0 - bid_add_frac) * bid_mult)
            ask_qty = int(self.HP_SIZE * (1.0 - ask_add_frac) * ask_mult)
            bid_qty = min(bid_qty, limit - pos)
            ask_qty = min(ask_qty, limit + pos)

        our_bid = bb + 1 - skew + flow_shift
        our_ask = ba - 1 - skew + flow_shift
        our_bid = min(our_bid, ba - 1)
        our_ask = max(our_ask, bb + 1)
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if bid_qty > 0:
            orders.append(Order(self.HP_PRODUCT, our_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order(self.HP_PRODUCT, our_ask, -ask_qty))
        return orders

    # ── VFE strategy ─────────────────────────────────────────────────────────

    def _trade_vfe(self, od: OrderDepth, pos: int,
                   options_book_delta: float, clumsy: float) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[self.UNDERLYING]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None: return orders

        inv  = pos / limit
        skew = int(round(inv * self.VFE_SKEW_TICKS))

        bv1 = od.buy_orders.get(bb, 0)
        av1 = -od.sell_orders.get(ba, 0)
        vbi = (bv1 - av1) / max(1, bv1 + av1)
        denom = max(1e-9, self.VFE_VBI_HARD - self.VFE_VBI_SOFT)
        strength = max(0.0, min(1.0, (abs(vbi) - self.VFE_VBI_SOFT) / denom))
        bid_mult = ask_mult = 1.0
        vbi_shift = 0
        if vbi > self.VFE_VBI_SOFT:
            ask_mult = max(0.0, 1.0 - strength)
            bid_mult = 1.0 + self.VFE_VBI_BOOST * strength
            vbi_shift = int(round(self.VFE_VBI_PRICE_TKS * strength))
        elif vbi < -self.VFE_VBI_SOFT:
            bid_mult = max(0.0, 1.0 - strength)
            ask_mult = 1.0 + self.VFE_VBI_BOOST * strength
            vbi_shift = -int(round(self.VFE_VBI_PRICE_TKS * strength))

        # CROSS-PRODUCT DELTA HEDGE.
        # If options book is long-delta, we want VFE to lean SHORT to flatten:
        # bid less likely to fill, ask more likely. Achieved by shifting both
        # quotes DOWN.
        hedge_shift = -int(round(options_book_delta / 100.0 *
                                  self.VFE_HEDGE_TICKS_PER_100D))
        # Clumsy alignment: positive clumsy → VFE up → BUY-bias.
        clumsy_shift = int(round(clumsy * 1.5))   # max ±4.5

        flow_shift = vbi_shift + hedge_shift + clumsy_shift

        # Hedge volume bias: when options are very long delta, upsize VFE asks.
        hedge_frac = max(-1.0, min(1.0, options_book_delta /
                                          max(1.0, self.DELTA_CAP)))
        if hedge_frac > 0:
            ask_mult *= 1.0 + self.VFE_HEDGE_VOLUME_FRAC * hedge_frac
            bid_mult *= max(0.2, 1.0 - self.VFE_HEDGE_VOLUME_FRAC * hedge_frac)
        elif hedge_frac < 0:
            bid_mult *= 1.0 + self.VFE_HEDGE_VOLUME_FRAC * (-hedge_frac)
            ask_mult *= max(0.2, 1.0 - self.VFE_HEDGE_VOLUME_FRAC * (-hedge_frac))

        if inv > self.VFE_THROTTLE_FRAC:
            avail = od.buy_orders.get(bb, 0)
            target_excess = pos - int(self.VFE_THROTTLE_FRAC * limit)
            qty = min(self.VFE_AGGR_SIZE, avail, target_excess, limit + pos)
            if qty > 0:
                orders.append(Order(self.UNDERLYING, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.VFE_SIZE, limit + pos - qty)
            flow_shift = 0
        elif inv < -self.VFE_THROTTLE_FRAC:
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.VFE_THROTTLE_FRAC * limit)
            qty = min(self.VFE_AGGR_SIZE, avail, target_excess, limit - pos)
            if qty > 0:
                orders.append(Order(self.UNDERLYING, ba, qty))
            ask_qty = 0
            bid_qty = min(self.VFE_SIZE, limit - pos - qty)
            flow_shift = 0
        else:
            bid_add_frac = max(0.0,  inv) / self.VFE_THROTTLE_FRAC
            ask_add_frac = max(0.0, -inv) / self.VFE_THROTTLE_FRAC
            bid_qty = int(self.VFE_SIZE * (1.0 - bid_add_frac) * bid_mult)
            ask_qty = int(self.VFE_SIZE * (1.0 - ask_add_frac) * ask_mult)
            bid_qty = min(bid_qty, limit - pos)
            ask_qty = min(ask_qty, limit + pos)

        our_bid = bb + 1 - skew + flow_shift
        our_ask = ba - 1 - skew + flow_shift
        our_bid = min(our_bid, ba - 1)
        our_ask = max(our_ask, bb + 1)
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        if bid_qty > 0:
            orders.append(Order(self.UNDERLYING, our_bid,  bid_qty))
        if ask_qty > 0:
            orders.append(Order(self.UNDERLYING, our_ask, -ask_qty))
        return orders

    # ── VEV call strategy (smile + clumsy + anti-stagnation) ─────────────────

    def _trade_call(
        self, prod: str, K: int, od: OrderDepth, pos: int,
        bb: int, ba: int, mid: float,
        fair: float, delta: float, vega: float, gamma: float,
        T: float, clumsy: float, stagnation_ticks: int,
        can_buy, can_sell,
    ) -> List[Order]:

        orders: List[Order] = []
        limit = self.POSITION_LIMITS[prod]
        spread = ba - bb
        is_itm = K <= 4500
        is_atm = 5000 <= K <= 5300

        loaded_long  = pos >  self.PER_STRIKE_THROTTLE * limit
        loaded_short = pos < -self.PER_STRIKE_THROTTLE * limit

        # Aggressive takes — clumsy-scaled threshold.
        if T > 0:
            base_thr = max(2.0 if is_itm else self.EDGE_TAKE_PRICE,
                           self.EDGE_TAKE_IV_RESID * vega)
            buy_thr  = base_thr * max(0.20, 1.0 - self.CLUMSY_TAKE_BOOST*max(0.0, clumsy))
            sell_thr = base_thr * max(0.20, 1.0 - self.CLUMSY_TAKE_BOOST*max(0.0, -clumsy))
            edge_buy  = fair - ba
            edge_sell = bb - fair

            if not loaded_long and edge_buy > buy_thr:
                avail = -od.sell_orders[ba]
                room  = limit - pos
                q = min(avail, self.AGGRO_SIZE_MAX, room)
                while q > 0 and not can_buy(q*delta, q*vega, q*gamma): q -= 1
                if q > 0:
                    orders.append(Order(prod, ba, q))
                    pos += q
            if not loaded_short and edge_sell > sell_thr:
                avail = od.buy_orders[bb]
                room  = limit + pos
                q = min(avail, self.AGGRO_SIZE_MAX, room)
                while q > 0 and not can_sell(q*delta, q*vega, q*gamma): q -= 1
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
        sig_skew = int(round(max(-2, min(2, signal / 1.0))))
        clumsy_skew = int(round(max(-self.CLUMSY_SKEW_MAX_TICKS,
                                    min(self.CLUMSY_SKEW_MAX_TICKS,
                                        clumsy * self.CLUMSY_SKEW_MAX_TICKS / 2.0))))

        # Anti-stagnation march: after no-fill threshold, walk our quotes
        # 1 tick toward the touch every STAGNATION_STEP_TICKS, both sides.
        march = 0
        if stagnation_ticks > self.STAGNATION_NOFILL_TICKS:
            extra = (stagnation_ticks - self.STAGNATION_NOFILL_TICKS) // self.STAGNATION_STEP_TICKS
            march = min(self.STAGNATION_MAX_MARCH, 1 + int(extra))

        if spread <= 2:
            inv_skew = int(round(inv * 1.0))
            net = sig_skew - inv_skew + clumsy_skew
            our_bid, our_ask = bb, ba
            if net > 0:    our_ask = ba + 1
            elif net < 0:  our_bid = bb - 1
        else:
            edge_in = max(1, (2 if is_itm else 1) - march)   # march eats edge
            base_bid = bb + edge_in
            base_ask = ba - edge_in
            inv_skew = int(round(inv * (spread / 2.0)))
            net = sig_skew - inv_skew + clumsy_skew
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

        # Clumsy volume bias.
        if clumsy >= 1.0:
            bid_qty = int(bid_qty * 1.5)
            ask_qty = int(ask_qty * 0.5)
        elif clumsy <= -1.0:
            bid_qty = int(bid_qty * 0.5)
            ask_qty = int(ask_qty * 1.5)

        # Stagnation upsize: after no fills, also bump size.
        if stagnation_ticks > self.STAGNATION_NOFILL_TICKS:
            bid_qty = int(bid_qty * 1.3)
            ask_qty = int(ask_qty * 1.3)

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