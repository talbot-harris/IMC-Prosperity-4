"""
IMC Prosperity Round 4 — FINAL V.2

Submission 524039 (FINAL V.1) made +$2,264 total — only ~$1,038 of that
on options (all on VEV_5300). Volume forensic on the trade history:

  Per-strike PnL:
    HP    +950   (28 fills, 34 q/fill)
    VFE   +100   (13 fills, 5 q/fill)         ← stagnation, max gap 36,300 ticks
    4000  +195   (10 fills)
    4500  +148   (10 fills)
    5000  -54    (15 fills)  ← ATM bleed
    5100  -64    (19 fills)  ← ATM bleed
    5200  -50    (15 fills)  ← ATM bleed
    5300  +1,038 (30 fills, 262 qty short into a 58→39 drop) ← THE win
    5400  0      (no fills at all)

  Counterparty bias on our fills:
    Mark 01:  0 buys / 12 sells from us. Pure call-BUYER (took our offers).
    Mark 14:  0 buys / 16 sells from us. Same — call-BUYER.
    Mark 22: 22 buys /  4 sells.         Pure call-SELLER (hit our bids).
    Mark 38: 37 buys / 36 sells.         Balanced MM noise.
    Mark 55:  8 buys /  5 sells.         Roughly balanced.

  Two interpretations are forced by this:
    1. Mark 01 + Mark 14 systematically OVERPAY for calls. Every contract
       we sold to them ended worth less by EOD. Being short at their fills
       is structurally +EV.
    2. Mark 22 systematically UNDERSELLS calls. Buying from them is +EV
       only when calls actually rise — which on a down-trending day they
       did not. So being long from Mark 22 was the bleed source on
       5000–5200.

  Net structural rule: BIAS NET-SHORT CALLS. Quote bigger asks than bids
  on every strike, and refuse aggressive longs unless we have a clear
  uptrend signal. The losing 5000–5200 fills came from buying into a
  trend (smile said "cheap relative to fair", actually just falling).

V.2 changes vs V.1:

  1. REGIME GATE on every option trade.
     drift = vfe_fast_ema − vfe_slow_ema   (alpha 0.05 vs 0.005)
     conviction = clip(drift / 5, -1, 1) * 1.0  +  clumsy_signal * 0.7
     |conviction| > 0.7 → directional mode (size up aligned side, kill other)
     |conviction| < 0.3 → mean-reversion mode (smile MM at half size)
     middle band → stand down (skip bids on ATM, keep asks)

  2. STRUCTURAL SHORT BIAS.
     Default ask size is 1.5× bid size on every call strike.
     When conviction < 0 (down-trend): bid size → 0, ask size → 2×.
     When conviction > +0.7 (clear up-trend): allow symmetric quoting.

  3. ROUND-3 EDGE_TAKE LADDER.
     EDGE_TAKE_SELL = 1.5  — we hit the bid aggressively on small mispricing.
     EDGE_TAKE_BUY  = 5.0  — we ONLY lift the offer on big mispricing
                              (need conviction to be long).
     V.1 had a single 0.4 threshold for both sides — that's why we got long
     on 5000–5200 noise.

  4. JOIN-THE-TOUCH STAGNATION FIX.
     Bots in this market only trade at the touch (bb / ba). Quoting at
     bb+1/ba-1 means we never fill except on Mark sweeps. After 1500
     ticks with no fill on a strike, we collapse our quote to AT bb / AT
     ba. Adverse-selection cost is repaid in volume — and the tape shows
     the alternative is sitting on $0 PnL for the whole day.

  5. VFE DIRECT MEAN-REVERSION (round-3 strategy).
     New aggressive take on VFE itself: when |vfe_mid − vfe_slow_ema| > 2.5,
     cross the spread to revert. This worked in round 3 (+1,703 on VFE
     alone). Combined with the existing VBI gate.

  6. HP FURTHER BOOSTED.
     SIZE 80 → 120, THROTTLE 0.70 → 0.85, VBI_BOOST 1.5 → 2.5.

  7. KILL THE BLEEDERS PASSIVELY.
     ATM bid sizes (5000–5200) are 30 → 12 in default regime, jumping to
     50 only when conviction > +0.7. ATM ask sizes are 30 → 50.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import math
import json


# ── Black–Scholes core ───────────────────────────────────────────────────────

SQRT_2PI = math.sqrt(2.0 * math.pi)

def _ncdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = math.exp(-0.5*x*x) / SQRT_2PI
    poly = t*(0.319381530+t*(-0.356563782+t*(1.781477937
                +t*(-1.821255978+t*1.330274429))))
    p = 1.0 - d*poly
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

# Exotic pricers (round-5 ready) ---------------------------------------------

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
    sv = sigma * math.sqrt(T); lam = 0.5
    if B < K:
        x1 = math.log(S/B)/sv + lam*sv
        y  = math.log(B*B/(S*K))/sv + lam*sv
        y1 = math.log(B/S)/sv + lam*sv
        BoS_2lam   = (B/S)**(2*lam)
        BoS_2lam_2 = (B/S)**(2*lam-2)
        p_di = (-S*_ncdf(-x1) + K*_ncdf(-x1+sv)
                + S*BoS_2lam*(_ncdf(y)-_ncdf(y1))
                - K*BoS_2lam_2*(_ncdf(y-sv)-_ncdf(y1-sv)))
        return max(0.0, _bs_put(S, K, T, sigma) - p_di)
    refl = B*B/S
    return max(0.0, min(_bs_put(S,K,T,sigma),
                        _bs_put(S,K,T,sigma) - (S/B)*_bs_put(refl,K,T,sigma)))

def _chooser_value(S, K, T_total, T_choose, sigma):
    if T_choose >= T_total: T_choose = T_total - 1e-6
    if T_choose <= 0:
        return max(_bs_call(S,K,T_total,sigma), _bs_put(S,K,T_total,sigma))
    return _bs_call(S,K,T_total,sigma) + _bs_put(S,K,T_choose,sigma)

# Calibration helpers --------------------------------------------------------

def _implied_vol(price, S, K, T):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or T <= 0 or S <= 0: return None
    lo, hi = 1e-4, 5.0
    for _ in range(50):
        m = (lo + hi) / 2
        if _bs_call(S,K,T,m) < price: lo = m
        else: hi = m
    return (lo + hi) / 2

def _infer_tte(price, S, K, sigma, lo=0.01, hi=6.0):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or S <= 0 or sigma <= 0: return None
    if _bs_call(S,K,hi,sigma) < price: return hi
    if _bs_call(S,K,lo,sigma) > price: return lo
    for _ in range(50):
        m = (lo + hi) / 2
        if _bs_call(S,K,m,sigma) < price: lo = m
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


# ── Counterparty profiles (FORENSIC, from 524039) ────────────────────────────

# Marks who BUY calls from us (we want to sell) — call-overpayers.
MARK_CALL_BUYERS = {"Mark 01", "Mark 14"}
# Marks who SELL calls to us — call-underprice-rs.
MARK_CALL_SELLERS = {"Mark 22"}
# Big-clumsy VFE traders (signal-generating).
MARK_ROLES = {
    "Mark 01": "OTM_FLOW",
    "Mark 14": "MM_LIKE",
    "Mark 22": "OTM_FLOW",
    "Mark 38": "MM_LIKE",
    "Mark 49": "BIG_CLUMSY_SHORT",
    "Mark 55": "MM_LIKE",
    "Mark 67": "BIG_CLUMSY_LONG",
}
CLUMSY_QTY_THRESHOLD = 8


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
    STRIKES_MM   = [4000, 4500, 5000, 5100, 5200, 5300, 5400]
    STRIKES_FIT  = [5000, 5100, 5200, 5300, 5400]
    STRIKES_DEAD = {5500, 6000, 6500}

    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        **{f"VEV_{k}": 300 for k in STRIKES_ALL},
    }

    TICKS_PER_DAY = 100_000
    TTE_START     = 4.0

    # ── HP ──────────────────────────────────────────────────────────────────
    HP_SIZE          = 1200        # 10× — limit clips at 200 anyway
    HP_SKEW_TICKS    = 8
    HP_THROTTLE_FRAC = 0.85
    HP_AGGR_SIZE     = 600         # 10×
    HP_VBI_SOFT      = 0.04
    HP_VBI_HARD      = 0.18
    HP_VBI_BOOST     = 2.5
    HP_VBI_PRICE_TKS = 4

    # ── VFE ─────────────────────────────────────────────────────────────────
    # Submission 524039 made only +$100 on VFE in 13 fills — wildly under-
    # used the limit. V.2 keeps a simpler structure: passive MM + a gentle
    # VBI lean + a CONVICTION-aligned directional skew. No more slow-EMA
    # aggressive take (regime-fragile, killed in V.2 sim).
    VFE_SIZE          = 800        # 10×
    VFE_SKEW_TICKS    = 6
    VFE_THROTTLE_FRAC = 0.75
    VFE_AGGR_SIZE     = 500        # 10×
    VFE_VBI_SOFT      = 0.05
    VFE_VBI_HARD      = 0.20
    VFE_VBI_BOOST     = 0.50
    VFE_VBI_PRICE_TKS = 2
    VFE_FAST_ALPHA = 0.05
    VFE_SLOW_ALPHA = 0.005
    VFE_HEDGE_TICKS_PER_100D = 2.0
    VFE_HEDGE_VOL_FRAC       = 0.5
    VFE_CONV_SHIFT_TICKS = 3
    VFE_CONV_VOL_FRAC    = 0.6

    # ── VEV options ─────────────────────────────────────────────────────────
    IV_PRIOR = 0.0195
    IV_MIN   = 0.005
    IV_MAX   = 0.10
    SMILE_EMA_ALPHA = 0.20
    S_EMA_ALPHA     = 0.5

    # Asymmetric edge thresholds (V.2 main change).
    EDGE_TAKE_SELL_PRICE = 1.5    # hit bid easily — selling calls is +EV
    EDGE_TAKE_BUY_PRICE  = 5.0    # lift offer ONLY on big edge (avoid bleed)
    EDGE_TAKE_IV_RESID   = 0.0003

    PER_STRIKE_THROTTLE  = 0.85
    DELTA_CAP            = 500.0
    VEGA_CAP             = 500_000.0
    GAMMA_CAP            = 600.0

    # Default sizes (signal-neutral regime). All 10× — VEV limit clips at 300.
    # ATM bid stays 0: the structural-bleed avoidance, not a size knob.
    PASSIVE_BID_ATM      = 0      # intentionally 0 — ATM longs are the bleed
    PASSIVE_ASK_ATM      = 600    # 10×
    PASSIVE_BID_ITM      = 200    # 10×
    PASSIVE_ASK_ITM      = 300    # 10×
    PASSIVE_BID_OTM      = 50     # 10×
    PASSIVE_ASK_OTM      = 800    # 10×

    # Directional regime sizes (|conviction| > 0.7).
    DIRECTIONAL_BID_BOOST = 2.5   # when uptrend conviction
    DIRECTIONAL_ASK_BOOST = 2.5   # when downtrend conviction

    AGGRO_SIZE_MAX       = 400    # 10× — limit clips at 300 anyway
    PIN_T_THRESHOLD      = 0.25
    MIN_PASSIVE_SPREAD   = 2
    MIN_ITM_PASSIVE_SPREAD = 3

    # Mark + drift signals.
    CLUMSY_HALFLIFE_TICKS = 500
    CLUMSY_SKEW_MAX_TICKS = 4
    CLUMSY_TAKE_BOOST     = 0.80
    DRIFT_TO_CONV_DIVISOR = 5.0   # drift / 5 → ±1 normalised
    CLUMSY_TO_CONV_GAIN   = 0.7
    CONV_DIRECTIONAL_THR  = 0.7
    CONV_NEUTRAL_THR      = 0.3

    # Stagnation: real fix is JOIN THE TOUCH after no-fill threshold.
    STAGNATION_NOFILL_TICKS = 1500
    STAGNATION_MAX_MARCH    = 99   # effectively unlimited; we just join

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
        td.setdefault("vfe_fast",      None)
        td.setdefault("vfe_slow",      None)
        td.setdefault("smile",         None)
        td.setdefault("iv_atm",        self.IV_PRIOR)
        td.setdefault("clumsy_signal", 0.0)
        td.setdefault("clumsy_last_ts", ts)
        td.setdefault("last_fill_ts",  {})
        td.setdefault("unknown",       [])

        # Mids
        mids: Dict[str, Tuple[float, int, int]] = {}
        for product, od in state.order_depths.items():
            bb = max(od.buy_orders)  if od.buy_orders  else None
            ba = min(od.sell_orders) if od.sell_orders else None
            if bb is not None and ba is not None:
                mids[product] = ((bb + ba) / 2.0, bb, ba)
        if self.UNDERLYING not in mids:
            return result, conversions, json.dumps(td)
        vfe_mid, _, _ = mids[self.UNDERLYING]

        # S, fast, slow EMAs
        if td["s_ema"] is None:    td["s_ema"]    = vfe_mid
        else: td["s_ema"]    += self.S_EMA_ALPHA * (vfe_mid - td["s_ema"])
        if td["vfe_fast"] is None: td["vfe_fast"] = vfe_mid
        else: td["vfe_fast"] += self.VFE_FAST_ALPHA * (vfe_mid - td["vfe_fast"])
        if td["vfe_slow"] is None: td["vfe_slow"] = vfe_mid
        else: td["vfe_slow"] += self.VFE_SLOW_ALPHA * (vfe_mid - td["vfe_slow"])
        S          = td["s_ema"]
        vfe_drift  = td["vfe_fast"] - td["vfe_slow"]

        # Track our own fills for anti-stagnation.
        for prod, trades in (state.market_trades or {}).items():
            if not trades: continue
            for tr in trades:
                if (getattr(tr,"buyer","") == "SUBMISSION"
                    or getattr(tr,"seller","") == "SUBMISSION"):
                    td["last_fill_ts"][prod] = ts

        # Clumsy signal
        elapsed_ts = max(0, ts - td["clumsy_last_ts"])
        if elapsed_ts > 0:
            half = self.CLUMSY_HALFLIFE_TICKS * 100
            td["clumsy_signal"] *= 0.5 ** (elapsed_ts / half)
        td["clumsy_last_ts"] = ts
        for tr in (state.market_trades or {}).get(self.UNDERLYING, []):
            buyer  = getattr(tr,"buyer","") or ""
            seller = getattr(tr,"seller","") or ""
            qty    = abs(int(getattr(tr,"quantity",0)))
            if qty < CLUMSY_QTY_THRESHOLD: continue
            if MARK_ROLES.get(buyer)  == "BIG_CLUMSY_LONG":  td["clumsy_signal"] += 1.0
            if MARK_ROLES.get(seller) == "BIG_CLUMSY_SHORT": td["clumsy_signal"] -= 1.0
        td["clumsy_signal"] = max(-3.0, min(3.0, td["clumsy_signal"]))
        clumsy = td["clumsy_signal"]

        # Conviction = drift + clumsy. Clipped to [-2, 2].
        drift_norm = max(-1.0, min(1.0, vfe_drift / self.DRIFT_TO_CONV_DIVISOR))
        conviction = drift_norm + self.CLUMSY_TO_CONV_GAIN * clumsy
        conviction = max(-2.0, min(2.0, conviction))

        # TTE
        if "T_calib" not in td:
            T_est, sig_est = 2.0, self.IV_PRIOR
            for _ in range(6):
                samples = []
                for k in (5000, 5100, 5200, 5300):
                    p = f"VEV_{k}"
                    if p in mids: samples.append((float(k), mids[p][0]))
                if not samples: break
                ivs = [_implied_vol(c,S,k,T_est) for k,c in samples]
                ivs = [x for x in ivs if x is not None]
                if ivs:
                    sig_est = max(self.IV_MIN, min(self.IV_MAX, sum(ivs)/len(ivs)))
                ttes = [_infer_tte(c,S,k,sig_est) for k,c in samples]
                ttes = [x for x in ttes if x is not None]
                if ttes:
                    T_est = max(0.01, min(self.TTE_START, sum(ttes)/len(ttes)))
            td["T_calib"]    = T_est
            td["T_calib_ts"] = ts
            td["iv_atm"]     = sig_est
        elapsed = (ts - td["T_calib_ts"]) / self.TICKS_PER_DAY
        T = max(0.0, td["T_calib"] - elapsed)

        # Smile
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
            if td["smile"] is None: td["smile"] = list(smile_now)
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

        # Aggregate book greeks
        book_delta = book_vega = book_gamma = 0.0
        for prod, p in state.position.items():
            if p == 0 or T <= 0: continue
            ptype = detect_type(prod)
            K = parse_strike(prod) or 0
            iv_k = smile_iv(float(K)) if K > 0 else td["iv_atm"]
            if ptype == "CALL":
                book_delta += p * _bs_call_delta(S,float(K),T,iv_k)
                book_vega  += p * _bs_vega(S,float(K),T,iv_k)
                book_gamma += p * _bs_gamma(S,float(K),T,iv_k)
            elif ptype == "BINARY_PUT":
                book_delta += p * _bs_digital_put_delta(S,float(K),T,iv_k)
            elif ptype == "BINARY_CALL":
                book_delta += p * (-_bs_digital_put_delta(S,float(K),T,iv_k))

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

        # ── Trade ──
        if self.HP_PRODUCT in state.order_depths:
            result[self.HP_PRODUCT] = self._trade_hp(
                state.order_depths[self.HP_PRODUCT],
                state.position.get(self.HP_PRODUCT, 0))

        if self.UNDERLYING in state.order_depths:
            result[self.UNDERLYING] = self._trade_vfe(
                state.order_depths[self.UNDERLYING],
                vfe_pos, book_delta, clumsy, vfe_drift, td["vfe_slow"],
                conviction)

        for product, od in state.order_depths.items():
            if detect_type(product) != "CALL": continue
            K = parse_strike(product)
            if K is None or K not in self.STRIKES_MM: continue
            if product not in mids: continue
            pos = state.position.get(product, 0)
            mid_k, bb, ba = mids[product]
            iv_k = smile_iv(float(K))
            fair = _bs_call(S,float(K),T,iv_k)
            d_k  = _bs_call_delta(S,float(K),T,iv_k)
            v_k  = _bs_vega(S,float(K),T,iv_k)
            g_k  = _bs_gamma(S,float(K),T,iv_k)
            stagnation = max(0, ts - td["last_fill_ts"].get(product, ts))
            result[product] = self._trade_call(
                product, K, od, pos, bb, ba, mid_k, S,
                fair, d_k, v_k, g_k, T, conviction, clumsy, stagnation,
                can_buy, can_sell)

        for product in state.order_depths:
            if detect_type(product) == "UNKNOWN" and product not in td["unknown"]:
                td["unknown"].append(product)

        td["debug"] = {
            "ts": ts, "T": round(T,4), "S": round(S,2),
            "vfe_drift": round(vfe_drift, 2),
            "iv_atm":  round(td["iv_atm"], 5),
            "book_delta": round(book_delta,1),
            "port_delta": round(portfolio_delta,1),
            "book_vega": round(book_vega,0),
            "book_gamma": round(book_gamma,1),
            "clumsy": round(clumsy,2),
            "conv":   round(conviction,2),
        }
        return result, conversions, json.dumps(td)

    # ── HP ───────────────────────────────────────────────────────────────────

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
            if qty > 0: orders.append(Order(self.HP_PRODUCT, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.HP_SIZE, limit + pos - qty)
            flow_shift = 0
        elif inv < -self.HP_THROTTLE_FRAC:
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.HP_THROTTLE_FRAC * limit)
            qty = min(self.HP_AGGR_SIZE, avail, target_excess, limit - pos)
            if qty > 0: orders.append(Order(self.HP_PRODUCT, ba, qty))
            ask_qty = 0
            bid_qty = min(self.HP_SIZE, limit - pos - qty)
            flow_shift = 0
        else:
            bf = max(0.0, inv) / self.HP_THROTTLE_FRAC
            af = max(0.0, -inv) / self.HP_THROTTLE_FRAC
            bid_qty = int(self.HP_SIZE * (1.0 - bf) * bid_mult)
            ask_qty = int(self.HP_SIZE * (1.0 - af) * ask_mult)
            bid_qty = min(bid_qty, limit - pos)
            ask_qty = min(ask_qty, limit + pos)

        our_bid = bb + 1 - skew + flow_shift
        our_ask = ba - 1 - skew + flow_shift
        our_bid = min(our_bid, ba - 1)
        our_ask = max(our_ask, bb + 1)
        if our_bid >= our_ask: our_bid = our_ask - 1

        if bid_qty > 0: orders.append(Order(self.HP_PRODUCT, our_bid,  bid_qty))
        if ask_qty > 0: orders.append(Order(self.HP_PRODUCT, our_ask, -ask_qty))
        return orders

    # ── VFE ──────────────────────────────────────────────────────────────────

    def _trade_vfe(self, od: OrderDepth, pos: int,
                   options_book_delta: float, clumsy: float,
                   vfe_drift: float, vfe_slow: float,
                   conviction: float = 0.0) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[self.UNDERLYING]
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None or ba is None: return orders

        # No round-3 slow-EMA MR take — regime-fragile (lost $70k in V.2 sim
        # on a trending day). Conviction-aligned directional skew below
        # carries the directional bet instead.

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

        # Cross-product hedge.
        hedge_shift = -int(round(options_book_delta / 100.0 *
                                  self.VFE_HEDGE_TICKS_PER_100D))
        clumsy_shift = int(round(clumsy * 1.5))
        # Conviction shift: positive conviction → push UP (long-bias).
        conv_shift = int(round(conviction * self.VFE_CONV_SHIFT_TICKS))
        # Conviction-aligned volume bias.
        if conviction >= 0.7:
            bid_mult *= 1.0 + self.VFE_CONV_VOL_FRAC
            ask_mult *= max(0.2, 1.0 - self.VFE_CONV_VOL_FRAC)
        elif conviction <= -0.7:
            ask_mult *= 1.0 + self.VFE_CONV_VOL_FRAC
            bid_mult *= max(0.2, 1.0 - self.VFE_CONV_VOL_FRAC)
        flow_shift = vbi_shift + hedge_shift + clumsy_shift + conv_shift

        hedge_frac = max(-1.0, min(1.0, options_book_delta /
                                          max(1.0, self.DELTA_CAP)))
        if hedge_frac > 0:
            ask_mult *= 1.0 + self.VFE_HEDGE_VOL_FRAC * hedge_frac
            bid_mult *= max(0.2, 1.0 - self.VFE_HEDGE_VOL_FRAC * hedge_frac)
        elif hedge_frac < 0:
            bid_mult *= 1.0 + self.VFE_HEDGE_VOL_FRAC * (-hedge_frac)
            ask_mult *= max(0.2, 1.0 - self.VFE_HEDGE_VOL_FRAC * (-hedge_frac))

        if inv > self.VFE_THROTTLE_FRAC:
            avail = od.buy_orders.get(bb, 0)
            target_excess = pos - int(self.VFE_THROTTLE_FRAC * limit)
            qty = min(self.VFE_AGGR_SIZE, avail, target_excess, limit + pos)
            if qty > 0: orders.append(Order(self.UNDERLYING, bb, -qty))
            bid_qty = 0
            ask_qty = min(self.VFE_SIZE, limit + pos - qty)
            flow_shift = 0
        elif inv < -self.VFE_THROTTLE_FRAC:
            avail = -od.sell_orders.get(ba, 0)
            target_excess = -pos - int(self.VFE_THROTTLE_FRAC * limit)
            qty = min(self.VFE_AGGR_SIZE, avail, target_excess, limit - pos)
            if qty > 0: orders.append(Order(self.UNDERLYING, ba, qty))
            ask_qty = 0
            bid_qty = min(self.VFE_SIZE, limit - pos - qty)
            flow_shift = 0
        else:
            bf = max(0.0,  inv) / self.VFE_THROTTLE_FRAC
            af = max(0.0, -inv) / self.VFE_THROTTLE_FRAC
            bid_qty = int(self.VFE_SIZE * (1.0 - bf) * bid_mult)
            ask_qty = int(self.VFE_SIZE * (1.0 - af) * ask_mult)
            bid_qty = min(bid_qty, limit - pos)
            ask_qty = min(ask_qty, limit + pos)

        our_bid = bb + 1 - skew + flow_shift
        our_ask = ba - 1 - skew + flow_shift
        our_bid = min(our_bid, ba - 1)
        our_ask = max(our_ask, bb + 1)
        if our_bid >= our_ask: our_bid = our_ask - 1

        if bid_qty > 0: orders.append(Order(self.UNDERLYING, our_bid,  bid_qty))
        if ask_qty > 0: orders.append(Order(self.UNDERLYING, our_ask, -ask_qty))
        return orders

    # ── VEV call (V.2: regime-gated, structural short bias, real stagnation) ─

    def _trade_call(
        self, prod: str, K: int, od: OrderDepth, pos: int,
        bb: int, ba: int, mid: float, S: float,
        fair: float, delta: float, vega: float, gamma: float,
        T: float, conviction: float, clumsy: float, stagnation_ticks: int,
        can_buy, can_sell,
    ) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[prod]
        spread = ba - bb
        is_itm = K <= 4500
        is_atm = 5000 <= K <= 5200
        is_otm = K >= 5300

        loaded_long  = pos >  self.PER_STRIKE_THROTTLE * limit
        loaded_short = pos < -self.PER_STRIKE_THROTTLE * limit

        # ── Aggressive takes (asymmetric thresholds) ────────────────────────
        if T > 0:
            iv_resid_thr = self.EDGE_TAKE_IV_RESID * vega
            buy_thr  = max(self.EDGE_TAKE_BUY_PRICE,  iv_resid_thr)
            sell_thr = max(self.EDGE_TAKE_SELL_PRICE, iv_resid_thr * 0.5)
            # Conviction modulates further — uptrend lowers buy_thr, downtrend
            # lowers sell_thr.
            if conviction > 0:
                buy_thr  *= max(0.20, 1.0 - 0.7*conviction)
            elif conviction < 0:
                sell_thr *= max(0.20, 1.0 - 0.7*(-conviction))

            edge_buy  = fair - ba
            edge_sell = bb - fair

            # Sell-side: aggressive when conviction down OR small edge.
            if not loaded_short and edge_sell > sell_thr:
                avail = od.buy_orders[bb]
                room  = limit + pos
                q = min(avail, self.AGGRO_SIZE_MAX, room)
                # Conviction boost: bigger size when downtrend confirmed
                if conviction <= -1.0:
                    q = min(int(q * 1.5), avail, room)
                while q > 0 and not can_sell(q*delta, q*vega, q*gamma): q -= 1
                if q > 0:
                    orders.append(Order(prod, bb, -q))
                    pos -= q

            # Buy-side: only when STRONG positive conviction, OR edge_buy huge.
            allow_buy = (conviction > self.CONV_DIRECTIONAL_THR
                         or edge_buy > 2*buy_thr)
            if allow_buy and not loaded_long and edge_buy > buy_thr:
                avail = -od.sell_orders[ba]
                room  = limit - pos
                q = min(avail, self.AGGRO_SIZE_MAX, room)
                if conviction >= 1.0:
                    q = min(int(q * 1.5), avail, room)
                while q > 0 and not can_buy(q*delta, q*vega, q*gamma): q -= 1
                if q > 0:
                    orders.append(Order(prod, ba, q))
                    pos += q

        # ── Passive MM with regime + structural bias ────────────────────────
        min_spread = self.MIN_ITM_PASSIVE_SPREAD if is_itm \
                     else self.MIN_PASSIVE_SPREAD
        if spread < min_spread:
            return orders

        # Default sizes by moneyness.
        if is_itm:
            bid_size_def, ask_size_def = self.PASSIVE_BID_ITM, self.PASSIVE_ASK_ITM
        elif is_atm:
            bid_size_def, ask_size_def = self.PASSIVE_BID_ATM, self.PASSIVE_ASK_ATM
        else:
            bid_size_def, ask_size_def = self.PASSIVE_BID_OTM, self.PASSIVE_ASK_OTM

        # Regime: scale by conviction.
        if conviction >= self.CONV_DIRECTIONAL_THR:
            # Uptrend → upsize bids, downsize asks.
            bid_size = int(bid_size_def * self.DIRECTIONAL_BID_BOOST)
            ask_size = int(ask_size_def * 0.5)
        elif conviction <= -self.CONV_DIRECTIONAL_THR:
            # Downtrend → upsize asks, downsize bids.
            bid_size = int(bid_size_def * 0.3)
            ask_size = int(ask_size_def * self.DIRECTIONAL_ASK_BOOST)
        elif abs(conviction) < self.CONV_NEUTRAL_THR:
            # Neutral: smile MM at default sizes (already short-biased on OTM).
            bid_size, ask_size = bid_size_def, ask_size_def
        else:
            # Middle band: lean per conviction sign.
            if conviction > 0:
                bid_size = int(bid_size_def * 1.3)
                ask_size = int(ask_size_def * 0.7)
            else:
                bid_size = int(bid_size_def * 0.5)
                ask_size = int(ask_size_def * 1.3)

        # Pin-risk damper near expiry.
        if is_atm and T < self.PIN_T_THRESHOLD:
            bid_size = max(2, bid_size // 2)
            ask_size = max(2, ask_size // 2)

        bid_qty = min(bid_size, limit - pos)
        ask_qty = min(ask_size, limit + pos)

        # Per-strike inventory taper on add side.
        inv = pos / limit
        thr = self.PER_STRIKE_THROTTLE
        bid_taper = max(0.0,  inv) / thr
        ask_taper = max(0.0, -inv) / thr
        bid_qty = int(bid_qty * (1.0 - min(1.0, bid_taper)))
        ask_qty = int(ask_qty * (1.0 - min(1.0, ask_taper)))
        if loaded_long:  bid_qty = 0
        if loaded_short: ask_qty = 0

        # Quote price.
        signal = fair - mid
        sig_skew = int(round(max(-2, min(2, signal))))
        clumsy_skew = int(round(max(-self.CLUMSY_SKEW_MAX_TICKS,
                                    min(self.CLUMSY_SKEW_MAX_TICKS,
                                        clumsy * self.CLUMSY_SKEW_MAX_TICKS / 2.0))))
        # Real stagnation fix: JOIN THE TOUCH after no-fill threshold.
        join_touch = stagnation_ticks > self.STAGNATION_NOFILL_TICKS

        if join_touch or spread <= 2:
            inv_skew = int(round(inv * 1.0))
            net = sig_skew - inv_skew + clumsy_skew
            our_bid, our_ask = bb, ba
            if net > 0:    our_ask = ba + 1
            elif net < 0:  our_bid = bb - 1
        else:
            edge_in = 2 if is_itm else 1
            base_bid = bb + edge_in
            base_ask = ba - edge_in
            inv_skew = int(round(inv * (spread / 2.0)))
            net = sig_skew - inv_skew + clumsy_skew
            our_bid = base_bid + net
            our_ask = base_ask + net
            our_bid = max(bb + 1, min(our_bid, ba - 2))
            our_ask = max(bb + 2, min(our_ask, ba - 1))
            if our_bid >= our_ask: our_ask = our_bid + 1
            if our_ask >= ba: return orders

        while bid_qty > 0 and not can_buy(bid_qty*delta, bid_qty*vega, bid_qty*gamma):
            bid_qty -= 1
        while ask_qty > 0 and not can_sell(ask_qty*delta, ask_qty*vega, ask_qty*gamma):
            ask_qty -= 1

        if bid_qty > 0: orders.append(Order(prod, our_bid,  bid_qty))
        if ask_qty > 0: orders.append(Order(prod, our_ask, -ask_qty))
        return orders
