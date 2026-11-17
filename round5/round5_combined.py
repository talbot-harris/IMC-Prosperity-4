"""
IMC Prosperity Round 5 — COMBINED ALGORITHM  (v7)
==================================================
v6 post-mortem: adaptive EMA strategy failed on volatile/noisy products because
EMA drift signals arrive AFTER the move — we enter at the top and get reversed.

v6 WINS kept (pairs signals that worked):
  ROBOT_IRONING      +3,123  LAUNDRY pairs (r=-0.949)       was 0
  ROBOT_MOPPING      +1,705  adaptive EMA                    was 0
  ROBOT_LAUNDRY      +3,938  explicit directional short       was +2,787
  SLEEP_POD_LAMB_WOOL+2,660  COTTON pairs (r=-0.888)         was 0
  TRANSLATOR_ASTRO_BLACK+1,178 SPACE_GRAY pairs (r=-0.832)   was 0
  GALAXY_SOUNDS_SOLAR_WINDS+56 adaptive (improved from -321)
  OXYGEN_SHAKE_MORNING_BREATH+27 adaptive (marginal, keep)

v6 LOSSES stripped (14 products, adaptive EMA failed):
  MICROCHIP_SQUARE        -13,011  GALAXY_SOUNDS_PLANETARY_RINGS -7,542
  PEBBLES_L                -5,260  GALAXY_SOUNDS_SOLAR_FLAMES    -3,742
  PEBBLES_M                -3,321  SLEEP_POD_NYLON               -2,871
  OXYGEN_SHAKE_EVENING_BREATH-1,939  MICROCHIP_TRIANGLE          -1,770
  ROBOT_DISHES               -703  GALAXY_SOUNDS_DARK_MATTER       -448
  ROBOT_VACUUMING             -51  SLEEP_POD_POLYESTER              -42
  GALAXY_SOUNDS_BLACK_HOLES   -33  SNACKPACK_VANILLA                -20

Active products: 36/50.
Grand total progression: v3=+36k  v4=+43k  v5=+62k  v6=+31k  v7 target ~+72k
Position limit: 10 per product (confirmed from sandbox).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

# ── Constants ─────────────────────────────────────────────────────────────────
LIMIT      = 10
MM_SIZE    = 7
DIR_MM     = 2
SKEW_T     = 2
SKEW_T_H   = 4

FAST_A     = 0.05
SLOW_A     = 0.003
SLOW_A_REG = 0.008
EMA_A      = 0.08

LIMITS: Dict[str, int] = {}

# Products confirmed net-negative across submissions — no orders placed
SKIP_PRODUCTS = {
    "MICROCHIP_SQUARE",                    # -13,011  adaptive EMA too late on high-vol
    "MICROCHIP_TRIANGLE",                  #  -1,770  OVAL pairs signal wrong direction
    "GALAXY_SOUNDS_PLANETARY_RINGS",       #  -7,542  adaptive failed
    "GALAXY_SOUNDS_SOLAR_FLAMES",          #  -3,742  adaptive failed
    "GALAXY_SOUNDS_DARK_MATTER",           #    -448  adaptive failed
    "GALAXY_SOUNDS_BLACK_HOLES",           #     -33  regime MM regressed (was +1,010)
    "PEBBLES_L",                           #  -5,260  adaptive failed (near-flat dp=-17)
    "PEBBLES_M",                           #  -3,321  XL pairs signal entered too late
    "SLEEP_POD_NYLON",                     #  -2,871  adaptive failed
    "SLEEP_POD_POLYESTER",                 #     -42  persistent small loss
    "OXYGEN_SHAKE_EVENING_BREATH",         #  -1,939  too choppy for adaptive
    "ROBOT_DISHES",                        #    -703  adaptive failed
    "ROBOT_VACUUMING",                     #     -51  improved from -790 but still negative
    "SNACKPACK_VANILLA",                   #     -20  persistent small loss
}

def _lim(p: str) -> int:
    return LIMITS.get(p, LIMIT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ema(td: dict, key: str, val: float, alpha: float) -> float:
    if td.get(key) is None:
        td[key] = val
    else:
        td[key] += alpha * (val - td[key])
    return td[key]


def _drift(td: dict, product: str) -> float:
    """Return fast-slow EMA drift for a product (0 if not yet initialised)."""
    f = td.get(product + "_f")
    s = td.get(product + "_s")
    if f is None or s is None:
        return 0.0
    return f - s


def _regime_mults(drift: float, threshold: float = 2.0, strong: float = 6.0):
    """
    (bid_mult, ask_mult) based on EMA drift.
    Downtrend → reduce bids to stop accumulating long inventory.
    Uptrend   → reduce asks to stop accumulating short inventory.
    """
    if drift < -strong:
        return 0.10, 1.60
    elif drift < -threshold:
        t = (drift + strong) / (strong - threshold)
        return max(0.10, 0.10 + 0.50 * t), min(1.60, 1.60 - 0.40 * t)
    elif drift > strong:
        return 1.60, 0.10
    elif drift > threshold:
        t = (drift - strong) / (strong - threshold)
        return min(1.60, 1.60 - 0.40 * t), max(0.10, 0.10 + 0.50 * t)
    return 1.0, 1.0


def _mm_core(product: str, od: OrderDepth, pos: int, td: dict,
             bid_tilt: int = 0, ask_tilt: int = 0,
             bid_mult: float = 1.0, ask_mult: float = 1.0,
             size: int = MM_SIZE, skew_t: int = SKEW_T) -> List[Order]:
    """Passive market-making with inventory skew, fair-value correction, regime scaling."""
    limit = _lim(product)
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    if ba <= bb:
        return []
    mid = (bb + ba) / 2.0

    ema = _ema(td, product + "_ema", mid, EMA_A)
    fair_skew = int(round(max(-1.0, min(1.0, (mid - ema) / 5.0))))
    inv = pos / limit
    inv_skew = int(round(inv * skew_t))

    bid_p = max(1, min(bb + 1 - inv_skew + bid_tilt - fair_skew, ba - 1))
    ask_p = max(bb + 1, ba - 1 - inv_skew + ask_tilt - fair_skew)
    if bid_p >= ask_p:
        bid_p = ask_p - 1

    bid_q = int(min(size * bid_mult, max(0.0, limit - pos)))
    ask_q = int(min(size * ask_mult, max(0.0, limit + pos)))

    if inv > 0.6:
        bid_q = int(bid_q * max(0.0, (1.0 - inv) / 0.4))
    if inv < -0.6:
        ask_q = int(ask_q * max(0.0, (1.0 + inv) / 0.4))

    orders = []
    if bid_q > 0:
        orders.append(Order(product, bid_p, bid_q))
    if ask_q > 0:
        orders.append(Order(product, ask_p, -ask_q))
    return orders


def _directional_core(product: str, od: OrderDepth, pos: int, td: dict,
                       side: int, frac: float, mm_sz: int = DIR_MM) -> List[Order]:
    """
    Trend-following: target = side × frac × LIMIT, confirmed by own EMA drift.
    target=0 when drift opposes direction (avoids early pre-positioning).
    Scales from 40% (drift just crossing guard=8) to 100% (drift >> guard+20).
    adj_pos fix: passive MM sized off remaining capacity after aggressive take.
    """
    limit = _lim(product)
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow
    guard = 8.0

    if side == 1:
        if drift >= guard:
            conf = min(1.0, 0.4 + (drift - guard) / 20.0)
            target = int(limit * frac * conf)
        elif drift > 0:
            target = int(limit * frac * 0.4)
        else:
            target = 0
    elif side == -1:
        if drift <= -guard:
            conf = min(1.0, 0.4 + (-drift - guard) / 20.0)
            target = -int(limit * frac * conf)
        elif drift < 0:
            target = -int(limit * frac * 0.4)
        else:
            target = 0
    else:
        target = 0

    orders = []
    adj_pos = pos

    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), limit - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))
            adj_pos += qty
    elif pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), limit + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))
            adj_pos -= qty

    if mm_sz > 0:
        inv = adj_pos / limit
        inv_skew = int(round(inv * SKEW_T))
        bid_p = max(1, min(bb + 1 - inv_skew, ba - 1))
        ask_p = max(bb + 1, ba - 1 - inv_skew)
        if bid_p >= ask_p:
            bid_p = ask_p - 1
        bid_q = min(mm_sz, max(0, limit - adj_pos))
        ask_q = min(mm_sz, max(0, limit + adj_pos))
        if bid_q > 0:
            orders.append(Order(product, bid_p, bid_q))
        if ask_q > 0:
            orders.append(Order(product, ask_p, -ask_q))

    return orders


def _adaptive_core(product: str, od: OrderDepth, pos: int, td: dict,
                   frac: float = 0.8, thresh: float = 8.0,
                   mm_sz: int = DIR_MM) -> List[Order]:
    """
    Fully adaptive: direction decided by live EMA crossover per product.
    drift > +thresh  → long  at frac (scales to 100% as drift grows)
    drift < -thresh  → short at frac
    |drift| <= thresh → target=0, unwind to flat
    Used for swing/reversal products where historical direction is unreliable.
    """
    limit = _lim(product)
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    if ba <= bb:
        return []
    mid = (bb + ba) / 2.0

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    if drift > thresh:
        conf = min(1.0, 0.4 + (drift - thresh) / 20.0)
        target = int(limit * frac * conf)
    elif drift < -thresh:
        conf = min(1.0, 0.4 + (-drift - thresh) / 20.0)
        target = -int(limit * frac * conf)
    else:
        target = 0

    orders = []
    adj_pos = pos

    if pos < target:
        qty = min(target - pos, -sum(od.sell_orders.values()), limit - pos)
        if qty > 0:
            orders.append(Order(product, ba, qty))
            adj_pos += qty
    elif pos > target:
        qty = min(pos - target, sum(od.buy_orders.values()), limit + pos)
        if qty > 0:
            orders.append(Order(product, bb, -qty))
            adj_pos -= qty

    if mm_sz > 0:
        inv = adj_pos / limit
        inv_skew = int(round(inv * SKEW_T))
        bid_p = max(1, min(bb + 1 - inv_skew, ba - 1))
        ask_p = max(bb + 1, ba - 1 - inv_skew)
        if bid_p >= ask_p:
            bid_p = ask_p - 1
        bid_q = min(mm_sz, max(0, limit - adj_pos))
        ask_q = min(mm_sz, max(0, limit + adj_pos))
        if bid_q > 0:
            orders.append(Order(product, bid_p, bid_q))
        if ask_q > 0:
            orders.append(Order(product, ask_p, -ask_q))

    return orders


# ── Family strategies ─────────────────────────────────────────────────────────

def _snackpack(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """Pure MM — STRAWBERRY/RASPBERRY/CHOCOLATE/PISTACHIO confirmed profitable."""
    TILTS = {
        "SNACKPACK_STRAWBERRY": ( 1,  0, 1.2, 0.9),
        "SNACKPACK_RASPBERRY":  ( 0,  0, 1.0, 1.0),
        "SNACKPACK_CHOCOLATE":  ( 0,  1, 0.9, 1.1),
        "SNACKPACK_PISTACHIO":  (-1,  1, 0.8, 1.2),
    }
    bt, at, bm, am = TILTS.get(product, (0, 0, 1.0, 1.0))
    bv = sum(od.buy_orders.values()) if od.buy_orders else 0
    av = -sum(od.sell_orders.values()) if od.sell_orders else 0
    vbi = (bv - av) / max(1, bv + av)
    vbi_s = 1 if vbi > 0.15 else (-1 if vbi < -0.15 else 0)
    return _mm_core(product, od, pos, td,
                    bid_tilt=bt + vbi_s, ask_tilt=at + vbi_s,
                    bid_mult=bm, ask_mult=am, size=8)


def _pebbles(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """XL long / S short / XS short — confirmed +12k.  L and M stripped."""
    CFGS = {
        "PEBBLES_XL": ( 1, 1.0),
        "PEBBLES_S":  (-1, 0.8),
        "PEBBLES_XS": (-1, 1.0),
    }
    side, frac = CFGS[product]
    return _directional_core(product, od, pos, td, side, frac, DIR_MM)


def _microchip(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    OVAL/RECTANGLE/CIRCLE: family vote 2/3 — confirmed working.
    SQUARE and TRIANGLE stripped (adaptive failed on both).
    """
    MC_VOTE = ["MICROCHIP_OVAL", "MICROCHIP_RECTANGLE", "MICROCHIP_CIRCLE"]

    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow
    td[product + "_vote"] = (-1 if drift < -10 else (1 if drift > 10 else 0))

    votes = [td.get(p + "_vote", 0) for p in MC_VOTE]
    short_c = sum(1 for v in votes if v == -1)
    long_c  = sum(1 for v in votes if v ==  1)

    if short_c >= 2:
        return _directional_core(product, od, pos, td, -1, 0.8, DIR_MM)
    elif long_c >= 2:
        return _directional_core(product, od, pos, td,  1, 0.8, DIR_MM)
    else:
        return _mm_core(product, od, pos, td, size=7)


def _galaxy_sounds(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    Only SOLAR_WINDS active (+56, improved from -321).
    BLACK_HOLES/DARK_MATTER/PLANETARY_RINGS/SOLAR_FLAMES all stripped.
    """
    # Only SOLAR_WINDS reaches here (others in SKIP_PRODUCTS)
    return _adaptive_core(product, od, pos, td, frac=0.6, thresh=8.0)


def _oxygen_shake(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    GARLIC: directional long with tight drift gate (confirmed +3,727).
    CHOCOLATE: MM with EMA lean (confirmed +1,167).
    MINT: regime-filtered MM (confirmed +36, marginal but positive).
    MORNING_BREATH: adaptive own EMA.
    EVENING_BREATH stripped (net negative).
    """
    if not od.buy_orders or not od.sell_orders:
        return []
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    mid = (bb + ba) / 2.0
    limit = _lim(product)

    fast = _ema(td, product + "_f", mid, FAST_A)
    slow = _ema(td, product + "_s", mid, SLOW_A)
    drift = fast - slow

    if product == "OXYGEN_SHAKE_GARLIC":
        if drift >= 8:
            conf = min(1.0, 0.4 + (drift - 8) / 20.0)
            target = int(limit * conf)
        elif drift >= 3:
            target = int(limit * 0.4)
        else:
            target = 0
        orders = []
        adj_pos = pos
        if pos < target:
            qty = min(target - pos, -sum(od.sell_orders.values()), limit - pos)
            if qty > 0:
                orders.append(Order(product, ba, qty))
                adj_pos += qty
        elif pos > target:
            qty = min(pos - target, sum(od.buy_orders.values()), limit + pos)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                adj_pos -= qty
        inv = adj_pos / limit
        bid_p = max(1, min(bb + 1 - int(round(inv * SKEW_T)) + 1, ba - 1))
        ask_p = max(bb + 1, ba - 1 - int(round(inv * SKEW_T)))
        if bid_p >= ask_p:
            bid_p = ask_p - 1
        bid_q = min(DIR_MM, max(0, limit - adj_pos))
        ask_q = min(DIR_MM, max(0, limit + adj_pos))
        if bid_q > 0:
            orders.append(Order(product, bid_p, bid_q))
        if ask_q > 0:
            orders.append(Order(product, ask_p, -ask_q))
        return orders

    if product == "OXYGEN_SHAKE_CHOCOLATE":
        ema_dev = mid - _ema(td, product + "_mr", mid, 0.08)
        bt = int(round(max(-1, min(1, -ema_dev / 6.0))))
        return _mm_core(product, od, pos, td, bid_tilt=bt, ask_tilt=bt, size=7)

    if product == "OXYGEN_SHAKE_MINT":
        fast_r = _ema(td, product + "_fr", mid, FAST_A)
        slow_r = _ema(td, product + "_sr", mid, SLOW_A_REG)
        bm, am = _regime_mults(fast_r - slow_r, threshold=2.0, strong=5.0)
        return _mm_core(product, od, pos, td,
                        bid_mult=bm, ask_mult=am, size=7, skew_t=SKEW_T_H)

    if product == "OXYGEN_SHAKE_MORNING_BREATH":
        return _adaptive_core(product, od, pos, td, frac=0.6, thresh=8.0)

    return []


def _uv_visor(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """All confirmed profitable — unchanged from v5."""
    limit = _lim(product)

    if product == "UV_VISOR_AMBER":
        if not od.buy_orders or not od.sell_orders:
            return []
        bb = max(od.buy_orders)
        ba = min(od.sell_orders)
        mid = (bb + ba) / 2.0
        fast = _ema(td, product + "_f", mid, FAST_A)
        slow = _ema(td, product + "_s", mid, SLOW_A)
        drift = fast - slow
        target = -limit if drift <= 5 else -limit // 2
        orders = []
        adj_pos = pos
        if pos > target:
            qty = min(pos - target, sum(od.buy_orders.values()), limit + pos)
            if qty > 0:
                orders.append(Order(product, bb, -qty))
                adj_pos -= qty
        inv = adj_pos / limit
        ask_p = max(bb + 1, ba - 1 - int(round(inv * SKEW_T)))
        ask_q = min(MM_SIZE, max(0, limit + adj_pos))
        if ask_q > 0:
            orders.append(Order(product, ask_p, -ask_q))
        return orders

    CFGS = {
        "UV_VISOR_MAGENTA": ( 1, 0, 1.2, 0.8),
        "UV_VISOR_RED":     ( 1, 0, 1.2, 0.8),
        "UV_VISOR_ORANGE":  ( 0, 0, 1.0, 1.0),
        "UV_VISOR_YELLOW":  ( 0, 0, 1.0, 1.0),
    }
    bt, at, bm, am = CFGS.get(product, (0, 0, 1.0, 1.0))
    return _mm_core(product, od, pos, td,
                    bid_tilt=bt, ask_tilt=at, bid_mult=bm, ask_mult=am)


def _sleep_pod(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    COTTON/SUEDE: directional long (confirmed profitable).
    LAMB_WOOL: COTTON pairs signal (r=-0.888).
               COTTON rising → WOOL falling → short WOOL.
               COTTON falling → WOOL rising → long WOOL.
    NYLON/POLYESTER stripped (net negative).
    """
    if product == "SLEEP_POD_LAMB_WOOL":
        cotton_drift = _drift(td, "SLEEP_POD_COTTON")
        if cotton_drift > 6:
            return _directional_core(product, od, pos, td, -1, 0.8, DIR_MM)
        elif cotton_drift < -6:
            return _directional_core(product, od, pos, td,  1, 0.6, DIR_MM)
        else:
            return _mm_core(product, od, pos, td, size=4, skew_t=SKEW_T_H)

    CFGS = {
        "SLEEP_POD_SUEDE":  1.0,
        "SLEEP_POD_COTTON": 0.9,
    }
    frac = CFGS[product]
    return _directional_core(product, od, pos, td, 1, frac, DIR_MM)


def _panel(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """Confirmed profitable — unchanged."""
    CFGS = {
        "PANEL_2X4": ( 0, 0.0),
        "PANEL_4X4": ( 0, 0.0),
        "PANEL_1X4": (-1, 0.7),
        "PANEL_2X2": (-1, 0.5),
        "PANEL_1X2": (-1, 0.3),
    }
    side, frac = CFGS[product]
    if side == 0:
        return _mm_core(product, od, pos, td, size=7)
    return _directional_core(product, od, pos, td, side, frac, DIR_MM)


def _translator(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    SPACE_GRAY short / VOID_BLUE long: confirmed +11k, unchanged.
    ASTRO_BLACK: SPACE_GRAY pairs signal (r=-0.832).
                 GRAY falling (drift < 0) → BLACK rising → LONG.
                 Previously shorted (0.4 frac) — exactly wrong direction.
    CHARCOAL/MIST: passive MM (confirmed +1,710/+3,000).
    """
    if product == "TRANSLATOR_ASTRO_BLACK":
        gray_drift = _drift(td, "TRANSLATOR_SPACE_GRAY")
        # GRAY and BLACK anti-correlated: GRAY down → BLACK up
        if gray_drift < -5:
            return _directional_core(product, od, pos, td, 1, 0.7, DIR_MM)
        elif gray_drift > 5:
            return _directional_core(product, od, pos, td, -1, 0.5, DIR_MM)
        else:
            return _mm_core(product, od, pos, td, size=5)

    CFGS = {
        "TRANSLATOR_VOID_BLUE":        ( 1, 1.0),
        "TRANSLATOR_SPACE_GRAY":       (-1, 1.0),
        "TRANSLATOR_ECLIPSE_CHARCOAL": ( 0, 0.0),
        "TRANSLATOR_GRAPHITE_MIST":    ( 0, 0.0),
    }
    side, frac = CFGS[product]
    if side == 0:
        return _mm_core(product, od, pos, td, size=7)
    return _directional_core(product, od, pos, td, side, frac, DIR_MM)


def _robot(product: str, od: OrderDepth, pos: int, td: dict) -> List[Order]:
    """
    LAUNDRY: explicit directional short (dp=-541, r=-0.949 with IRONING).
    IRONING: LAUNDRY pairs signal — LAUNDRY falling → IRONING rising → LONG.
    MOPPING: adaptive own EMA.
    VACUUMING/DISHES stripped (net negative).
    """
    if product == "ROBOT_LAUNDRY":
        return _directional_core(product, od, pos, td, -1, 1.0, DIR_MM)

    if product == "ROBOT_IRONING":
        # IRONING is the anti-twin of LAUNDRY (r=-0.949)
        laundry_drift = _drift(td, "ROBOT_LAUNDRY")
        if laundry_drift < -5:
            return _directional_core(product, od, pos, td, 1, 0.8, DIR_MM)
        elif laundry_drift > 5:
            return _directional_core(product, od, pos, td, -1, 0.5, DIR_MM)
        else:
            return _mm_core(product, od, pos, td, size=4, skew_t=SKEW_T_H)

    if product == "ROBOT_MOPPING":
        return _adaptive_core(product, od, pos, td, frac=0.7, thresh=6.0)

    return []


# ── Family dispatch ───────────────────────────────────────────────────────────

FAMILY_MAP = {
    "SNACKPACK":     _snackpack,
    "PEBBLES":       _pebbles,
    "MICROCHIP":     _microchip,
    "GALAXY_SOUNDS": _galaxy_sounds,
    "OXYGEN_SHAKE":  _oxygen_shake,
    "UV_VISOR":      _uv_visor,
    "SLEEP_POD":     _sleep_pod,
    "PANEL":         _panel,
    "TRANSLATOR":    _translator,
    "ROBOT":         _robot,
}

def _detect_family(product: str) -> str:
    for prefix in ("GALAXY_SOUNDS", "OXYGEN_SHAKE", "SLEEP_POD", "UV_VISOR"):
        if product.startswith(prefix):
            return prefix
    for prefix in ("SNACKPACK", "PEBBLES", "MICROCHIP", "PANEL", "TRANSLATOR", "ROBOT"):
        if product.startswith(prefix):
            return prefix
    return None


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        for product, od in state.order_depths.items():
            if product in SKIP_PRODUCTS:
                continue
            family = _detect_family(product)
            if family is None:
                continue
            handler = FAMILY_MAP.get(family)
            if handler is None:
                continue
            pos = state.position.get(product, 0)
            orders = handler(product, od, pos, td)
            if orders:
                result[product] = orders

        return result, 0, json.dumps(td)
