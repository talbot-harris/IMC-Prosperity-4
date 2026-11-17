"""
================================================================
  IMC PROSPERITY ROUND 4 — MONTE CARLO BACKTESTER
  Products:
    HYDROGEL_PACK            (mean-reverting near 10008)
    VELVETFRUIT_EXTRACT      (underlying, drifting random walk)
    VEV_{4000..6500}         (European calls on VFE; 10 strikes)
================================================================

USAGE:
  python mc_backtester_round4.py <your_algorithm.py>

CONFIG (bottom of file):
  N_SIMS         number of Monte Carlo runs       (default 200)
  N_TICKS        ticks per simulated day          (default 1000)
  TTE_START      time to expiry at day start, days (default 2.0)

CALIBRATION:
  All product parameters are calibrated from the day-3 submission
  log (500649_log/500649.json). Per-tick stdev, mean spread, mean
  L1 volume, and observed price levels are taken directly from the
  1000 ticks of that day. The daily *drift* is sampled fresh per
  simulation (centered at zero, sigma chosen so the realized day-3
  drift is ~1 sigma) so each MC run sees a different trend regime.

OPTION PRICING:
  Per tick, each VEV strike's *true* mid is BS(S, K, T, sigma)
  where S = current VFE mid, T decreases linearly through the day,
  and sigma is a per-strike IV with a small smile. A persistent
  Ornstein-Uhlenbeck noise term is added to model market-maker
  pricing imperfection — this is the edge a competent algo earns
  by quoting tight around the true BS value.

OUTPUT:
  - mc_backtest_round4.png  (PnL distribution + per-product + traces)
  - Console statistics
================================================================
"""

import sys, os, types, importlib.util, math
import numpy as np
from collections import defaultdict


# ==============================================================
#  DATAMODEL STUBS
# ==============================================================

class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol   = symbol
        self.price    = int(price)
        self.quantity = int(quantity)

class OrderDepth:
    def __init__(self):
        self.buy_orders  = {}
        self.sell_orders = {}

class TradingState:
    def __init__(self):
        self.traderData    = ""
        self.timestamp     = 0
        self.listings      = {}
        self.order_depths  = {}
        self.own_trades    = {}
        self.market_trades = {}
        self.position      = {}
        self.observations  = None

class Listing:
    def __init__(self, symbol, product, denomination):
        self.symbol       = symbol
        self.product      = product
        self.denomination = denomination

class Trade:
    def __init__(self, symbol, price, quantity, buyer=None, seller=None, timestamp=0):
        self.symbol    = symbol
        self.price     = price
        self.quantity  = quantity
        self.buyer     = buyer
        self.seller    = seller
        self.timestamp = timestamp

UserId      = str
Symbol      = str
Product     = str
Position    = int
Observation = object


def _build_datamodel_module():
    mod = types.ModuleType("datamodel")
    for name in ("Order", "OrderDepth", "TradingState", "Listing", "Trade",
                 "UserId", "Symbol", "Product", "Position", "Observation"):
        setattr(mod, name, globals()[name])
    return mod


def load_trader(algo_path: str):
    if not os.path.isfile(algo_path):
        print(f"  ERROR: algorithm file not found: {algo_path}")
        sys.exit(1)
    sys.modules.setdefault("datamodel", _build_datamodel_module())
    spec   = importlib.util.spec_from_file_location("_trader_algo", algo_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"  ERROR loading {algo_path}: {exc}")
        sys.exit(1)
    if not hasattr(module, "Trader"):
        print(f"  ERROR: no Trader class in {algo_path}")
        sys.exit(1)
    return module.Trader


# ==============================================================
#  BLACK-SCHOLES (same closed-form approx as the algo)
# ==============================================================

def _ncdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    poly = t * (0.319381530
          + t * (-0.356563782
          + t * (1.781477937
          + t * (-1.821255978
          + t *   1.330274429))))
    p = 1.0 - d * poly
    return p if x >= 0 else 1.0 - p


def _bs_call(S, K, T, sigma):
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)


# ==============================================================
#  PRODUCT CALIBRATION (from 500649 day-3 log)
# ==============================================================

VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Underlying / non-option params. l1_vol/l2_vol are (lo, hi) for rng.integers.
NON_OPTION_PARAMS = {
    "HYDROGEL_PACK": {
        "type":          "mean_revert",
        "fair":          10_008.0,
        "mr_strength":   0.05,        # pulls back to fair each tick
        "drift_std":     2.17,        # per-tick noise (empirical)
        "half_spreads":  [7, 8, 9],
        "half_spread_ps":[0.40, 0.40, 0.20],
        "l1_vol":        (8, 18),
        "l2_vol":        (15, 35),
        "l2_offset":     (2, 4),
        "fill_prob":     0.030,
        "fill_qty":      (2, 8),
        "limit":         200,
    },
    "VELVETFRUIT_EXTRACT": {
        "type":          "drifting",
        "fair":          5_295.0,     # day-3 open
        "drift_day_std": 0.05,        # std of *daily* drift (per tick)
        "drift_std":     1.13,        # per-tick noise (empirical)
        "half_spreads":  [2, 3],
        "half_spread_ps":[0.55, 0.45],
        "l1_vol":        (15, 60),
        "l2_vol":        (20, 50),
        "l2_offset":     (1, 2),
        "fill_prob":     0.040,
        "fill_qty":      (3, 12),
        "limit":         200,
    },
}

# Per-strike option market params. Spreads/volumes empirical from day-3.
# IV smile: low IV across the board, slight upward skew for OTM puts (lower
# strikes) consistent with calls being underpriced at deep ITM.
VEV_PARAMS = {
    4000: {"half_spread_mean": 10, "half_spread_std": 2, "l1_vol": (5, 15),  "iv": 0.0175, "limit": 300},
    4500: {"half_spread_mean": 8,  "half_spread_std": 2, "l1_vol": (5, 14),  "iv": 0.0172, "limit": 300},
    5000: {"half_spread_mean": 3,  "half_spread_std": 1, "l1_vol": (8, 22),  "iv": 0.0170, "limit": 300},
    5100: {"half_spread_mean": 2,  "half_spread_std": 1, "l1_vol": (10, 28), "iv": 0.0170, "limit": 300},
    5200: {"half_spread_mean": 2,  "half_spread_std": 1, "l1_vol": (12, 32), "iv": 0.0168, "limit": 300},
    5300: {"half_spread_mean": 1,  "half_spread_std": 1, "l1_vol": (12, 30), "iv": 0.0168, "limit": 300},
    5400: {"half_spread_mean": 1,  "half_spread_std": 1, "l1_vol": (12, 30), "iv": 0.0168, "limit": 300},
    5500: {"half_spread_mean": 1,  "half_spread_std": 0, "l1_vol": (12, 30), "iv": 0.0168, "limit": 300},
    6000: {"half_spread_mean": 1,  "half_spread_std": 0, "l1_vol": (12, 30), "iv": 0.0168, "limit": 300},
    6500: {"half_spread_mean": 1,  "half_spread_std": 0, "l1_vol": (8, 22),  "iv": 0.0168, "limit": 300},
}

# Persistent OU noise on each option mid (per-strike), modeling MM pricing
# imperfection that the algo can pick off:
VEV_NOISE_THETA = 0.05    # mean-reversion of noise
VEV_NOISE_SIGMA = 0.40    # per-tick shock std

PRODUCTS = list(NON_OPTION_PARAMS.keys()) + [f"VEV_{k}" for k in VEV_STRIKES]

POSITION_LIMITS = {p: NON_OPTION_PARAMS[p]["limit"] for p in NON_OPTION_PARAMS}
POSITION_LIMITS.update({f"VEV_{k}": VEV_PARAMS[k]["limit"] for k in VEV_STRIKES})


# ==============================================================
#  SYNTHETIC MARKET DATA GENERATOR
# ==============================================================

def _make_book_around(rng, mid_int, half_spread, l1_vol_range, l2_vol_range=None,
                      l2_offset_range=None):
    """Generic book builder around an integer mid with a given half-spread."""
    depth = OrderDepth()
    bb    = mid_int - half_spread
    ba    = mid_int + half_spread

    l1v = int(rng.integers(*l1_vol_range))
    depth.buy_orders[bb]  =  l1v
    depth.sell_orders[ba] = -l1v

    if l2_vol_range is not None and l2_offset_range is not None:
        l2v  = int(rng.integers(*l2_vol_range))
        l2o  = int(rng.integers(*l2_offset_range))
        depth.buy_orders[bb - l2o]  =  l2v
        depth.sell_orders[ba + l2o] = -l2v

    return depth


def _gen_underlying_paths(rng, n_ticks):
    """Generate VFE and HP price paths for one day."""
    # VFE: random walk with sampled daily drift
    vfe_p   = NON_OPTION_PARAMS["VELVETFRUIT_EXTRACT"]
    vfe_drift = rng.normal(0.0, vfe_p["drift_day_std"])
    vfe_noise = rng.normal(0.0, vfe_p["drift_std"], size=n_ticks)
    vfe = vfe_p["fair"] + np.cumsum(vfe_drift + vfe_noise)

    # HP: mean-reverting around fair
    hp_p   = NON_OPTION_PARAMS["HYDROGEL_PACK"]
    hp     = np.empty(n_ticks)
    hp[0]  = hp_p["fair"]
    noise  = rng.normal(0.0, hp_p["drift_std"], size=n_ticks)
    for t in range(1, n_ticks):
        hp[t] = hp[t-1] + hp_p["mr_strength"] * (hp_p["fair"] - hp[t-1]) + noise[t]

    return vfe, hp


def generate_day(rng, n_ticks, tte_start, ticks_per_day):
    """
    Generate one full simulated day.
    Returns: (ticks: {ts: {prod: OrderDepth}}, mids: {ts: {prod: float}})
    """
    vfe, hp = _gen_underlying_paths(rng, n_ticks)

    # Persistent OU noise per option strike
    vev_noise = {k: 0.0 for k in VEV_STRIKES}

    ticks = {}
    mids  = {}

    for i in range(n_ticks):
        ts          = i * 100
        elapsed_day = i / n_ticks
        T           = max(1e-4, tte_start - elapsed_day)

        ticks_dict = {}
        mids_dict  = {}

        # ── HYDROGEL_PACK ──
        hp_p   = NON_OPTION_PARAMS["HYDROGEL_PACK"]
        hp_mid = int(round(hp[i]))
        hs     = int(rng.choice(hp_p["half_spreads"], p=hp_p["half_spread_ps"]))
        depth  = _make_book_around(rng, hp_mid, hs,
                                   hp_p["l1_vol"], hp_p["l2_vol"], hp_p["l2_offset"])
        ticks_dict["HYDROGEL_PACK"] = depth
        mids_dict ["HYDROGEL_PACK"] = float(hp_mid)

        # ── VELVETFRUIT_EXTRACT ──
        vfe_p   = NON_OPTION_PARAMS["VELVETFRUIT_EXTRACT"]
        vfe_mid = int(round(vfe[i]))
        hs      = int(rng.choice(vfe_p["half_spreads"], p=vfe_p["half_spread_ps"]))
        depth   = _make_book_around(rng, vfe_mid, hs,
                                    vfe_p["l1_vol"], vfe_p["l2_vol"], vfe_p["l2_offset"])
        ticks_dict["VELVETFRUIT_EXTRACT"] = depth
        mids_dict ["VELVETFRUIT_EXTRACT"] = float(vfe_mid)

        # ── VEV options ──
        for K in VEV_STRIKES:
            p = VEV_PARAMS[K]
            # OU noise update
            vev_noise[K] += -VEV_NOISE_THETA * vev_noise[K] + \
                            rng.normal(0.0, VEV_NOISE_SIGMA)
            true_bs = _bs_call(float(vfe[i]), float(K), T, p["iv"])
            opt_mid = max(0.0, true_bs + vev_noise[K])
            opt_mid_int = int(round(opt_mid))

            hs = max(1, int(round(rng.normal(p["half_spread_mean"],
                                             p["half_spread_std"]))))
            # Floor handling: deep OTM with mid near 0 — keep bb >= 0
            bb = max(0, opt_mid_int - hs)
            ba = bb + 2 * hs if bb == 0 else opt_mid_int + hs
            if ba <= bb:
                ba = bb + 1

            depth = OrderDepth()
            l1v   = int(rng.integers(*p["l1_vol"]))
            depth.buy_orders[bb]  =  l1v
            depth.sell_orders[ba] = -l1v

            sym = f"VEV_{K}"
            ticks_dict[sym] = depth
            mids_dict [sym] = (bb + ba) / 2.0

        ticks[ts] = ticks_dict
        mids [ts] = mids_dict

    return ticks, mids


# ==============================================================
#  MATCHING ENGINE
# ==============================================================

def match_aggressive(orders, book_depth, position, limit):
    fills, passive = [], []
    for order in orders:
        qty = order.quantity
        if qty > 0:
            for ask_price in sorted(book_depth.sell_orders.keys()):
                if ask_price <= order.price and qty > 0:
                    avail   = abs(book_depth.sell_orders[ask_price])
                    can_buy = min(qty, avail, limit - position)
                    if can_buy > 0:
                        fills.append((order.symbol, ask_price, can_buy))
                        position += can_buy
                        qty      -= can_buy
            if qty > 0:
                passive.append(Order(order.symbol, order.price, qty))
        elif qty < 0:
            sq = abs(qty)
            for bid_price in sorted(book_depth.buy_orders.keys(), reverse=True):
                if bid_price >= order.price and sq > 0:
                    avail    = book_depth.buy_orders[bid_price]
                    can_sell = min(sq, avail, limit + position)
                    if can_sell > 0:
                        fills.append((order.symbol, bid_price, -can_sell))
                        position -= can_sell
                        sq       -= can_sell
            if sq > 0:
                passive.append(Order(order.symbol, order.price, -sq))
    return fills, position, passive


def match_passive_stochastic(passive_orders, position, product, rng):
    """Stochastic passive fill, using product's fill_prob/fill_qty if defined."""
    fills = []
    if product in NON_OPTION_PARAMS:
        p = NON_OPTION_PARAMS[product]
        prob, fq = p.get("fill_prob", 0.02), p.get("fill_qty", (2, 7))
        limit    = p["limit"]
    else:
        # Generic option fill assumption: lower fill rate than underlying.
        prob, fq = 0.020, (2, 8)
        limit    = POSITION_LIMITS.get(product, 300)

    for order in passive_orders:
        if rng.random() >= prob:
            continue
        fill_qty = int(rng.integers(fq[0], fq[1] + 1))
        if order.quantity > 0:
            can_buy = min(order.quantity, fill_qty, limit - position)
            if can_buy > 0:
                fills.append((order.symbol, order.price, can_buy))
                position += can_buy
        elif order.quantity < 0:
            can_sell = min(abs(order.quantity), fill_qty, limit + position)
            if can_sell > 0:
                fills.append((order.symbol, order.price, -can_sell))
                position -= can_sell
    return fills, position


# ==============================================================
#  SINGLE SIMULATION
# ==============================================================

def simulate_one(TraderClass, rng, n_ticks, tte_start, ticks_per_day,
                 return_trace=False):
    ticks, mids = generate_day(rng, n_ticks, tte_start, ticks_per_day)

    trader      = TraderClass()
    trader_data = ""
    positions   = defaultdict(int)
    cash        = defaultdict(float)
    timestamps  = sorted(ticks.keys())
    trace       = [] if return_trace else None

    for ts in timestamps:
        state                = TradingState()
        state.timestamp      = ts
        state.traderData     = trader_data
        state.order_depths   = ticks[ts]
        state.position       = dict(positions)
        state.own_trades     = {}
        state.market_trades  = {}

        try:
            result, _conv, trader_data = trader.run(state)
        except Exception as exc:
            print(f"    [warn] trader.run raised at ts={ts}: {exc}")
            result, trader_data = {}, ""

        for product, orders in result.items():
            limit = POSITION_LIMITS.get(product, 300)
            book  = ticks[ts].get(product)
            if book is None:
                continue

            agg_fills, _, passive = match_aggressive(
                orders, book, positions[product], limit
            )
            for _, fill_price, fill_qty in agg_fills:
                cash[product]      -= fill_price * fill_qty
                positions[product] += fill_qty

            if passive:
                pas_fills, _ = match_passive_stochastic(
                    passive, positions[product], product, rng
                )
                for _, fill_price, fill_qty in pas_fills:
                    cash[product]      -= fill_price * fill_qty
                    positions[product] += fill_qty

        if return_trace:
            total_pnl = 0.0
            ppnl     = {}
            for prod in PRODUCTS:
                mid       = mids[ts].get(prod, 0.0)
                pnl       = cash[prod] + positions[prod] * mid
                ppnl[prod] = pnl
                total_pnl += pnl
            trace.append({
                "ts":          ts,
                "total_pnl":   total_pnl,
                "product_pnl": dict(ppnl),
                "positions":   dict(positions),
            })

    last_ts = timestamps[-1]
    total, ppnl = 0.0, {}
    for prod in PRODUCTS:
        mid       = mids[last_ts].get(prod, 0.0)
        pnl       = cash[prod] + positions[prod] * mid
        ppnl[prod] = pnl
        total    += pnl

    return total, ppnl, dict(positions), trace


# ==============================================================
#  MONTE CARLO RUNNER
# ==============================================================

def run_monte_carlo(TraderClass, n_sims, n_ticks, tte_start, ticks_per_day, seed):
    rng          = np.random.default_rng(seed)
    results      = []
    pnls_by_prod = {p: [] for p in PRODUCTS}
    max_abs_pos  = {p: [] for p in PRODUCTS}   # peak |position| per sim
    sample_trace = None

    for i in range(n_sims):
        child_rng    = np.random.default_rng(rng.integers(0, 2**31))
        return_trace = True   # always trace so we can measure peak |pos|

        total, ppnl, _pos, trace = simulate_one(
            TraderClass, child_rng, n_ticks, tte_start, ticks_per_day,
            return_trace=return_trace,
        )

        results.append(total)
        for prod in PRODUCTS:
            pnls_by_prod[prod].append(ppnl.get(prod, 0))
            peak = max((abs(r["positions"].get(prod, 0)) for r in trace), default=0)
            max_abs_pos[prod].append(peak)

        if i == 0:
            sample_trace = trace

        if (i + 1) % 25 == 0:
            print(f"    Completed {i + 1}/{n_sims} simulations...")

    return (
        np.array(results),
        {p: np.array(v) for p, v in pnls_by_prod.items()},
        {p: np.array(v) for p, v in max_abs_pos.items()},
        sample_trace,
    )


# ==============================================================
#  VISUALIZATION
# ==============================================================

def plot_results(results, pnls_by_prod, sample_trace, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monte Carlo Backtester — IMC Prosperity Round 4",
                 fontsize=14, fontweight="bold")

    # Panel 1: Total PnL distribution
    ax = axes[0, 0]
    ax.hist(results, bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(results), color="red", linewidth=2,
               label=f"Mean: {np.mean(results):.0f}")
    ax.axvline(np.median(results), color="orange", linewidth=2, linestyle="--",
               label=f"Median: {np.median(results):.0f}")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Total PnL Distribution (n={len(results)})")
    ax.set_xlabel("PnL (SeaShells)")
    ax.set_ylabel("Count")
    ax.legend()

    # Panel 2: Per-product PnL means (bar chart — too many to histogram)
    ax = axes[0, 1]
    prod_names = list(pnls_by_prod.keys())
    means      = [np.mean(pnls_by_prod[p]) for p in prod_names]
    stds       = [np.std(pnls_by_prod[p])  for p in prod_names]
    short      = [p.replace("VELVETFRUIT_EXTRACT", "VFE")
                   .replace("HYDROGEL_PACK", "HP")
                   .replace("VEV_", "V") for p in prod_names]
    colors     = ["#2196F3", "#FF5722"] + ["#4CAF50"] * len(VEV_STRIKES)
    x = np.arange(len(prod_names))
    ax.bar(x, means, yerr=stds, color=colors[:len(prod_names)],
           alpha=0.75, capsize=3, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_title("Per-Product PnL (mean ± 1σ)")
    ax.set_ylabel("PnL")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Sample run total PnL trace (with HP / VFE / VEV-aggregate split)
    ax = axes[1, 0]
    if sample_trace:
        ts_arr = [r["ts"] for r in sample_trace]
        total  = [r["total_pnl"] for r in sample_trace]
        hp_p   = [r["product_pnl"].get("HYDROGEL_PACK", 0) for r in sample_trace]
        vfe_p  = [r["product_pnl"].get("VELVETFRUIT_EXTRACT", 0) for r in sample_trace]
        vev_p  = [sum(r["product_pnl"].get(f"VEV_{k}", 0) for k in VEV_STRIKES)
                  for r in sample_trace]
        ax.plot(ts_arr, total, "b-",  linewidth=1.5, label="Total")
        ax.plot(ts_arr, hp_p,  "-",   linewidth=1, alpha=0.7, label="HP")
        ax.plot(ts_arr, vfe_p, "-",   linewidth=1, alpha=0.7, label="VFE")
        ax.plot(ts_arr, vev_p, "-",   linewidth=1, alpha=0.7, label="VEV (sum)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Sample Run — PnL Trace")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("PnL")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 4: Sample positions (HP, VFE, sum of VEV long delta if available)
    ax = axes[1, 1]
    if sample_trace:
        hp_pos  = [r["positions"].get("HYDROGEL_PACK", 0) for r in sample_trace]
        vfe_pos = [r["positions"].get("VELVETFRUIT_EXTRACT", 0) for r in sample_trace]
        vev_sum = [sum(r["positions"].get(f"VEV_{k}", 0) for k in VEV_STRIKES)
                   for r in sample_trace]
        ax.plot(ts_arr, hp_pos,  "-", linewidth=1, label="HP")
        ax.plot(ts_arr, vfe_pos, "-", linewidth=1, label="VFE")
        ax.plot(ts_arr, vev_sum, "-", linewidth=1, label="Σ VEV positions")
        ax.axhline(0,    color="black", linewidth=0.5)
        ax.axhline( 200, color="gray",  linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(-200, color="gray",  linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title("Sample Run — Positions")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Net contracts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"\n  Chart saved: {outpath}")
    plt.close()


# ==============================================================
#  MAIN
# ==============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python mc_backtester_round4.py <algorithm_file.py>")
        sys.exit(1)

    algo_path = sys.argv[1]

    # ── Configuration ──
    N_SIMS         = 200
    N_TICKS        = 1000      # Round 4: 1000 ticks/day, ts step 100
    TTE_START      = 2.0       # day-3 equivalent
    TICKS_PER_DAY  = 100_000   # tick-units per day (matches algo constant)
    SEED           = 42
    OUTPATH        = "mc_backtest_round4.png"

    print("=" * 60)
    print("  IMC PROSPERITY ROUND 4 — MONTE CARLO BACKTESTER")
    print("=" * 60)
    print(f"\n  Algorithm:   {algo_path}")
    print(f"  Simulations: {N_SIMS}")
    print(f"  Ticks/day:   {N_TICKS}")
    print(f"  TTE start:   {TTE_START} days")
    print(f"  Products:    {len(PRODUCTS)} "
          f"(HP, VFE, {len(VEV_STRIKES)} VEV strikes)")

    print(f"\n  Loading algorithm...")
    TraderClass = load_trader(algo_path)
    print(f"  Trader class loaded OK.")

    print(f"\n  Running simulations...")
    results, pnls_by_prod, max_abs_pos, sample_trace = run_monte_carlo(
        TraderClass, N_SIMS, N_TICKS, TTE_START, TICKS_PER_DAY, SEED
    )

    # ── Statistics ──
    print(f"\n  {'=' * 50}")
    print(f"  RESULTS ({N_SIMS} simulations)")
    print(f"  {'=' * 50}")
    print(f"  Total PnL:")
    print(f"    Mean:   {np.mean(results):>10.1f}")
    print(f"    Median: {np.median(results):>10.1f}")
    print(f"    Std:    {np.std(results):>10.1f}")
    print(f"    Min:    {np.min(results):>10.1f}")
    print(f"    Max:    {np.max(results):>10.1f}")
    sharpe   = np.mean(results) / np.std(results) if np.std(results) > 0 else 0
    win_rate = np.mean(results > 0) * 100
    print(f"    Sharpe: {sharpe:>10.2f}")
    print(f"    Win %:  {win_rate:>9.1f}%")

    print(f"\n  Per-product PnL (mean ± std)  |  peak |pos| mean / max  vs limit:")
    for prod in PRODUCTS:
        arr = pnls_by_prod[prod]
        if np.std(arr) < 1e-6 and abs(np.mean(arr)) < 1e-6:
            continue   # untraded products — skip noise
        short = (prod.replace("VELVETFRUIT_EXTRACT", "VFE")
                     .replace("HYDROGEL_PACK", "HP"))
        peaks = max_abs_pos[prod]
        limit = POSITION_LIMITS[prod]
        util  = np.mean(peaks) / limit * 100 if limit else 0
        print(f"    {short:14s} {np.mean(arr):>9.1f} ± {np.std(arr):>7.1f}"
              f"  |  peak {np.mean(peaks):>5.0f} / {np.max(peaks):>3.0f}"
              f"  vs limit {limit:>3d}  ({util:>4.0f}% mean util)")

    pcts = [5, 25, 50, 75, 95]
    print(f"\n  Percentiles (total PnL):")
    for p in pcts:
        print(f"    {p:3d}th: {np.percentile(results, p):>10.1f}")

    plot_results(results, pnls_by_prod, sample_trace, OUTPATH)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
