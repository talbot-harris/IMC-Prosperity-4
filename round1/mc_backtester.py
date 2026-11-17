"""
================================================================
  IMC PROSPERITY - MONTE CARLO BACKTESTER
  Products: ASH_COATED_OSMIUM  |  INTARIAN_PEPPER_ROOT
================================================================

HOW TO USE:
  python mc_backtester.py <your_algorithm.py>

  The algorithm file must contain a Trader class with a run()
  method matching the standard IMC datamodel signature. You do
  NOT need to modify this file - just point it at your algo.

CONFIGURATION (bottom of this file, in main()):
  N_SIMS   = number of Monte Carlo runs (default 200)
  N_TICKS  = ticks per simulated day    (default 2000)

OUTPUT:
  - mc_backtest_output.png  (PnL distribution + traces)
  - Console statistics (mean, std, Sharpe, win rate, percentiles)

MARKET MODEL (adjust PRODUCT_PARAMS below to recalibrate):
  ASH_COATED_OSMIUM    - stationary fair value, tight spread
  INTARIAN_PEPPER_ROOT - drifting random walk, wider spread
================================================================
"""

import sys, os, types, importlib.util, json, math
import numpy as np
from collections import defaultdict

# ==============================================================
#  DATAMODEL STUBS
#  These mirror the real IMC datamodel so your algorithm's
#  imports work without any modification.
# ==============================================================

class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol   = symbol
        self.price    = int(price)
        self.quantity = int(quantity)

class OrderDepth:
    def __init__(self):
        self.buy_orders  = {}   # {price: +qty}
        self.sell_orders = {}   # {price: -qty}

class TradingState:
    def __init__(self):
        self.traderData   = ""
        self.timestamp    = 0
        self.listings     = {}
        self.order_depths = {}
        self.own_trades   = {}
        self.market_trades = {}
        self.position     = {}
        self.observations = None

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

UserId = str
Symbol = str
Product = str
Position = int
Observation = object


def _build_datamodel_module():
    """Return a fake 'datamodel' module exposing all stub classes."""
    mod = types.ModuleType("datamodel")
    mod.Order        = Order
    mod.OrderDepth   = OrderDepth
    mod.TradingState = TradingState
    mod.Listing      = Listing
    mod.Trade        = Trade
    mod.UserId       = UserId
    mod.Symbol       = Symbol
    mod.Product      = Product
    mod.Position     = Position
    mod.Observation  = Observation
    return mod


def load_trader(algo_path: str):
    """
    Dynamically load Trader from an external algorithm file.
    Injects a fake 'datamodel' module first so IMC-style
    'from datamodel import ...' lines work without modification.
    """
    if not os.path.isfile(algo_path):
        print(f"  ERROR: algorithm file not found: {algo_path}")
        sys.exit(1)

    # Register fake datamodel before importing the algo
    sys.modules.setdefault("datamodel", _build_datamodel_module())

    spec   = importlib.util.spec_from_file_location("_trader_algo", algo_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"  ERROR loading {algo_path}: {exc}")
        sys.exit(1)

    if not hasattr(module, "Trader"):
        print(f"  ERROR: no Trader class found in {algo_path}")
        sys.exit(1)

    return module.Trader


# ==============================================================
#  PRODUCT CONFIGURATION
#  Adjust these parameters to match observed round data.
# ==============================================================

PRODUCT_OSMIUM = "ASH_COATED_OSMIUM"
PRODUCT_PEPPER = "INTARIAN_PEPPER_ROOT"

PRODUCT_PARAMS = {
    PRODUCT_OSMIUM: {
        # Calibrated from prices_round_1_day_-2/-1/0.csv (27644 valid ticks)
        # Stationary: near 10000, mean-reversion keeps daily drift to ~+-10
        "type":            "stationary",
        "fair":            10_000,
        "drift_mean":      0.0,
        "drift_std":       1.93,       # intra-day return std (empirical)
        # Half-spread distribution: 8 (64%), 9 (21%), 10 (7%) - from half-spread counter
        "half_spreads":    [8, 9, 10],
        "half_spread_ps":  [0.72, 0.21, 0.07],
        "l1_vol":          (10, 28),   # p5=10, p95=27
        "l2_vol":          (15, 30),   # p5=15, p95=30
        "l2_offset":       (2, 4),     # rng.integers -> 2 or 3 (observed ~equal split)
        "tighten_prob":    0.053,      # 5.3% of valid ticks have 3 levels on one side
        "tighten_vol":     (2, 10),    # inner bot level vol: mean=4.7, range=1-10
        "tighten_hs":      (1, 5),     # inner level half-spread (1-4 ticks from fair)
        "fill_prob":       0.046,      # market trades / valid ticks across all 3 days
        "fill_qty":        (2, 11),    # trade qty: mean=5.2, range=2-10
        "limit":           80,
    },
    PRODUCT_PEPPER: {
        # Calibrated from prices_round_1_day_-2/-1/0.csv (27688 valid ticks)
        # Strong upward trend: ~1000/day, 0.1085/tick drift
        "type":            "drifting",
        "fair":            10_000,     # day -2 opens near 10000
        "drift_mean":      0.1085,     # per-tick drift (empirical intra-day mean)
        "drift_std":       1.7473,     # per-tick return std (empirical)
        # Half-spread distribution: 6 (39%), 7 (32%), 8 (14%), 9 (4%)
        "half_spreads":    [6, 7, 8, 9],
        "half_spread_ps":  [0.46, 0.36, 0.14, 0.04],
        "l1_vol":          (8, 23),    # p5=8, p95=22
        "l2_vol":          (15, 26),   # p5=15, p95=25
        "l2_offset":       (3, 5),     # rng.integers -> 3 or 4 (3 is dominant)
        "tighten_prob":    0.032,      # 3.15% of valid ticks have 3 levels on one side
        "tighten_vol":     (3, 12),    # inner bot level vol: mean=6.5, range=1-12
        "tighten_hs":      (1, 5),     # inner level half-spread (1-4 ticks from fair)
        "fill_prob":       0.037,      # market trades / valid ticks
        "fill_qty":        (3, 8),     # trade qty: mean=5.2, range=2-8
        "limit":           80,
    },
}

PRODUCTS = list(PRODUCT_PARAMS.keys())


# ==============================================================
#  SYNTHETIC MARKET DATA GENERATOR
# ==============================================================

def _make_book(rng, fair_int, p):
    """Build one OrderDepth from product params p at given fair_int."""
    depth = OrderDepth()

    hs   = int(rng.choice(p["half_spreads"], p=p["half_spread_ps"]))
    l1v  = int(rng.integers(*p["l1_vol"]))
    l2v  = int(rng.integers(*p["l2_vol"]))
    l2o  = int(rng.integers(*p["l2_offset"]))

    best_bid = fair_int - hs
    best_ask = fair_int + hs

    if rng.random() < p["tighten_prob"]:
        tv    = int(rng.integers(*p["tighten_vol"]))
        inner = int(rng.integers(*p["tighten_hs"]))
        inner = max(1, min(inner, hs - 1))  # clamp: must be strictly inside spread
        if rng.random() < 0.5:
            inner_bid = fair_int - inner
            depth.buy_orders  = {inner_bid: tv,   best_bid: l1v, best_bid - l2o: l2v}
            depth.sell_orders = {best_ask: -l1v,  best_ask + l2o: -l2v}
        else:
            inner_ask = fair_int + inner
            depth.buy_orders  = {best_bid: l1v,   best_bid - l2o: l2v}
            depth.sell_orders = {inner_ask: -tv,  best_ask: -l1v, best_ask + l2o: -l2v}
    else:
        depth.buy_orders  = {best_bid: l1v,   best_bid - l2o: l2v}
        depth.sell_orders = {best_ask: -l1v,  best_ask + l2o: -l2v}

    return depth


def generate_books(rng, product, n_ticks):
    """Generate n_ticks order books for a product. Returns (books, fairs)."""
    p     = PRODUCT_PARAMS[product]
    books = []
    fairs = []
    fair  = float(p["fair"])

    for _ in range(n_ticks):
        fair += rng.normal(p["drift_mean"], p["drift_std"])
        fairs.append(fair)
        books.append(_make_book(rng, round(fair), p))

    return books, fairs


def generate_day(rng, n_ticks=2000):
    """Generate one full trading day. Returns (ticks_dict, mids_dict)."""
    all_books = {}
    all_fairs = {}
    for prod in PRODUCTS:
        books, fairs = generate_books(rng, prod, n_ticks)
        all_books[prod] = books
        all_fairs[prod] = fairs

    ticks = {}
    mids  = {}
    for i in range(n_ticks):
        ts = i * 100
        ticks[ts] = {prod: all_books[prod][i] for prod in PRODUCTS}
        mids[ts]  = {}
        for prod in PRODUCTS:
            depth = all_books[prod][i]
            bb = max(depth.buy_orders.keys())
            ba = min(depth.sell_orders.keys())
            mids[ts][prod] = (bb + ba) / 2.0

    return ticks, mids


# ==============================================================
#  MATCHING ENGINE
# ==============================================================

def match_aggressive(orders, book_depth, position, limit):
    fills   = []
    passive = []

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
    """Stochastic passive fill based on product-specific fill probability."""
    fills = []
    p     = PRODUCT_PARAMS.get(product, {})
    prob  = p.get("fill_prob", 0.02)
    fq_lo, fq_hi = p.get("fill_qty", (2, 7))
    limit = p.get("limit", 50)

    for order in passive_orders:
        if rng.random() >= prob:
            continue
        fill_qty = int(rng.integers(fq_lo, fq_hi + 1))

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

def simulate_one(TraderClass, rng, n_ticks=2000, return_trace=False):
    ticks, mids = generate_day(rng, n_ticks)

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
            result, conversions, trader_data = trader.run(state)
        except Exception:
            result      = {}
            trader_data = ""

        for product, orders in result.items():
            limit = PRODUCT_PARAMS.get(product, {}).get("limit", 50)
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
            ppnl      = {}
            for prod in PRODUCTS:
                mid         = mids[ts].get(prod, 0)
                pnl         = cash[prod] + positions[prod] * mid
                ppnl[prod]  = pnl
                total_pnl  += pnl
            trace.append({
                "ts":          ts,
                "total_pnl":   total_pnl,
                "product_pnl": dict(ppnl),
                "positions":   dict(positions),
            })

    last_ts    = timestamps[-1]
    total_pnl  = 0.0
    product_pnl = {}
    for prod in PRODUCTS:
        mid             = mids[last_ts].get(prod, 0)
        pnl             = cash[prod] + positions[prod] * mid
        product_pnl[prod] = pnl
        total_pnl      += pnl

    return total_pnl, product_pnl, dict(positions), trace


# ==============================================================
#  MONTE CARLO RUNNER
# ==============================================================

def run_monte_carlo(TraderClass, n_sims=200, n_ticks=2000, seed=42):
    rng          = np.random.default_rng(seed)
    results      = []
    pnls_by_prod = {p: [] for p in PRODUCTS}
    sample_trace = None

    for i in range(n_sims):
        child_rng    = np.random.default_rng(rng.integers(0, 2**31))
        return_trace = (i == 0)

        total, ppnl, pos, trace = simulate_one(
            TraderClass, child_rng, n_ticks, return_trace=return_trace
        )

        results.append(total)
        for prod in PRODUCTS:
            pnls_by_prod[prod].append(ppnl.get(prod, 0))

        if return_trace:
            sample_trace = trace

        if (i + 1) % 50 == 0:
            print(f"    Completed {i + 1}/{n_sims} simulations...")

    return np.array(results), {p: np.array(v) for p, v in pnls_by_prod.items()}, sample_trace


# ==============================================================
#  VISUALIZATION
# ==============================================================

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

def plot_results(results, pnls_by_prod, sample_trace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Monte Carlo Backtester - IMC Prosperity",
        fontsize=14, fontweight="bold"
    )

    # ── Panel 1: Total PnL Distribution ──
    ax = axes[0, 0]
    ax.hist(results, bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(results),   color="red",    linewidth=2,
               label=f"Mean: {np.mean(results):.0f}")
    ax.axvline(np.median(results), color="orange", linewidth=2, linestyle="--",
               label=f"Median: {np.median(results):.0f}")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Total PnL Distribution (n={len(results)})")
    ax.set_xlabel("PnL (SeaShells)")
    ax.set_ylabel("Count")
    ax.legend()

    # ── Panel 2: Per-product PnL ──
    ax = axes[0, 1]
    for i, (prod, pnl_arr) in enumerate(pnls_by_prod.items()):
        short = prod.replace("_", " ").title()
        ax.hist(pnl_arr, bins=30, alpha=0.6, color=COLORS[i % len(COLORS)],
                label=f"{short}: {np.mean(pnl_arr):.0f}+/-{np.std(pnl_arr):.0f}")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Per-Product PnL Distribution")
    ax.set_xlabel("PnL")
    ax.legend(fontsize=8)

    # ── Panel 3: Sample PnL Trace ──
    ax = axes[1, 0]
    if sample_trace:
        ts    = [r["ts"] for r in sample_trace]
        total = [r["total_pnl"] for r in sample_trace]
        ax.plot(ts, total, "b-", linewidth=1.5, label="Total")
        for i, prod in enumerate(PRODUCTS):
            short = prod.replace("_", " ").title()
            prod_pnl = [r["product_pnl"].get(prod, 0) for r in sample_trace]
            ax.plot(ts, prod_pnl, color=COLORS[i % len(COLORS)],
                    linewidth=1, alpha=0.6, label=short)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Sample Run - PnL Trace")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("PnL")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Panel 4: Sample Position Trace ──
    ax = axes[1, 1]
    if sample_trace:
        for i, prod in enumerate(PRODUCTS):
            short    = prod.replace("_", " ").title()
            pos_list = [r["positions"].get(prod, 0) for r in sample_trace]
            ax.plot(ts, pos_list, color=COLORS[i % len(COLORS)],
                    linewidth=1, label=short)
        limit = max(p["limit"] for p in PRODUCT_PARAMS.values())
        ax.axhline(0,      color="black", linewidth=0.5)
        ax.axhline( limit, color="gray",  linewidth=0.5, linestyle="--")
        ax.axhline(-limit, color="gray",  linewidth=0.5, linestyle="--")
        ax.set_title("Sample Run - Position")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Position")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = "mc_backtest_output.png"
    plt.savefig(outpath, dpi=150)
    print(f"\n  Chart saved: {outpath}")
    plt.close()


# ==============================================================
#  MAIN
# ==============================================================

def main():
    # ── Parse argument ──
    if len(sys.argv) < 2:
        print("Usage: python mc_backtester.py <algorithm_file.py>")
        sys.exit(1)

    algo_path = sys.argv[1]

    # ── Configuration ──
    N_SIMS  = 200
    N_TICKS = 2000
    SEED    = 42

    print("=" * 60)
    print("  IMC PROSPERITY - MONTE CARLO BACKTESTER")
    print("=" * 60)
    print(f"\n  Algorithm:   {algo_path}")
    print(f"  Simulations: {N_SIMS}")
    print(f"  Ticks/day:   {N_TICKS}")
    print(f"  Products:    {', '.join(PRODUCTS)}")

    print(f"\n  Loading algorithm...")
    TraderClass = load_trader(algo_path)
    print(f"  Trader class loaded OK.")

    print(f"\n  Running simulations...")
    results, pnls_by_prod, sample_trace = run_monte_carlo(
        TraderClass, N_SIMS, N_TICKS, SEED
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

    for prod, pnl_arr in pnls_by_prod.items():
        short = prod.replace("_", " ").title()
        print(f"\n  {short}:")
        print(f"    Mean: {np.mean(pnl_arr):>8.1f} +/- {np.std(pnl_arr):.1f}")

    pcts = [5, 25, 50, 75, 95]
    print(f"\n  Percentiles (total PnL):")
    for p in pcts:
        print(f"    {p:3d}th: {np.percentile(results, p):>10.1f}")

    plot_results(results, pnls_by_prod, sample_trace)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
