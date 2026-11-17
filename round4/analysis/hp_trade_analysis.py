"""
Replay the HP algo against the recorded day-3 book and analyse every fill.

For each tick we:
  1. Reconstruct the OrderDepth from activitiesLog (L1 + L2).
  2. Call Trader.run() to get the orders the algo intended to place.
  3. Simulate fills:
       - Aggressive (price crosses the book) — fills against recorded volume.
       - Passive bid at bb+1 / ask at ba-1 — stochastic fill with prob p_imp,
         capped by recorded L1 volume on the opposite side at the next tick.
       - Passive at the touch (bb / ba) — stochastic fill with prob p_touch.
       - Worse than touch — assumed never to fill (passive but stale).
     p_imp / p_touch are calibrated so total simulated PnL matches the actual
     +864 from graphLog.
  4. For each fill, snapshot: position before, recent mid slope (10-tick),
     recent volume imbalance (bv-av)/(bv+av), spread regime, mid+30 ahead.

Outputs a CSV-ish summary table to stdout.
"""

import sys, json, math, statistics, random
from collections import deque

sys.path.insert(0, ".")
import importlib.util, types

# ── datamodel stubs (mirror IMC's) ───────────────────────────────────────────

class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol, self.price, self.quantity = symbol, int(price), int(quantity)

class OrderDepth:
    def __init__(self):
        self.buy_orders, self.sell_orders = {}, {}

class TradingState:
    def __init__(self):
        self.traderData = ""; self.timestamp = 0
        self.listings = {}; self.order_depths = {}
        self.own_trades = {}; self.market_trades = {}
        self.position = {}; self.observations = None

class Listing:
    def __init__(self, *a, **k): pass
class Trade:
    def __init__(self, *a, **k): pass

UserId = str; Symbol = str; Product = str; Position = int; Observation = object

def _build_dm():
    m = types.ModuleType("datamodel")
    for n in ("Order","OrderDepth","TradingState","Listing","Trade",
             "UserId","Symbol","Product","Position","Observation"):
        setattr(m, n, globals()[n])
    return m
sys.modules.setdefault("datamodel", _build_dm())

spec = importlib.util.spec_from_file_location("hp_algo", "round.4.algo.HP.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
TraderClass = mod.Trader

# ── parse activitiesLog ──────────────────────────────────────────────────────

data = json.loads(open("501140_log/501140.json").read())
csv  = data["activitiesLog"].split("\n")
hdr  = csv[0].split(";")
idx  = {h:i for i,h in enumerate(hdr)}

ticks = []
for line in csv[1:]:
    if not line: continue
    f = line.split(";")
    if f[idx["product"]] != "HYDROGEL_PACK": continue
    def fnum(k):
        v = f[idx[k]]
        return float(v) if v else None
    ticks.append({
        "ts":   int(f[idx["timestamp"]]),
        "bb1":  int(fnum("bid_price_1")),  "bv1": int(fnum("bid_volume_1")),
        "ba1":  int(fnum("ask_price_1")),  "av1": int(fnum("ask_volume_1")),
        "bb2":  int(fnum("bid_price_2") or 0) or None, "bv2": int(fnum("bid_volume_2") or 0) or None,
        "ba2":  int(fnum("ask_price_2") or 0) or None, "av2": int(fnum("ask_volume_2") or 0) or None,
        "mid":  fnum("mid_price"),
        "pnl":  fnum("profit_and_loss"),
    })
ticks.sort(key=lambda x: x["ts"])

# ── replay loop with fill model ──────────────────────────────────────────────

P_IMP   = 0.04    # calibrated so total simulated PnL ≈ actual +865
P_TOUCH = 0.01    # touch-level passive fill prob
SEED    = 42
rng = random.Random(SEED)

trader = TraderClass()
trader_data = ""
position = 0
cash = 0.0
fills = []                         # list of dicts
mid_hist = deque(maxlen=20)        # recent mids for slope

for i, t in enumerate(ticks):
    od = OrderDepth()
    od.buy_orders[t["bb1"]]  =  t["bv1"]
    od.sell_orders[t["ba1"]] = -t["av1"]
    if t["bb2"]: od.buy_orders[t["bb2"]]  =  t["bv2"]
    if t["ba2"]: od.sell_orders[t["ba2"]] = -t["av2"]

    state = TradingState()
    state.timestamp     = t["ts"]
    state.traderData    = trader_data
    state.order_depths  = {"HYDROGEL_PACK": od}
    state.position      = {"HYDROGEL_PACK": position}
    state.own_trades    = {}; state.market_trades = {}

    result, _conv, trader_data = trader.run(state)
    orders = result.get("HYDROGEL_PACK", [])

    bb, ba = t["bb1"], t["ba1"]
    spread = ba - bb
    mid    = t["mid"]
    mid_hist.append(mid)

    # Recent slope (10-tick): (mid_now - mid_10_ago) / 10
    slope = (mid - mid_hist[0]) / max(1, len(mid_hist) - 1)
    # Volume imbalance
    vbi   = (t["bv1"] - t["av1"]) / (t["bv1"] + t["av1"])

    LIMIT = 200
    for o in orders:
        qty  = o.quantity
        prc  = o.price
        sym  = o.symbol

        if qty > 0:
            # buy: aggressive if prc >= ba; passive otherwise
            if prc >= ba:
                avail = -od.sell_orders.get(ba, 0)
                fq = min(qty, avail, LIMIT - position)
                if fq > 0:
                    fills.append(dict(ts=t["ts"], side="B", price=ba, qty=fq,
                                      kind="agg", mid=mid, spread=spread,
                                      slope=slope, vbi=vbi, pos_before=position))
                    cash -= ba * fq; position += fq
            else:
                p_fill = P_IMP if prc > bb else (P_TOUCH if prc == bb else 0.0)
                if p_fill > 0 and rng.random() < p_fill:
                    cap = t["av1"] // 4 + 1   # conservative passive cap
                    fq = min(qty, cap, LIMIT - position)
                    if fq > 0:
                        fills.append(dict(ts=t["ts"], side="B", price=prc, qty=fq,
                                          kind="pas_imp" if prc > bb else "pas_touch",
                                          mid=mid, spread=spread,
                                          slope=slope, vbi=vbi, pos_before=position))
                        cash -= prc * fq; position += fq
        elif qty < 0:
            sq = -qty
            if prc <= bb:
                avail = od.buy_orders.get(bb, 0)
                fq = min(sq, avail, LIMIT + position)
                if fq > 0:
                    fills.append(dict(ts=t["ts"], side="S", price=bb, qty=fq,
                                      kind="agg", mid=mid, spread=spread,
                                      slope=slope, vbi=vbi, pos_before=position))
                    cash += bb * fq; position -= fq
            else:
                p_fill = P_IMP if prc < ba else (P_TOUCH if prc == ba else 0.0)
                if p_fill > 0 and rng.random() < p_fill:
                    cap = t["bv1"] // 4 + 1
                    fq = min(sq, cap, LIMIT + position)
                    if fq > 0:
                        fills.append(dict(ts=t["ts"], side="S", price=prc, qty=fq,
                                          kind="pas_imp" if prc < ba else "pas_touch",
                                          mid=mid, spread=spread,
                                          slope=slope, vbi=vbi, pos_before=position))
                        cash += prc * fq; position -= fq

# ── post-process fills: forward mid, signed PnL contribution ─────────────────

ts_to_mid = {t["ts"]: t["mid"] for t in ticks}
sorted_ts = [t["ts"] for t in ticks]
def forward_mid(ts0, lookahead_ticks):
    target = ts0 + lookahead_ticks * 100
    # nearest ≤ target
    return ts_to_mid.get(target) or ticks[-1]["mid"]

for f in fills:
    fwd30 = forward_mid(f["ts"], 30)
    sign  = +1 if f["side"] == "B" else -1
    f["fwd30"]    = fwd30
    f["signal_pnl"] = sign * (fwd30 - f["price"]) * f["qty"]
    f["spread_pnl"] = sign * (f["mid"] - f["price"]) * f["qty"]

total_sim_pnl   = cash + position * ticks[-1]["mid"]
spread_captured = sum(f["spread_pnl"] for f in fills)
signal_pnl      = sum(f["signal_pnl"] for f in fills)

print(f"{'='*60}")
print(f"  HP ALGO REPLAY — day-3 (501140)")
print(f"{'='*60}")
print(f"  actual final PnL:    +{data['profit']:.2f}")
print(f"  simulated final PnL: {total_sim_pnl:+.2f}")
print(f"  end position:        {position}")
print(f"  total fills:         {len(fills)}")
print(f"  spread captured:     {spread_captured:+.1f}")
print(f"  signal pnl (30-tk):  {signal_pnl:+.1f}")
print()

# Bucket fills by direction × kind
from collections import defaultdict
buckets = defaultdict(list)
for f in fills:
    buckets[(f["side"], f["kind"])].append(f)
print(f"  {'side':>4s} {'kind':>10s} {'n':>4s} {'avg_qty':>8s} {'avg_signal_pnl':>15s} {'win%':>6s}")
for (side, kind), arr in sorted(buckets.items()):
    avg_q   = sum(f["qty"] for f in arr) / len(arr)
    avg_sp  = sum(f["signal_pnl"] for f in arr) / len(arr)
    win_pct = sum(1 for f in arr if f["signal_pnl"] > 0) / len(arr) * 100
    print(f"  {side:>4s} {kind:>10s} {len(arr):>4d} {avg_q:>8.1f} {avg_sp:>+15.2f} {win_pct:>5.0f}%")

# Winning vs losing fills: signal histogram
def stats(arr, key):
    vals = [f[key] for f in arr]
    if not vals: return "n/a"
    return f"mean={statistics.mean(vals):+.2f} med={statistics.median(vals):+.2f}"

winners = [f for f in fills if f["signal_pnl"] > 0]
losers  = [f for f in fills if f["signal_pnl"] < 0]
print(f"\n  Winners ({len(winners)}) vs Losers ({len(losers)})  [signed by side]:")
for key in ["slope", "vbi", "spread", "pos_before"]:
    w = [f[key] * (+1 if f["side"]=="B" else -1) if key in ("slope","vbi","pos_before") else f[key] for f in winners]
    l = [f[key] * (+1 if f["side"]=="B" else -1) if key in ("slope","vbi","pos_before") else f[key] for f in losers]
    if w and l:
        print(f"    {key:12s}  winners {statistics.mean(w):+8.3f}   losers {statistics.mean(l):+8.3f}   diff {statistics.mean(w)-statistics.mean(l):+.3f}")

# Phase-by-phase
print(f"\n  {'phase':>30s} {'n_fills':>8s} {'spread_pnl':>11s} {'signal_pnl':>11s} {'avg_pos':>8s}")
phases = [(0,9999),(10000,29999),(30000,49999),(50000,69999),(70000,89999),(90000,100000)]
for lo, hi in phases:
    sub = [f for f in fills if lo <= f["ts"] <= hi]
    if not sub: continue
    sp = sum(f["spread_pnl"] for f in sub)
    sg = sum(f["signal_pnl"] for f in sub)
    ap = sum(f["pos_before"] for f in sub) / len(sub)
    print(f"  {f'ts {lo:>6}..{hi:>6}':>30s} {len(sub):>8d} {sp:>+11.1f} {sg:>+11.1f} {ap:>+8.1f}")

# Worst 10 individual losers + best 10 winners (by signal_pnl)
print(f"\n  Top 10 losing fills (by 30-tick fwd):")
for f in sorted(fills, key=lambda x: x["signal_pnl"])[:10]:
    print(f"    ts={f['ts']:>6} {f['side']} {f['qty']:>3}@{f['price']:>5} "
          f"mid={f['mid']:>6.1f} fwd30={f['fwd30']:>6.1f} "
          f"slope={f['slope']:+.2f} vbi={f['vbi']:+.2f} pos={f['pos_before']:+4d} "
          f"signal_pnl={f['signal_pnl']:+.1f}")

print(f"\n  Top 10 winning fills (by 30-tick fwd):")
for f in sorted(fills, key=lambda x: -x["signal_pnl"])[:10]:
    print(f"    ts={f['ts']:>6} {f['side']} {f['qty']:>3}@{f['price']:>5} "
          f"mid={f['mid']:>6.1f} fwd30={f['fwd30']:>6.1f} "
          f"slope={f['slope']:+.2f} vbi={f['vbi']:+.2f} pos={f['pos_before']:+4d} "
          f"signal_pnl={f['signal_pnl']:+.1f}")
