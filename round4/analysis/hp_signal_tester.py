"""
HP signal tester.

Replays the HP order book from a recorded submission log, generates a
hypothetical passive fill at bb+1 (buy) and ba-1 (sell) on every tick,
and evaluates a battery of signals against forward mid PnL at multiple
horizons.

Output is one line per signal — copy the whole block and paste it back
in chat.

USAGE:
    python hp_signal_tester.py                          # uses 501140_log/
    python hp_signal_tester.py 501140_log               # any submission dir
    python hp_signal_tester.py 501140_log 50            # custom horizon

Each line shows:
    name   n=N   r10=X.XX  r30=X.XX  r60=X.XX  | Q1..Q5 mean signed_pnl_30
                                                | Q1 win% / Q5 win%

A signal that separates winners from losers monotonically is what we
want. Best signal: |r30| large, monotonic Q1..Q5, big Q1 vs Q5 spread.

The script is read-only — it does not write any files. Output goes to
stdout. No dependencies beyond the Python stdlib.
"""

import sys, os, json, math, statistics
from collections import deque

PRODUCT       = "HYDROGEL_PACK"
DEFAULT_LOG   = "501140_log"
DEFAULT_HORIZ = 30           # ticks
HORIZONS      = (10, 30, 60) # ticks to evaluate signed PnL at


# ── 1. find the activitiesLog json in the chosen log dir ────────────────────

def find_log(arg_or_dir: str) -> str:
    if os.path.isfile(arg_or_dir) and arg_or_dir.endswith(".json"):
        return arg_or_dir
    if os.path.isdir(arg_or_dir):
        for f in os.listdir(arg_or_dir):
            if f.endswith(".json"):
                return os.path.join(arg_or_dir, f)
    raise FileNotFoundError(f"no .json log under {arg_or_dir}")


log_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
log_path = find_log(log_dir)
print(f"# log: {log_path}")

# ── 2. parse HP rows from activitiesLog ──────────────────────────────────────

data = json.loads(open(log_path).read())
csv  = data["activitiesLog"].split("\n")
hdr  = csv[0].split(";")
idx  = {h: i for i, h in enumerate(hdr)}

ticks = []
for line in csv[1:]:
    if not line:
        continue
    f = line.split(";")
    if f[idx["product"]] != PRODUCT:
        continue
    def fnum(k):
        v = f[idx[k]]
        return float(v) if v else None

    # All-level totals for full-book imbalance
    bv_total = sum(int(fnum(f"bid_volume_{i}") or 0) for i in (1, 2, 3))
    av_total = sum(int(fnum(f"ask_volume_{i}") or 0) for i in (1, 2, 3))

    ticks.append({
        "ts":   int(f[idx["timestamp"]]),
        "bb":   int(fnum("bid_price_1")),
        "ba":   int(fnum("ask_price_1")),
        "bv":   int(fnum("bid_volume_1")),
        "av":   int(fnum("ask_volume_1")),
        "bv_t": bv_total,
        "av_t": av_total,
        "mid":  fnum("mid_price"),
    })

ticks.sort(key=lambda x: x["ts"])
n_ticks = len(ticks)
print(f"# product: {PRODUCT}   n_ticks: {n_ticks}   day actual PnL: "
      f"{data.get('profit', 'n/a')}")
print(f"# horizons: {HORIZONS} ticks")
print()

# ── 3. precompute rolling features ───────────────────────────────────────────

mid     = [t["mid"]    for t in ticks]
spread  = [t["ba"] - t["bb"] for t in ticks]
vbi_L1  = [(t["bv"]   - t["av"])   / max(1, t["bv"]   + t["av"])   for t in ticks]
vbi_tot = [(t["bv_t"] - t["av_t"]) / max(1, t["bv_t"] + t["av_t"]) for t in ticks]


def rolling_mean(x, w):
    out = []
    s   = 0.0
    dq  = deque()
    for v in x:
        dq.append(v); s += v
        if len(dq) > w:
            s -= dq.popleft()
        out.append(s / len(dq))
    return out


def rolling_std(x, w):
    out = []
    dq  = deque()
    for v in x:
        dq.append(v)
        if len(dq) > w:
            dq.popleft()
        if len(dq) > 1:
            out.append(statistics.pstdev(dq))
        else:
            out.append(0.0)
    return out


def rolling_slope(x, w):
    """Slope of x over last w samples (least squares)."""
    out = []
    dq  = deque()
    for v in x:
        dq.append(v)
        if len(dq) > w:
            dq.popleft()
        if len(dq) >= 2:
            n  = len(dq); xs = list(range(n)); ys = list(dq)
            mx = sum(xs) / n; my = sum(ys) / n
            num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
            den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1.0
            out.append(num / den)
        else:
            out.append(0.0)
    return out


def rolling_max(x, w):
    out = []
    dq  = deque()
    for v in x:
        dq.append(v)
        if len(dq) > w:
            dq.popleft()
        out.append(max(dq))
    return out


def rolling_min(x, w):
    out = []
    dq  = deque()
    for v in x:
        dq.append(v)
        if len(dq) > w:
            dq.popleft()
        out.append(min(dq))
    return out


# Slopes
slope_5   = rolling_slope(mid, 5)
slope_10  = rolling_slope(mid, 10)
slope_20  = rolling_slope(mid, 20)
slope_50  = rolling_slope(mid, 50)
slope_100 = rolling_slope(mid, 100)

# Moving averages and distance-from-MA
ma_20  = rolling_mean(mid, 20)
ma_50  = rolling_mean(mid, 50)
ma_100 = rolling_mean(mid, 100)
ma_200 = rolling_mean(mid, 200)
dist_ma_20  = [mid[i] - ma_20[i]  for i in range(n_ticks)]
dist_ma_50  = [mid[i] - ma_50[i]  for i in range(n_ticks)]
dist_ma_100 = [mid[i] - ma_100[i] for i in range(n_ticks)]
dist_ma_200 = [mid[i] - ma_200[i] for i in range(n_ticks)]

# Range position (mid relative to last-N high/low)
def range_pct(window):
    hi = rolling_max(mid, window)
    lo = rolling_min(mid, window)
    return [(mid[i] - lo[i]) / max(1.0, hi[i] - lo[i]) - 0.5
            for i in range(n_ticks)]
range_50  = range_pct(50)
range_200 = range_pct(200)

# Realized vol of mid changes
mid_chg = [0.0] + [mid[i] - mid[i-1] for i in range(1, n_ticks)]
vol_20  = rolling_std(mid_chg, 20)
vol_50  = rolling_std(mid_chg, 50)

# Spread z-score
spread_ma  = rolling_mean(spread, 100)
spread_std = rolling_std(spread, 100)
spread_z   = [(spread[i] - spread_ma[i]) / max(0.01, spread_std[i])
              for i in range(n_ticks)]

# VBI smoothed
vbi_L1_smooth  = rolling_mean(vbi_L1,  10)
vbi_tot_smooth = rolling_mean(vbi_tot, 10)

# Time of day (centered at 0)
ts_norm = [(t["ts"] / max(1, ticks[-1]["ts"])) - 0.5 for t in ticks]

SIGNALS = {
    # Slopes — mean-reversion hypothesis says positive slope predicts negative
    # forward signed_pnl (i.e. trading WITH slope loses)
    "slope_5":     slope_5,
    "slope_10":    slope_10,
    "slope_20":    slope_20,
    "slope_50":    slope_50,
    "slope_100":   slope_100,
    # Distance from MA — same direction as slope under MR
    "dist_ma_20":  dist_ma_20,
    "dist_ma_50":  dist_ma_50,
    "dist_ma_100": dist_ma_100,
    "dist_ma_200": dist_ma_200,
    # Range position
    "range_50":    range_50,
    "range_200":   range_200,
    # Volume imbalance — bid pressure vs ask pressure
    "vbi_L1":          vbi_L1,
    "vbi_L1_smooth":   vbi_L1_smooth,
    "vbi_tot":         vbi_tot,
    "vbi_tot_smooth":  vbi_tot_smooth,
    # Vol regimes
    "vol_20":      vol_20,
    "vol_50":      vol_50,
    # Spread regime
    "spread_z":    spread_z,
    # Time-of-day
    "ts_norm":     ts_norm,
}

# ── 4. build hypothetical fills at every tick ────────────────────────────────
# For each tick, generate: buy@bb+1 and sell@ba-1
# signal_pnl_H = (mid_at_t+H - fill_price) * side_sign

def fwd_mid(i, h):
    return mid[min(i + h, n_ticks - 1)]

records = []   # list of (i, side_sign, fill_price, mid_now, signed_pnls...)
for i in range(n_ticks):
    bb, ba, m = ticks[i]["bb"], ticks[i]["ba"], mid[i]
    # buy
    fill_p = bb + 1
    pnls   = {h: (fwd_mid(i, h) - fill_p) for h in HORIZONS}
    records.append((i, +1, fill_p, m, pnls))
    # sell
    fill_p = ba - 1
    pnls   = {h: -(fwd_mid(i, h) - fill_p) for h in HORIZONS}
    records.append((i, -1, fill_p, m, pnls))

# ── 5. evaluate each signal ──────────────────────────────────────────────────

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ys[i] - my) ** 2 for i in range(n))
    den = math.sqrt(dx2 * dy2)
    return num / den if den > 0 else 0.0


def quintile_stats(signed_signals, signed_pnls):
    pairs = sorted(zip(signed_signals, signed_pnls))
    n = len(pairs)
    bounds = [int(n * k / 5) for k in range(6)]
    means, wins = [], []
    for k in range(5):
        chunk = pairs[bounds[k]:bounds[k+1]]
        if not chunk:
            means.append(0.0); wins.append(0.0); continue
        pnl_chunk = [p for _, p in chunk]
        means.append(statistics.mean(pnl_chunk))
        wins.append(sum(1 for p in pnl_chunk if p > 0) / len(chunk) * 100)
    return means, wins


# Header
print(f"{'signal':<18s} {'n':>5s}  "
      f"{'r10':>6s} {'r30':>6s} {'r60':>6s}  | "
      f"{'Q1':>6s} {'Q2':>6s} {'Q3':>6s} {'Q4':>6s} {'Q5':>6s}  "
      f"{'win1':>5s} {'win5':>5s}")

results = []
for name, sig_series in SIGNALS.items():
    signed_sig = [sig_series[r[0]] * r[1] for r in records]   # × side_sign
    signed_pnl = {h: [r[4][h] * r[1] for r in records] for h in HORIZONS}
    rs = {h: pearson(signed_sig, signed_pnl[h]) for h in HORIZONS}
    means, wins = quintile_stats(signed_sig, signed_pnl[DEFAULT_HORIZ])
    results.append((name, rs, means, wins))

# Sort by absolute correlation at default horizon — best at top
results.sort(key=lambda r: -abs(r[1][DEFAULT_HORIZ]))

for name, rs, means, wins in results:
    print(f"{name:<18s} {len(records):>5d}  "
          f"{rs[10]:>+6.3f} {rs[30]:>+6.3f} {rs[60]:>+6.3f}  | "
          f"{means[0]:>+6.2f} {means[1]:>+6.2f} {means[2]:>+6.2f} "
          f"{means[3]:>+6.2f} {means[4]:>+6.2f}  "
          f"{wins[0]:>4.0f}% {wins[4]:>4.0f}%")

print()
print("# How to read:")
print("#   r30 < 0  ->  trading WITH signal direction LOSES (mean-reversion)")
print("#   r30 > 0  ->  trading WITH signal direction WINS  (continuation)")
print("#   |r| > 0.10 is meaningful with this many samples")
print("#   Q1..Q5 are mean signed-pnl-30 by quintile of signed_signal")
print("#   monotonic Q1..Q5 with big Q1-Q5 spread = strong predictor")
print("#   win1 is win-rate in Q1 (signed_signal lowest -> trade most")
print("#   against signal); win5 is when most aligned with signal")
