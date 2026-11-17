"""
IMC Prosperity Round 4 -- Strategy Optimizer  (fast version)
============================================================
Replays actual Round 4 market data (prices + trades) across all 3 days.
Includes Mark-bot trades as fill signal for passive orders.
Uses flat arrays per product for fast simulation.

Usage:
    python round4_optimizer.py

Outputs ranked configurations with per-product PnL breakdown.
"""

import zipfile, csv, io, math, json, random, time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

ROUND4_ZIP    = r"C:\Users\talbo\Downloads\ROUND_4.zip"
DAYS          = [1, 2, 3]
EXPIRY_DAY    = 7
TICKS_PER_DAY = 1_000_000   # timestamps go 0..999900 per day
DOWNSAMPLE    = 5      # only process every Nth tick (5 = 2000 ticks/day, fast)
N_RANDOM      = 600    # random search configurations
N_WORKERS     = max(1, cpu_count() - 1)
SEED          = 42

POSITION_LIMITS = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]},
}
ALL_PRODUCTS = list(POSITION_LIMITS.keys())
VEV_STRIKES  = [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]

# ------------------------------------------------------------------------------
# BLACK-SCHOLES
# ------------------------------------------------------------------------------

def _ncdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = math.exp(-0.5 * x * x) * 0.3989422804
    p = 1.0 - d * t*(0.319381530 + t*(-0.356563782
        + t*(1.781477937 + t*(-1.821255978 + t*1.330274429))))
    return p if x >= 0 else 1.0 - p

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sv
    return S * _ncdf(d1) - K * _ncdf(d1 - sv)

def implied_vol(price, S, K, T):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5 or T <= 0 or S <= 0:
        return None
    lo, hi = 0.001, 5.0
    for _ in range(40):
        m = (lo + hi) * 0.5
        if bs_call(S, K, T, m) < price: lo = m
        else:                            hi = m
    return (lo + hi) * 0.5

# ------------------------------------------------------------------------------
# DATA LOADING  →  flat arrays per product
# ------------------------------------------------------------------------------
# Per product we store parallel arrays:
#   days[], timestamps[], bb[], ba[], bvol[], avol[], mid[]
#   mark_sell_qty[]  -- total qty where Mark bots sold at each tick (hits bids → fills our bids)
#   mark_buy_qty[]   -- total qty where Mark bots bought at each tick (lifts asks → fills our asks)

class ProductData:
    __slots__ = ('days','ts','bb','ba','bvol','avol','mid',
                 'mark_sell_qty','mark_buy_qty')
    def __init__(self):
        self.days         = []
        self.ts           = []
        self.bb           = []
        self.ba           = []
        self.bvol         = []
        self.avol         = []
        self.mid          = []
        self.mark_sell_qty = []  # qty sold by Mark bots (fills passive buys)
        self.mark_buy_qty  = []  # qty bought by Mark bots (fills passive sells)


def load_data(zip_path):
    """Returns dict[product] -> ProductData (downsampled)."""

    # -- Read all prices ------------------------------------------------------
    # raw_prices[day][ts][product] = (bb, ba, bvol, avol, mid)
    raw = defaultdict(lambda: defaultdict(dict))

    with zipfile.ZipFile(zip_path) as z:
        for day in DAYS:
            content = z.read(f"prices_round_4_day_{day}.csv").decode()
            reader  = csv.DictReader(io.StringIO(content), delimiter=';')
            tick_counter = defaultdict(int)
            for row in reader:
                ts   = int(row['timestamp'])
                prod = row['product']
                tick_counter[prod] += 1
                if tick_counter[prod] % DOWNSAMPLE != 0:
                    continue
                try:
                    bb   = float(row['bid_price_1']) if row['bid_price_1'] else 0.0
                    ba   = float(row['ask_price_1']) if row['ask_price_1'] else 0.0
                    bvol = int(row['bid_volume_1'])  if row['bid_volume_1'] else 0
                    avol = int(row['ask_volume_1'])  if row['ask_volume_1'] else 0
                    mid  = float(row['mid_price'])   if row['mid_price']   else 0.0
                except (ValueError, KeyError):
                    continue
                raw[day][ts][prod] = (bb, ba, bvol, avol, mid)

        # -- Read all trades --------------------------------------------------
        # mark_trades[day][ts][product] = (half_qty, half_qty)
        # Each Mark-bot trade has one buyer and one seller. We don't know which
        # was aggressive, so we split 50/50: half fills our passive sells (buy
        # side activity) and half fills our passive buys (sell side activity).
        mark_trades = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
        for day in DAYS:
            content = z.read(f"trades_round_4_day_{day}.csv").decode()
            reader  = csv.DictReader(io.StringIO(content), delimiter=';')
            for row in reader:
                try:
                    ts   = int(row['timestamp'])
                    prod = row['symbol']
                    qty  = int(row['quantity'])
                    half = max(1, qty // 2)
                    mark_trades[day][ts][prod][0] += half  # fills our passive sells
                    mark_trades[day][ts][prod][1] += half  # fills our passive buys
                except (ValueError, KeyError):
                    continue

    # -- Build ProductData arrays ---------------------------------------------
    result = {p: ProductData() for p in ALL_PRODUCTS}

    for day in DAYS:
        for ts in sorted(raw[day].keys()):
            for prod in ALL_PRODUCTS:
                if prod not in raw[day][ts]:
                    continue
                bb, ba, bvol, avol, mid = raw[day][ts][prod]
                if bb <= 0 or ba <= 0:
                    continue
                mt        = mark_trades[day][ts].get(prod, [0, 0])
                pd        = result[prod]
                pd.days.append(day)
                pd.ts.append(ts)
                pd.bb.append(bb)
                pd.ba.append(ba)
                pd.bvol.append(bvol)
                pd.avol.append(avol)
                pd.mid.append(mid)
                pd.mark_buy_qty.append(mt[0])
                pd.mark_sell_qty.append(mt[1])

    return result


# ------------------------------------------------------------------------------
# SINGLE-PRODUCT SIMULATION
# (called independently per product; returns cash pnl as float)
# ------------------------------------------------------------------------------

def sim_hp(pd: ProductData, cfg: dict) -> float:
    hp    = cfg['hp']
    strat = hp['strategy']
    alpha = hp['ema_alpha']
    hs    = hp['half_spread']
    sz    = hp['size']
    skew_p= hp.get('pos_skew', 1.0)
    take  = hp.get('take_thresh', 0)
    tak_sz= hp.get('take_size', sz)
    sk_sc = hp.get('skew_scale', 2)
    limit = 200

    pos   = 0
    cash  = 0.0
    ema   = pd.mid[0] if pd.mid else 9990.0

    for i in range(len(pd.ts)):
        bb   = pd.bb[i];  ba   = pd.ba[i]
        bvol = pd.bvol[i];avol = pd.avol[i]
        mid  = pd.mid[i]
        ms   = pd.mark_sell_qty[i]   # Mark sells → fills our passive BUYS
        mb   = pd.mark_buy_qty[i]    # Mark buys  → fills our passive SELLS
        ema += alpha * (mid - ema)
        dev  = mid - ema

        buy_room  = limit - pos
        sell_room = limit + pos

        # -- Step 1: Aggressive mean-reversion takes (mm strategy only) -------
        if strat == 'mm' and take > 0:
            if dev < -take and buy_room > 0:
                q = min(tak_sz, buy_room, avol)
                if q > 0:
                    cash -= ba * q;  pos += q
                    buy_room = limit - pos;  sell_room = limit + pos
            elif dev > take and sell_room > 0:
                q = min(tak_sz, sell_room, bvol)
                if q > 0:
                    cash += bb * q;  pos -= q
                    buy_room = limit - pos;  sell_room = limit + pos

        # -- Step 2: Generate passive quote prices -----------------------------
        bid_px = bid_qty = ask_px = ask_qty = 0

        if strat == 'mm':
            skew   = int((pos / limit) * skew_p)
            bid_px = int(ema - hs - skew)
            ask_px = int(ema + hs - skew) + 1
            # Clamp inside market so these are passive (not aggressive)
            bid_px = min(bid_px, int(bb))
            ask_px = max(ask_px, int(ba))
            if bid_px >= ask_px: bid_px = ask_px - 1
            bid_qty = min(sz, buy_room)
            ask_qty = min(sz, sell_room)

        elif strat == 'mr_both':
            skew   = round(max(-sk_sc, min(sk_sc, dev / max(abs(dev), 0.1))) * sk_sc)
            bid_px = max(int(bb) - 3, min(int(bb) + 1 - skew, int(bb) + 1))
            ask_px = max(int(ba) - 1, min(int(ba) - 1 - skew, int(ba) + 3))
            if bid_px >= ask_px: bid_px = ask_px - 1
            bid_qty = min(sz, buy_room)
            ask_qty = min(sz, sell_room)

        else:  # mr_one
            if dev > 0 and sell_room > 0:
                ask_px  = int(ba) - 1
                ask_qty = min(sz, sell_room)
            elif dev <= 0 and buy_room > 0:
                bid_px  = int(bb) + 1
                bid_qty = min(sz, buy_room)

        # -- Step 3: Passive fills from Mark-bot trades ------------------------
        # Mark sells → hit bids → fills our passive BUYS at bid_px
        if ms > 0 and bid_qty > 0 and buy_room > 0:
            q = min(bid_qty, ms, buy_room)
            cash -= bid_px * q;  pos += q
            buy_room = limit - pos

        # Mark buys → lift asks → fills our passive SELLS at ask_px
        if mb > 0 and ask_qty > 0 and sell_room > 0:
            q = min(ask_qty, mb, sell_room)
            cash += ask_px * q;  pos -= q

    return cash + pos * pd.mid[-1] if pd.mid else cash


def sim_vfe(pd: ProductData, cfg: dict):
    """Returns (cash_pnl, ema_history) -- ema_history for options use."""
    vfe   = cfg['vfe']
    alpha = vfe['ema_alpha']
    take  = vfe['take_thresh']
    hs    = vfe['half_spread']
    sz    = vfe['passive_size']
    skew_s= vfe['pos_skew']
    limit = 200

    pos   = 0
    cash  = 0.0
    ema   = pd.mid[0] if pd.mid else 5248.0
    mids  = []   # raw mid at each tick for options

    for i in range(len(pd.ts)):
        bb   = pd.bb[i];  ba   = pd.ba[i]
        bvol = pd.bvol[i];avol = pd.avol[i]
        mid  = pd.mid[i]
        ms   = pd.mark_sell_qty[i]
        mb   = pd.mark_buy_qty[i]
        ema += alpha * (mid - ema)
        mids.append(mid)
        fair = ema

        buy_room  = limit - pos
        sell_room = limit + pos

        # Aggressive takes
        if buy_room > 0 and ba < fair - take:
            q = min(avol, buy_room)
            if q > 0:
                cash -= ba * q;  pos += q;  buy_room -= q
        if sell_room > 0 and bb > fair + take:
            q = min(bvol, sell_room)
            if q > 0:
                cash += bb * q;  pos -= q;  sell_room -= q

        # Passive quotes
        skew    = (pos / limit) * skew_s
        bid_px  = int(fair - hs - skew)
        ask_px  = int(fair + hs - skew) + 1
        if ba is not None and bid_px >= ba: bid_px = int(ba) - 1
        if bb is not None and ask_px <= bb: ask_px = int(bb) + 1
        if bid_px >= ask_px: bid_px = ask_px - 1

        bid_qty = min(sz, buy_room)
        ask_qty = min(sz, sell_room)

        # aggressive check
        if bid_px >= ba and buy_room > 0:
            q = min(bid_qty, avol, buy_room)
            if q > 0:
                cash -= ba * q;  pos += q;  buy_room -= q
                bid_qty = 0
        if ask_px <= bb and sell_room > 0:
            q = min(ask_qty, bvol, sell_room)
            if q > 0:
                cash += bb * q;  pos -= q;  sell_room -= q
                ask_qty = 0

        # passive fills
        if ms > 0 and bid_qty > 0 and buy_room > 0:
            q = min(bid_qty, ms, buy_room)
            cash -= bid_px * q;  pos += q
        if mb > 0 and ask_qty > 0 and sell_room > 0:
            q = min(ask_qty, mb, sell_room)
            cash += ask_px * q;  pos -= q

    return cash + pos * pd.mid[-1], mids


def sim_vev(pd: ProductData, strike: int, cfg: dict, vfe_mids: list) -> float:
    """Simulate one VEV option series. vfe_mids aligned by index with pd.ts."""
    vev   = cfg['vev']
    strat = vev['strategy']
    skip  = vev.get('skip_strikes', set())
    if strike in skip:
        return 0.0

    alpha_iv  = vev['iv_ema_alpha']
    iv_prior  = vev['iv_prior']
    alpha_mr  = vev['vfe_mr_alpha']
    edge      = vev['edge_take']
    agg_max   = vev['agg_max_size']
    pas_sz    = vev['passive_size']
    mm_margin = vev.get('mm_margin', 1)
    limit     = 300

    pos       = 0
    cash      = 0.0
    iv_ema    = iv_prior
    vfe_mr    = vfe_mids[0] if vfe_mids else 5248.0

    # Build a lookup from index to vfe_mid (pd.ts aligned with vfe_pd.ts approx)
    n = len(pd.ts)
    if n == 0:
        return 0.0

    # vfe_mids must be same length (both are same day+downsample ticks)
    # If lengths differ (products might have different tick counts), use modulo
    vfe_n = len(vfe_mids)

    for i in range(n):
        bb   = pd.bb[i];  ba   = pd.ba[i]
        bvol = pd.bvol[i];avol = pd.avol[i]
        mid  = pd.mid[i]
        ms   = pd.mark_sell_qty[i]
        mb   = pd.mark_buy_qty[i]
        day  = pd.days[i]
        ts   = pd.ts[i]

        vfe_spot = vfe_mids[i % vfe_n]
        vfe_mr  += alpha_mr * (vfe_spot - vfe_mr)

        T = max(0.0, (EXPIRY_DAY - day) - ts / TICKS_PER_DAY)
        spread = ba - bb
        buy_room  = limit - pos
        sell_room = limit + pos

        # IV calibration from this strike's mid
        if T > 0 and vfe_spot > 0:
            iv_c = implied_vol(mid, vfe_spot, float(strike), T)
            if iv_c is not None:
                iv_ema += alpha_iv * (iv_c - iv_ema)

        bid_qty = 0;  ask_qty = 0
        bid_px  = 0;  ask_px  = 0

        if strat == 'pure_mm':
            if spread < 2 or buy_room <= 0 and sell_room <= 0:
                continue
            pos_skew = round((pos / limit) * max(1, spread / 4))
            our_bid  = int(bb) + mm_margin - pos_skew
            our_ask  = int(ba) - mm_margin - pos_skew
            our_bid  = max(int(bb) + 1, min(our_bid, int(ba) - 1))
            our_ask  = max(int(bb) + 1, min(our_ask, int(ba) - 1))
            if our_bid >= our_ask: our_ask = our_bid + 1
            if our_ask >= ba:      continue
            bid_px = our_bid;  bid_qty = min(pas_sz, buy_room)
            ask_px = our_ask;  ask_qty = min(pas_sz, sell_room)

        else:  # bs_spot, bs_mr, aggressive, vol_surface all use BS fair value
            if T <= 0 or vfe_spot <= 0:
                continue
            S   = vfe_spot if strat in ('bs_spot','aggressive') else vfe_mr
            fair = bs_call(S, float(strike), T, iv_ema)

            # Aggressive takes
            if buy_room > 0 and (fair - ba) > edge:
                q = min(avol, agg_max, buy_room)
                if q > 0:
                    cash -= ba * q;  pos += q;  buy_room -= q
            if sell_room > 0 and (bb - fair) > edge:
                q = min(bvol, agg_max, sell_room)
                if q > 0:
                    cash += bb * q;  pos -= q;  sell_room -= q

            # Passive quotes
            if spread >= 2:
                sig_skew = -round(max(-2, min(2, (fair - mid) / max(edge, 0.5))))
                pos_skew =  round((pos / limit) * (spread / 4.0))
                skew     = sig_skew + pos_skew
                our_bid  = int(bb) + 1 - skew
                our_ask  = int(ba) - 1 - skew
                our_bid  = max(int(bb) + 1, min(our_bid, int(ba) - 1))
                our_ask  = max(int(bb) + 1, min(our_ask, int(ba) - 1))
                if our_bid >= our_ask: our_ask = our_bid + 1
                if our_ask < ba:
                    bid_px = our_bid;  bid_qty = min(pas_sz, buy_room)
                    ask_px = our_ask;  ask_qty = min(pas_sz, sell_room)

        # -- Passive fills ----------------------------------------------------
        if ms > 0 and bid_qty > 0 and buy_room > 0:
            q = min(bid_qty, ms, buy_room)
            cash -= bid_px * q;  pos += q

        if mb > 0 and ask_qty > 0 and sell_room > 0:
            q = min(ask_qty, mb, sell_room)
            cash += ask_px * q;  pos -= q

    return cash + pos * pd.mid[-1]


# ------------------------------------------------------------------------------
# FULL BACKTEST
# ------------------------------------------------------------------------------

def backtest(cfg: dict, pdata: dict) -> tuple:
    """Returns (total_pnl, per_product_pnl_dict)."""
    ppnl = {}

    hp_pnl  = sim_hp(pdata["HYDROGEL_PACK"], cfg)
    ppnl["HYDROGEL_PACK"] = hp_pnl

    vfe_pnl, vfe_mids = sim_vfe(pdata["VELVETFRUIT_EXTRACT"], cfg)
    ppnl["VELVETFRUIT_EXTRACT"] = vfe_pnl

    for strike in VEV_STRIKES:
        prod = f"VEV_{strike}"
        pnl  = sim_vev(pdata[prod], strike, cfg, vfe_mids)
        ppnl[prod] = pnl

    total = sum(ppnl.values())
    return total, ppnl


# ------------------------------------------------------------------------------
# PARAMETER SPACE
# ------------------------------------------------------------------------------

def build_configs(n: int, seed: int) -> list:
    rng = random.Random(seed)

    hp_strats  = ['mm', 'mr_both', 'mr_one']
    vev_strats = ['bs_spot', 'bs_mr', 'pure_mm', 'aggressive']
    skip_sets  = [
        set(),
        {4000, 4500},
        {4000, 4500, 6000, 6500},
        {4000, 4500, 5500, 6000, 6500},
        {4000, 4500, 5400, 5500, 6000, 6500},
    ]

    configs = []

    # -- Baseline: V.4 equivalent ---------------------------------------------
    baseline = {
        'hp':  {'strategy':'mr_one','ema_alpha':0.001,'half_spread':2,'size':5,
                'take_thresh':0,'take_size':5,'pos_skew':0.0,'skew_scale':2},
        'vfe': {'ema_alpha':0.005,'take_thresh':2.5,'half_spread':1.0,
                'passive_size':15,'pos_skew':1.0},
        'vev': {'strategy':'bs_mr','iv_prior':0.014,'iv_ema_alpha':0.005,
                'vfe_mr_alpha':0.01,'edge_take':4.0,'agg_max_size':30,
                'passive_size':10,'skip_strikes':{4000,4500},'mm_margin':1},
    }
    configs.append(('baseline', baseline))

    # -- Fixed grids for high-impact parameters --------------------------------

    # HP: full market-making grid
    for sz in [20, 40, 80, 120]:
        for hs in [3, 5, 7]:
            for ema in [0.002, 0.005, 0.01]:
                for take in [0, 5, 10]:
                    c = {
                        'hp':  {'strategy':'mm','ema_alpha':ema,'half_spread':hs,
                                'size':sz,'take_thresh':take,'take_size':sz//2,
                                'pos_skew':2.0,'skew_scale':2},
                        'vfe': dict(baseline['vfe']),
                        'vev': dict(baseline['vev']),
                    }
                    configs.append((f'hp_mm_sz{sz}_hs{hs}_ema{ema}_take{take}', c))

    # VEV pure market-making grid
    for margin in [1, 2]:
        for ps in [20, 50, 100, 200]:
            for skip in skip_sets[1:3]:
                c = {
                    'hp':  {'strategy':'mm','ema_alpha':0.005,'half_spread':5,
                            'size':40,'take_thresh':5,'take_size':20,'pos_skew':2.0,'skew_scale':2},
                    'vfe': dict(baseline['vfe']),
                    'vev': {'strategy':'pure_mm','iv_prior':0.014,'iv_ema_alpha':0.005,
                            'vfe_mr_alpha':0.01,'edge_take':2.0,'agg_max_size':50,
                            'passive_size':ps,'skip_strikes':skip,'mm_margin':margin},
                }
                configs.append((f'vev_mm_m{margin}_ps{ps}', c))

    # VEV aggressive BS grid
    for edge in [0.5, 1.0, 2.0, 3.0]:
        for agg in [50, 100, 200]:
            for ps in [30, 80, 150]:
                for iv in [0.012, 0.014, 0.016]:
                    c = {
                        'hp':  {'strategy':'mm','ema_alpha':0.005,'half_spread':5,
                                'size':40,'take_thresh':5,'take_size':20,'pos_skew':2.0,'skew_scale':2},
                        'vfe': dict(baseline['vfe']),
                        'vev': {'strategy':'aggressive','iv_prior':iv,'iv_ema_alpha':0.005,
                                'vfe_mr_alpha':0.01,'edge_take':edge,'agg_max_size':agg,
                                'passive_size':ps,'skip_strikes':{4000,4500,6000,6500},
                                'mm_margin':1},
                    }
                    configs.append((f'vev_agg_e{edge}_a{agg}_ps{ps}', c))

    # -- Random search ---------------------------------------------------------
    while len(configs) < n:
        hp_strat = rng.choice(hp_strats)
        c = {
            'hp': {
                'strategy':   hp_strat,
                'ema_alpha':  rng.choice([0.0005,0.001,0.002,0.005,0.01,0.02]),
                'half_spread':rng.choice([2,3,4,5,6,7,8]),
                'size':       rng.choice([5,10,20,30,40,60,80,100,120]),
                'take_thresh':rng.choice([0,2,4,6,8,12,15]),
                'take_size':  rng.choice([10,20,30,50]),
                'pos_skew':   rng.choice([0.0,0.5,1.0,2.0,3.0]),
                'skew_scale': rng.choice([1,2,3]),
            },
            'vfe': {
                'ema_alpha':   rng.choice([0.002,0.003,0.005,0.007,0.01,0.02]),
                'take_thresh': rng.choice([1.0,1.5,2.0,2.5,3.0,4.0,5.0]),
                'half_spread': rng.choice([0.5,1.0,1.5,2.0]),
                'passive_size':rng.choice([10,15,20,30,50]),
                'pos_skew':    rng.choice([0.5,1.0,1.5,2.0]),
            },
            'vev': {
                'strategy':     rng.choice(vev_strats),
                'iv_prior':     rng.choice([0.010,0.012,0.014,0.016,0.018,0.020]),
                'iv_ema_alpha': rng.choice([0.002,0.005,0.008,0.01,0.015,0.02]),
                'vfe_mr_alpha': rng.choice([0.005,0.01,0.015,0.02,0.03]),
                'edge_take':    rng.choice([0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0]),
                'agg_max_size': rng.choice([20,30,50,100,150,200,250]),
                'passive_size': rng.choice([5,10,20,30,50,80,100,150,200]),
                'skip_strikes': rng.choice(skip_sets),
                'mm_margin':    rng.choice([1,2]),
            },
        }
        configs.append((f'rand_{len(configs)}', c))

    return configs


# ------------------------------------------------------------------------------
# MULTIPROCESSING WORKER
# ------------------------------------------------------------------------------

_PDATA = None

def _init_worker(pdata):
    global _PDATA
    _PDATA = pdata

def _run_one(args):
    idx, label, cfg = args
    try:
        total, ppnl = backtest(cfg, _PDATA)
        return idx, label, total, ppnl
    except Exception as e:
        return idx, label, float('-inf'), {}


# ------------------------------------------------------------------------------
# REPORTING
# ------------------------------------------------------------------------------

def print_config(cfg):
    hp  = cfg['hp']
    vfe = cfg['vfe']
    vev = cfg['vev']
    print("  # --- Paste these into Round.4.algo.py -------------------------")
    print(f"  # HP")
    print(f"  HP_STRATEGY     = '{hp['strategy']}'")
    print(f"  HP_EMA_ALPHA    = {hp['ema_alpha']}")
    print(f"  HP_HALF_SPREAD  = {hp['half_spread']}")
    print(f"  HP_SIZE         = {hp['size']}")
    print(f"  HP_TAKE_THRESH  = {hp.get('take_thresh',0)}")
    print(f"  HP_POS_SKEW     = {hp.get('pos_skew',1.0)}")
    print(f"  # VFE")
    print(f"  VFE_EMA_ALPHA   = {vfe['ema_alpha']}")
    print(f"  VFE_TAKE_THRESH = {vfe['take_thresh']}")
    print(f"  VFE_HALF_SPREAD = {vfe['half_spread']}")
    print(f"  VFE_PASSIVE_SZ  = {vfe['passive_size']}")
    print(f"  VFE_POS_SKEW    = {vfe['pos_skew']}")
    print(f"  # VEV Options")
    print(f"  VEV_STRATEGY    = '{vev['strategy']}'")
    print(f"  VEV_IV_PRIOR    = {vev['iv_prior']}")
    print(f"  VEV_IV_EMA_A    = {vev['iv_ema_alpha']}")
    print(f"  VEV_VFE_MR_A    = {vev['vfe_mr_alpha']}")
    print(f"  VEV_EDGE_TAKE   = {vev['edge_take']}")
    print(f"  VEV_AGG_MAX_SZ  = {vev['agg_max_size']}")
    print(f"  VEV_PASSIVE_SZ  = {vev['passive_size']}")
    print(f"  VEV_SKIP_STRIK  = {sorted(vev['skip_strikes'])}")
    print(f"  VEV_MM_MARGIN   = {vev.get('mm_margin',1)}")
    print(f"  TTE_START_DAY1  = 6.0  # Round 4: options expire Day 7, data starts Day 1")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  IMC PROSPERITY -- ROUND 4 STRATEGY OPTIMIZER")
    print("=" * 65)

    t0 = time.time()
    print(f"\n  Loading Round 4 data (downsampled {DOWNSAMPLE}x)...")
    pdata = load_data(ROUND4_ZIP)
    n_ticks = len(pdata['HYDROGEL_PACK'].ts)
    print(f"  {n_ticks} ticks/product × 12 products × 3 days")
    print(f"  Load time: {time.time()-t0:.1f}s")

    print(f"\n  Building configurations...")
    configs = build_configs(N_RANDOM, SEED)
    print(f"  Total configs to test: {len(configs)}")

    print(f"\n  Running on {N_WORKERS} CPU cores...")
    args = [(i, label, cfg) for i, (label, cfg) in enumerate(configs)]

    t1 = time.time()
    results = []
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(pdata,)) as pool:
        done = 0
        for idx, label, total, ppnl in pool.imap_unordered(_run_one, args, chunksize=8):
            results.append((total, ppnl, label, configs[idx][1]))
            done += 1
            if done % 100 == 0 or done == len(configs):
                elapsed = time.time() - t1
                eta     = elapsed / done * (len(configs) - done)
                print(f"    {done:4d}/{len(configs)} done "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining) "
                      f"best so far: {max(r[0] for r in results):,.0f}")

    results.sort(key=lambda x: -x[0])
    print(f"\n  Total run time: {time.time()-t0:.1f}s")

    # -- Find baseline rank ----------------------------------------------------
    baseline_pnl = None
    for total, ppnl, label, cfg in results:
        if label == 'baseline':
            baseline_pnl = total
            break

    # -- Print top 20 ----------------------------------------------------------
    opt_prods = [f"VEV_{k}" for k in VEV_STRIKES]
    print("\n" + "=" * 65)
    print("  TOP 20 CONFIGURATIONS")
    print("=" * 65)
    print(f"  {'Rank':>4}  {'Total':>10}  {'HP':>9}  {'VFE':>8}  {'Options':>10}  "
          f"hp.strat  hp.sz  vev.strat    vev.edge  vev.ps")
    print("  " + "-"*95)

    for rank, (total, ppnl, label, cfg) in enumerate(results[:20], 1):
        opt = sum(ppnl.get(p,0) for p in opt_prods)
        hp  = ppnl.get("HYDROGEL_PACK",0)
        vfe = ppnl.get("VELVETFRUIT_EXTRACT",0)
        hs  = cfg['hp']['strategy']
        hsz = cfg['hp']['size']
        vs  = cfg['vev']['strategy']
        ve  = cfg['vev']['edge_take']
        vps = cfg['vev']['passive_size']
        print(f"  {rank:>4}  {total:>10,.0f}  {hp:>9,.0f}  {vfe:>8,.0f}  {opt:>10,.0f}  "
              f"{hs:8s}  {hsz:>5}  {vs:12s}  {ve:>8}  {vps:>6}")

    if baseline_pnl is not None:
        print(f"\n  Baseline (V.4 equivalent): {baseline_pnl:,.0f}")

    best_total, best_ppnl, best_label, best_cfg = results[0]
    print(f"  Best config:               {best_total:,.0f}"
          f"  (+{best_total - (baseline_pnl or 0):,.0f} vs baseline)")

    print("\n" + "=" * 65)
    print("  BEST CONFIGURATION PARAMETERS")
    print("=" * 65)
    print_config(best_cfg)

    print("\n  Per-product PnL (best config):")
    for prod, pnl in sorted(best_ppnl.items(), key=lambda x: -x[1]):
        bar = "#" * min(40, max(0, int(pnl / max(best_total, 1) * 40)))
        print(f"    {prod:25s} {pnl:>10,.0f}  {bar}")

    print("\n" + "=" * 65)
    print("  NEXT STEPS")
    print("  1. Use the parameters above in your Round.4.algo.py")
    print("  2. Set TTE_START = 4.0 for the final day (day 3) of Round 4")
    print("     Or: track the day in traderData and use (7 - current_day)")
    print("  3. Options expire at end of Day 7 -- gamma/theta spike on Day 3")
    print("=" * 65)


if __name__ == "__main__":
    main()
