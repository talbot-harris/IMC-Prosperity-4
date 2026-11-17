import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import statistics

DATA_DIR = r'C:\Users\talbo\Downloads\ROUND_3_extracted'
OUT_PATH = r'C:\Users\talbo\OneDrive\Desktop\Prosperity\round3_goods_visualizer.png'

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
OPTION_COLORS = plt.cm.plasma(np.linspace(0.05, 0.95, len(STRIKES)))

# ── Load data ──────────────────────────────────────────────────────────────────
def load_series(product, downsample=10):
    """Returns (time_axis, prices) where time_axis is in days (0..3)."""
    times, prices = [], []
    for day in [0, 1, 2]:
        f = os.path.join(DATA_DIR, f'prices_round_3_day_{day}.csv')
        with open(f) as fh:
            reader = csv.DictReader(fh, delimiter=';')
            for row in reader:
                if row['product'] == product:
                    try:
                        t = day + int(row['timestamp']) / 1_000_000
                        prices.append(float(row['mid_price']))
                        times.append(t)
                    except ValueError:
                        pass
    # downsample for plotting speed
    return times[::downsample], prices[::downsample]

print("Loading price data…")
hydro_t, hydro_p = load_series('HYDROGEL_PACK')
vfe_t, vfe_p = load_series('VELVETFRUIT_EXTRACT')
opt_data = {s: load_series(f'VEV_{s}') for s in STRIKES}

# ── Classification heuristics ──────────────────────────────────────────────────
def autocorr_lag1(prices):
    diffs = [prices[i+1] - prices[i] for i in range(len(prices) - 1)]
    if len(diffs) < 2:
        return 0
    m = statistics.mean(diffs)
    v = statistics.variance(diffs)
    if v == 0:
        return 0
    cov = sum((diffs[i] - m) * (diffs[i-1] - m) for i in range(1, len(diffs))) / len(diffs)
    return cov / v

hydro_ac = autocorr_lag1(hydro_p)
vfe_ac = autocorr_lag1(vfe_p)
hydro_cv = statistics.stdev(hydro_p) / statistics.mean(hydro_p) * 100
vfe_cv = statistics.stdev(vfe_p) / statistics.mean(vfe_p) * 100

# Both mean-revert; HYDROGEL is exceptionally stable near a round number → bond/pegged commodity
hydro_label = "Mean-Reverting Commodity\n(Bond-like, pegged ~10 000)"
vfe_label = "Spot Commodity\n(Options Underlying)"

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor='#0d1117')
fig.patch.set_facecolor('#0d1117')

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[1.1, 1.4, 1.1],
    hspace=0.48,
    wspace=0.32,
    left=0.07, right=0.97, top=0.91, bottom=0.07
)

ax_hydro = fig.add_subplot(gs[0, 0])
ax_vfe   = fig.add_subplot(gs[0, 1])
ax_opts  = fig.add_subplot(gs[1, :])
ax_badge_h = fig.add_subplot(gs[2, 0])
ax_badge_v = fig.add_subplot(gs[2, 1])

PANEL_BG = '#161b22'
TEXT_COLOR = '#e6edf3'
GRID_COLOR = '#30363d'
DAY_VLINE_COLOR = '#58a6ff'

def style_ax(ax, title, ylabel, xlabel='Day'):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, color=TEXT_COLOR, fontsize=9)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    # day boundary lines
    for d in [1, 2]:
        ax.axvline(x=d, color=DAY_VLINE_COLOR, linewidth=0.8, linestyle='--', alpha=0.5)

# ── Plot HYDROGEL_PACK ─────────────────────────────────────────────────────────
ax_hydro.plot(hydro_t, hydro_p, color='#39d353', linewidth=0.8, alpha=0.9)
mean_h = statistics.mean(hydro_p)
ax_hydro.axhline(mean_h, color='#f78166', linewidth=1.2, linestyle=':', label=f'Mean {mean_h:.0f}')
ax_hydro.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)
style_ax(ax_hydro, 'HYDROGEL_PACK  —  Mid Price', 'Price (units)')

# ── Plot VELVETFRUIT_EXTRACT ───────────────────────────────────────────────────
ax_vfe.plot(vfe_t, vfe_p, color='#d29922', linewidth=0.8, alpha=0.9)
mean_v = statistics.mean(vfe_p)
ax_vfe.axhline(mean_v, color='#f78166', linewidth=1.2, linestyle=':', label=f'Mean {mean_v:.0f}')
ax_vfe.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR, framealpha=0.8)
style_ax(ax_vfe, 'VELVETFRUIT_EXTRACT  —  Mid Price', 'Price (units)')

# ── Plot VEV Options ───────────────────────────────────────────────────────────
# Shade the post-day-2 expiry region
ax_opts.axvspan(2 + 999900/1e6, 7, color='#58a6ff', alpha=0.06)

for i, strike in enumerate(STRIKES):
    t, p = opt_data[strike]
    label = f'K={strike:,}'
    lw = 1.6 if strike in (5000, 5100, 5200) else 1.0
    ax_opts.plot(t, p, color=OPTION_COLORS[i], linewidth=lw, label=label, alpha=0.92)

# Mark expiry at day 7
ax_opts.axvline(x=7, color='#ff7b72', linewidth=1.8, linestyle='-', alpha=0.9, label='Expiry (Day 7)')

style_ax(ax_opts, 'VEV Options (VELVETFRUIT_EXTRACT Calls)  —  Mid Price by Strike', 'Option Price (units)')
ax_opts.set_xlim(-0.05, 7.3)

legend = ax_opts.legend(
    ncol=6, fontsize=8.5, facecolor=PANEL_BG, labelcolor=TEXT_COLOR,
    framealpha=0.85, loc='upper right'
)

# ── Classification badge panels ────────────────────────────────────────────────
for ax in [ax_badge_h, ax_badge_v]:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.set_xticks([])
    ax.set_yticks([])

def stat_block(ax, title, color, label, bullets):
    ax.text(0.5, 0.92, title, ha='center', va='top', transform=ax.transAxes,
            color=color, fontsize=13, fontweight='bold')
    ax.text(0.5, 0.73, label, ha='center', va='top', transform=ax.transAxes,
            color=TEXT_COLOR, fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=color, linewidth=1.5))
    for j, b in enumerate(bullets):
        ax.text(0.08, 0.52 - j * 0.17, b, ha='left', va='top', transform=ax.transAxes,
                color=TEXT_COLOR, fontsize=9)

stat_block(
    ax_badge_h,
    'HYDROGEL_PACK',
    '#39d353',
    hydro_label,
    [f'● Mean price:  {statistics.mean(hydro_p):,.1f}',
     f'● Std dev:      {statistics.stdev(hydro_p):.1f}   (CV {hydro_cv:.3f}%)',
     f'● Autocorr Δp: {hydro_ac:.4f}  (mean-reverting)',
     f'● Range [days 0–2]: {min(hydro_p):.0f} – {max(hydro_p):.0f}']
)

stat_block(
    ax_badge_v,
    'VELVETFRUIT_EXTRACT',
    '#d29922',
    vfe_label,
    [f'● Mean price:  {statistics.mean(vfe_p):,.1f}',
     f'● Std dev:      {statistics.stdev(vfe_p):.1f}   (CV {vfe_cv:.3f}%)',
     f'● Autocorr Δp: {vfe_ac:.4f}  (mean-reverting)',
     f'● Range [days 0–2]: {min(vfe_p):.0f} – {max(vfe_p):.0f}']
)

# ── Overall title ──────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.965,
    'IMC Prosperity  ·  Round 3  —  New Tradeable Goods Price Visualizer',
    ha='center', va='top', color=TEXT_COLOR, fontsize=15, fontweight='bold'
)
fig.text(
    0.5, 0.943,
    'HYDROGEL_PACK  ·  VELVETFRUIT_EXTRACT  ·  VEV Options (K=4000–6500, matures Day 7)',
    ha='center', va='top', color='#8b949e', fontsize=10
)

print(f"Saving -> {OUT_PATH}")
fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Done.")
