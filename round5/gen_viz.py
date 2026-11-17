import json, csv, collections, statistics

# ── load prices ──────────────────────────────────────────────────────────────
raw = collections.defaultdict(list)
spreads_raw = collections.defaultdict(list)

for day in [2, 3, 4]:
    path = f"C:/Users/talbo/Downloads/ROUND_5_extracted/prices_round_5_day_{day}.csv"
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                mid = float(row["mid_price"])
                ts  = int(row["timestamp"])
                b1  = float(row["bid_price_1"]) if row["bid_price_1"] else None
                a1  = float(row["ask_price_1"]) if row["ask_price_1"] else None
                raw[row["product"]].append({"day": int(row["day"]), "ts": ts, "mid": mid})
                if b1 and a1:
                    spreads_raw[row["product"]].append(a1 - b1)
            except Exception:
                pass

# ── compute stats ────────────────────────────────────────────────────────────
stats = {}
for prod, rows in raw.items():
    prices  = [r["mid"] for r in rows]
    returns = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    tick_vol  = statistics.stdev(returns) if len(returns) > 1 else 0
    avg_spread = statistics.mean(spreads_raw[prod]) if spreads_raw[prod] else 0
    net = prices[-1] - prices[0]
    mean_r = sum(returns) / len(returns) if returns else 0
    if len(returns) > 2:
        cr, nx = returns[:-1], returns[1:]
        cov = sum((cr[i]-mean_r)*(nx[i]-mean_r) for i in range(len(cr)))
        var = sum((r-mean_r)**2 for r in returns)
        ac1 = cov / var if var > 0 else 0
    else:
        ac1 = 0
    stats[prod] = {
        "net": round(net, 1),
        "tick_vol": round(tick_vol, 2),
        "avg_spread": round(avg_spread, 1),
        "ratio": round(avg_spread / tick_vol, 2) if tick_vol > 0 else 0,
        "ac1": round(ac1, 3),
    }

# ── downsample for chart (~500 pts per product) ──────────────────────────────
sampled = {}
for prod, rows in raw.items():
    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    pts = []
    for day in [2, 3, 4]:
        d_rows = by_day[day]
        step = max(1, len(d_rows) // 500)
        pts.extend(d_rows[::step])
    sampled[prod] = [{"x": i, "y": round(r["mid"], 1)} for i, r in enumerate(pts)]

# ── families ─────────────────────────────────────────────────────────────────
families = {
    "SNACKPACK":     ["SNACKPACK_CHOCOLATE","SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY","SNACKPACK_STRAWBERRY","SNACKPACK_VANILLA"],
    "PEBBLES":       ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "MICROCHIP":     ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_RECTANGLE","MICROCHIP_SQUARE","MICROCHIP_TRIANGLE"],
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_FLAMES","GALAXY_SOUNDS_SOLAR_WINDS"],
    "OXYGEN_SHAKE":  ["OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_GARLIC","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_MORNING_BREATH"],
    "UV_VISOR":      ["UV_VISOR_AMBER","UV_VISOR_MAGENTA","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_YELLOW"],
    "SLEEP_POD":     ["SLEEP_POD_COTTON","SLEEP_POD_LAMB_WOOL","SLEEP_POD_NYLON","SLEEP_POD_POLYESTER","SLEEP_POD_SUEDE"],
    "PANEL":         ["PANEL_1X2","PANEL_1X4","PANEL_2X2","PANEL_2X4","PANEL_4X4"],
    "TRANSLATOR":    ["TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_SPACE_GRAY","TRANSLATOR_VOID_BLUE"],
    "ROBOT":         ["ROBOT_DISHES","ROBOT_IRONING","ROBOT_LAUNDRY","ROBOT_MOPPING","ROBOT_VACUUMING"],
}

strategies = {
    "SNACKPACK":     ("Market Make (ratio 2–3x)", "#22c55e"),
    "PEBBLES":       ("Directional: Long XL / Short XS", "#3b82f6"),
    "MICROCHIP":     ("Pairs: Long SQUARE / Short OVAL+RECT", "#8b5cf6"),
    "GALAXY_SOUNDS": ("Market Make + Long BLACK_HOLES", "#06b6d4"),
    "OXYGEN_SHAKE":  ("Market Make + Long GARLIC", "#10b981"),
    "UV_VISOR":      ("Market Make (skip AMBER)", "#f59e0b"),
    "SLEEP_POD":     ("Directional Long (all)", "#ec4899"),
    "PANEL":         ("Directional: Long 2X4", "#6366f1"),
    "TRANSLATOR":    ("Pairs: Long VOID_BLUE / Short SPACE_GRAY", "#14b8a6"),
    "ROBOT":         ("Mean Reversion DISHES + Short IRONING", "#f97316"),
}

COLORS = ["#60a5fa","#f87171","#34d399","#fbbf24","#a78bfa","#fb923c","#38bdf8","#4ade80","#e879f9","#facc15"]

# ── build per-family chart datasets & stats ──────────────────────────────────
family_prefix = {
    "GALAXY_SOUNDS": "GALAXY_SOUNDS_",
    "OXYGEN_SHAKE":  "OXYGEN_SHAKE_",
    "SLEEP_POD":     "SLEEP_POD_",
}

def variant_label(fam, prod):
    prefix = family_prefix.get(fam, fam + "_")
    return prod.replace(prefix, "")

family_charts = {}
for fam, prods in families.items():
    datasets = []
    for i, prod in enumerate(prods):
        datasets.append({
            "label": variant_label(fam, prod),
            "data":  sampled.get(prod, []),
            "borderColor": COLORS[i % len(COLORS)],
            "backgroundColor": "transparent",
            "borderWidth": 1.5,
            "pointRadius": 0,
            "tension": 0.1,
        })
    family_charts[fam] = datasets

family_stats = {}
for fam, prods in families.items():
    rows = []
    for prod in prods:
        s = stats.get(prod, {})
        rows.append({
            "name":   variant_label(fam, prod),
            "net":    s.get("net", 0),
            "spread": s.get("avg_spread", 0),
            "vol":    s.get("tick_vol", 0),
            "ratio":  s.get("ratio", 0),
            "ac1":    s.get("ac1", 0),
        })
    family_stats[fam] = rows

# ── HTML ─────────────────────────────────────────────────────────────────────
fam_list_js   = json.dumps(list(families.keys()))
strat_js      = json.dumps({k: v[0] for k, v in strategies.items()})
strat_col_js  = json.dumps({k: v[1] for k, v in strategies.items()})
charts_js     = json.dumps(family_charts)
stats_js      = json.dumps(family_stats)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IMC Prosperity Round 5 — Visualizer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}}
header{{background:#1e293b;border-bottom:1px solid #334155;padding:14px 24px;display:flex;align-items:center;gap:16px}}
header h1{{font-size:1.15rem;font-weight:700;color:#f8fafc}}
header span{{font-size:0.78rem;color:#94a3b8}}
.layout{{display:flex;height:calc(100vh - 53px)}}
.sidebar{{width:190px;background:#1e293b;border-right:1px solid #334155;overflow-y:auto;flex-shrink:0;padding:6px 0}}
.sb{{display:block;width:100%;text-align:left;padding:9px 14px;background:none;border:none;color:#94a3b8;cursor:pointer;font-size:0.78rem;font-weight:600;transition:all .15s;border-left:3px solid transparent}}
.sb:hover{{background:#273344;color:#e2e8f0}}
.sb.active{{background:#1d3557;color:#60a5fa;border-left-color:#60a5fa}}
.sb-strat{{display:block;font-size:0.63rem;color:#64748b;margin-top:2px;line-height:1.3;font-weight:400}}
.sb.active .sb-strat{{color:#93c5fd}}
.main{{flex:1;overflow-y:auto;padding:22px}}
.panel{{display:none}}.panel.active{{display:block}}
.ph{{margin-bottom:18px}}
.ph h2{{font-size:1.4rem;font-weight:700;color:#f1f5f9}}
.badge{{display:inline-block;margin-top:6px;padding:3px 11px;border-radius:20px;font-size:0.72rem;font-weight:600}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px}}
.card .lbl{{font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}}
.card .val{{font-size:1.35rem;font-weight:700}}
.card .sub{{font-size:.72rem;margin-top:3px}}
.cw{{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:18px;margin-bottom:18px}}
.cw h3{{font-size:.78rem;color:#94a3b8;margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em}}
.cc{{position:relative;height:290px}}
table{{width:100%;border-collapse:collapse;font-size:.8rem}}
th{{text-align:left;padding:7px 11px;color:#64748b;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #334155}}
td{{padding:9px 11px;border-bottom:1px solid #1e293b}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#273344}}
.up{{color:#4ade80}}.down{{color:#f87171}}.neu{{color:#94a3b8}}
.rh{{color:#4ade80;font-weight:700}}.rm{{color:#fbbf24}}.rl{{color:#f87171}}
.hint{{font-size:.68rem;color:#64748b;margin-top:9px;padding-top:9px;border-top:1px solid #334155}}
</style>
</head>
<body>
<header>
  <h1>IMC Prosperity — Round 5 Visualizer</h1>
  <span>3 days &nbsp;·&nbsp; 50 instruments &nbsp;·&nbsp; 10 families</span>
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar"></nav>
  <main class="main" id="main"></main>
</div>
<script>
const FAMILIES = {fam_list_js};
const STRATEGIES = {strat_js};
const STRAT_COL = {strat_col_js};
const FAM_CHARTS = {charts_js};
const FAM_STATS  = {stats_js};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
const charts = {{}};

function switchFamily(fam) {{
  document.querySelectorAll('.sb').forEach(b => b.classList.toggle('active', b.dataset.fam === fam));
  document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'pnl-'+fam));
  if (!charts[fam]) renderChart(fam);
}}

function renderChart(fam) {{
  const canvas = document.getElementById('ch-'+fam);
  if (!canvas) return;
  charts[fam] = new Chart(canvas.getContext('2d'), {{
    type: 'line',
    data: {{ datasets: FAM_CHARTS[fam] || [] }},
    options: {{
      animation: false, responsive: true, maintainAspectRatio: false, parsing: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ position:'top', labels:{{ boxWidth:11, padding:14, font:{{size:10}} }} }},
        tooltip: {{ callbacks: {{
          title: items => 'Tick ' + items[0].parsed.x,
          label: item => item.dataset.label + ': ' + item.parsed.y.toFixed(1)
        }} }}
      }},
      scales: {{
        x: {{ type:'linear', ticks:{{ maxTicksLimit:8 }}, grid:{{ color:'#1e293b' }} }},
        y: {{ grid:{{ color:'#1e293b' }}, ticks:{{ callback: v => v.toLocaleString() }} }}
      }}
    }}
  }});
}}

function rc(r) {{ return r >= 1.5 ? 'rh' : r >= 1.0 ? 'rm' : 'rl'; }}

FAMILIES.forEach((fam, i) => {{
  // sidebar
  const btn = document.createElement('button');
  btn.className = 'sb' + (i===0?' active':'');
  btn.dataset.fam = fam;
  btn.innerHTML = fam.replace(/_/g,' ') + '<span class="sb-strat">' + STRATEGIES[fam] + '</span>';
  btn.onclick = () => switchFamily(fam);
  document.getElementById('sidebar').appendChild(btn);

  // panel
  const div = document.createElement('div');
  div.className = 'panel' + (i===0?' active':'');
  div.id = 'pnl-' + fam;
  const col = STRAT_COL[fam] || '#60a5fa';
  const rows = FAM_STATS[fam] || [];
  const best  = rows.reduce((a,b) => b.net > a.net ? b : a, {{name:'?',net:-Infinity}});
  const worst = rows.reduce((a,b) => b.net < a.net ? b : a, {{name:'?',net:Infinity}});
  const trs = rows.map(r => {{
    const nc = r.net > 500 ? 'up' : r.net < -500 ? 'down' : 'neu';
    const ac = r.ac1 < -0.05 ? 'up' : r.ac1 > 0.05 ? 'down' : 'neu';
    return `<tr>
      <td><strong>${{r.name}}</strong></td>
      <td class="${{nc}}">${{r.net > 0 ? '+' : ''}}${{r.net.toFixed(0)}}</td>
      <td>${{r.spread.toFixed(1)}}</td>
      <td>${{r.vol.toFixed(1)}}</td>
      <td class="${{rc(r.ratio)}}">${{r.ratio.toFixed(2)}}</td>
      <td class="${{ac}}">${{r.ac1 >= 0 ? '+' : ''}}${{r.ac1.toFixed(3)}}</td>
    </tr>`;
  }}).join('');

  div.innerHTML = `
    <div class="ph">
      <h2>${{fam.replace(/_/g,' ')}}</h2>
      <span class="badge" style="background:${{col}}22;color:${{col}};border:1px solid ${{col}}44">${{STRATEGIES[fam]}}</span>
    </div>
    <div class="grid2">
      <div class="card">
        <div class="lbl">Best Variant</div>
        <div class="val up">${{best.name}}</div>
        <div class="sub up">+${{best.net.toFixed(0)}} net trend</div>
      </div>
      <div class="card">
        <div class="lbl">Worst Variant</div>
        <div class="val down">${{worst.name}}</div>
        <div class="sub down">${{worst.net.toFixed(0)}} net trend</div>
      </div>
    </div>
    <div class="cw">
      <h3>Mid Price — Days 2 · 3 · 4</h3>
      <div class="cc"><canvas id="ch-${{fam}}"></canvas></div>
    </div>
    <div class="cw">
      <h3>Variant Statistics</h3>
      <table>
        <thead><tr><th>Variant</th><th>Net Trend</th><th>Avg Spread</th><th>Tick Vol</th><th>MM Ratio</th><th>AC1</th></tr></thead>
        <tbody>${{trs}}</tbody>
      </table>
      <div class="hint">MM Ratio = spread ÷ tick_vol &nbsp;|&nbsp; &gt;1.5 excellent MM &nbsp;|&nbsp; AC1 &lt; −0.05 = mean-reverting &nbsp;|&nbsp; Net Trend = final − initial over all 3 days</div>
    </div>`;
  document.getElementById('main').appendChild(div);
}});

switchFamily(FAMILIES[0]);
</script>
</body>
</html>"""

out = "C:/Users/talbo/OneDrive/Desktop/Prosperity/round5_visualizer.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"Written: {out}  ({len(HTML):,} bytes)")
