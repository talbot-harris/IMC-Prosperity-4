# IMC-Prosperity-4

Complete description of thought processes and techniques for competing in the IMC Prosperity 4 Challenge.

> **Disclaimer:** This README and the way this repository is organized were put together by AI (Claude). All of the code, strategies and ideas from the competition are my own work.

This repo holds my work from the **IMC Prosperity** algorithmic trading competition (April 2026). Each round, you submit a Python `Trader` class that trades simulated products against bots in an exchange. Below is a short summary of what I built each round.

## Repository layout

```
round1/            Round 1 algorithm + Monte Carlo backtester
round2/            Round 2 submission (re-tuned Round 1 strategy)
round3/            Round 3 algorithm + market visualizer
round4/            Round 4 algorithms (many versions) + analysis/ tools
round5/            Round 5 combined algorithm, per-product research, visualizer
submission_logs/   Official submission logs (code, JSON results, raw logs)
```

## Round summaries

### Round 1 — Pepper Root & Osmium
- **INTARIAN_PEPPER_ROOT:** almost perfectly linear uptrend, so the strategy stays max long, using a fast/slow EMA filter to go flat only during strong downtrends.
- **ASH_COATED_OSMIUM:** mean-reverts around 10,000, so I market-made around that fair value with inventory skew.
- Built a Monte Carlo backtester (`mc_backtester.py`) to tune parameters.
- **Result:** +66,308 (submission 272572)

### Round 2
- Same products. Re-tuned the Round 1 strategy after finding the position limit was 80, not 50.
- **Result:** +92,048 (submission 361412)

### Round 3 — Hydrogel, Velvetfruit & options
- New products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT (VFE) and VEV call options on VFE.
- HP: switched from market-making to passive mean reversion after it lost money.
- VEV options: priced with Black-Scholes using a slow EMA of the underlying and of implied volatility, buying/selling when the market moved away from fair value.
- Wrote `visualize_round3.py` to chart the goods.

### Round 4 — Options deep dive
The hardest round. I went through many versions and learned a lot from the failures:
| Submission | Version | Profit |
|---|---|---|
| 500649 | V.3 combined | −52,172 |
| 500913 | V.4 combined | −3,334 |
| 501054 | VFE-only | +451 |
| 501140 | HP-only | +865 |
| 514293 | Options-only | +423 |
| 523472 | Options-only v2 | +419 |
| 524039 | FINAL unified | +2,264 |

- Found the big early losses came from a time-to-expiry bug and a lagging EMA being fed into the option pricer.
- Split the problem into separate HP-only, VFE-only and options-only algos, then merged them into one final algorithm.
- Final version fits a volatility smile, delta-hedges across products, uses an order-flow imbalance gate on HP, and treats each option strike as its own asset with its own profile.
- Also tracked the bot traders ("Marks") and classified their behavior to use as a signal.
- Analysis tools in `round4/analysis/`: a Monte Carlo backtester, a historical-replay optimizer, an HP signal tester and a fill-by-fill trade analyzer.

### Round 5 — 50 products
- 50 new products across 10 categories (robots, sleep pods, pebbles, microchips, etc.), position limit 10 each.
- Researched each category separately (`round5/products/`) and built an interactive visualizer (`round5_visualizer.html`).
- Final combined algorithm uses pairs trading on strongly correlated products, directional trend-following, and market-making, and turns off the products that consistently lost money.
- Backtest progression: v3 +36k → v4 +43k → v5 +62k → v6 +31k → v7 (final).

## Running the code
The algorithms import `datamodel`, which is provided by the competition. The backtesters can be run directly, e.g.:

```bash
python round4/analysis/mc_backtester_round4.py round4/round.4.algo.FINAL.V3.py
```

Some scripts reference local data files from the competition that are not included here.
