# Phenotype analysis — scalping-lay-quality-gate cohort

**Status:** interim (cohort mid-flight, 68/96 agents complete). The held-out
reeval will land at Phase 6 and decide the official verdict.

**Cohort:** `_predictor_SCALPING_layq_1778712871`
**Gate config:**
- `race_confidence_threshold = 0.50`
- `predictor_p_win_back_threshold = 0.20`
- `predictor_p_win_lay_threshold = 0.20` (tightened from predecessor's 0.40)
- `lay_price_max = 20` (new — drops the leverage trap bucket)
- `force_close_before_off_seconds = 0` (training)
- 6 safety genes: stop_loss_pnl_threshold, open_cost,
  matured_arb_bonus_weight, naked_loss_scale, mature_prob_loss_weight,
  fill_prob_loss_weight.

## In-sample headline (3-day training eval, 2026-05-04/05/06)

Across the 68 completed agents:

| Gen | n | mean pnl | mean locked | mean naked | profitable/n |
|---:|---:|---:|---:|---:|---:|
| 0 | 12 | +£214 | +£100 | +£129 | 11/12 |
| 1 | 12 | +£178 | +£107 | +£87 | 10/12 |
| 2 | 12 | +£115 | +£106 | +£25 | 11/12 |
| 3 | 12 | +£154 | +£105 | +£65 | 11/12 |
| 4 | 12 | +£84 | +£96 | +£5 | 11/12 |
| 5 (so far) | 8 | +£141 | +£99 | +£57 | 7/8 |

**Locked floor +£96–107/day vs predecessor's +£85–88/day** — a structural
+£12-20/day improvement from the new gate.

## Structural vs. luck — across the 68-agent cohort

Linear regression of pair outcomes on pair counts across agents:

| Channel | Pearson r | R² | OLS slope | Reading |
|---|---:|---:|---:|---|
| matured_pnl vs n_matured | +0.943 | 0.89 | +£3.30/pair | **Structural** — every pair worth ~£3.30 of spread |
| naked_pnl vs n_naked | +0.327 | 0.11 | +£6.33/pair | Mostly luck. Mean +£1.28/pair, σ £2.69. |

**Conclusion:** the locked channel is bankable per-pair. The naked channel is
+EV in expectation (matches the held-out probe's +£0.098/£ structural EV) but
per-agent noise dominates the per-agent mean.

**Selection consequence:** sort the top-5 for held-out reeval by
`eval_locked_pnl` not `eval_day_pnl`. The day-pnl-ranking surfaces
naked-lucky agents that won't generalise — same trap the predecessor's
findings.md identified.

## Phenotype split: back-first vs lay-first

Bet-log sweep ran every completed agent in deterministic-eval mode against
2026-05-04/05/06 (CPU, default env config — see methodology caveats below).
Aggressive-side mix derived from naked pairs (only the matched leg is the
aggressive one in a naked pair).

| Type | n | % of cohort | Avg locked (3 day) | Avg back pwin | Avg back price | Avg lay price |
|---|---:|---:|---:|---:|---:|---:|
| **Back-first** (`agg_back_pct ≥ 0.5`) | 20 | 34% | **+£275** (~£92/day) | 0.45 | 4.7 | (passive only) |
| **Lay-first** (`agg_back_pct < 0.5`) | 38 | 66% | +£141 (~£47/day) | 0.45 | (passive only) | **7.1** |

Two findings:

1. **The cohort is dominated by lay-first agents** (66%), but the GA is
   evolving toward MORE lay-first over generations:
   `agg_back_pct` per gen: 0 → 0.40, 1 → 0.44, 2 → 0.41, 3 → 0.33,
   4 → 0.11, 5 → 0.25. Late gens are 75-90% lay-first.

2. **Back-first agents have a meaningfully higher locked floor**
   (+£92/day vs +£47/day). The back-favorite-with-passive-lay arb extracts
   more spread per matured pair than the lay-favorite-with-passive-back
   variant. Despite the GA's drift, the back-first cluster is the more
   productive group.

3. **The "lay-first" cluster in THIS cohort lays at average price 7.1, not
   the predecessor's 43.** The new `lay_price_max = 20` cap squeezed the
   lay action into the 2-20 price band, well below the predecessor's
   outsider zone (30-50). What looks like "predecessor-style lay-first"
   is actually a different trader entirely — laying short-priced
   favourites, not long-priced outsiders.

## Phenotype examples

**Pure scalper (locked-dominant; "type 1"):**
```
9b3a2b39   gen 2   locked +£455   naked +£351   back% 60%   total +£805
abdfa0f3   gen 3   locked +£436   naked +£894   back% 100%  total +£1330
2e92886c   gen 0   locked +£416   naked +£958   back% 77%   total +£1372
```
Back-first, mid-favorite (avg price ~4.7), avg pwin 0.45. Same
mechanism as the predecessor's locked-driven generalisers (e.g.
35297cd3 in raceconf findings.md).

**Naked-windfall (high total, mostly variance; "type 2"):**
```
747d3d62   gen 1   locked +£391   naked +£1520   total +£1898
f3336b32   gen 1   locked +£392   naked +£1581   total +£1968
8b2683b2   gen 0   locked +£107   naked +£430    total +£537
```
Same back-first structure as type 1 but the naked component happened
to break their way on these specific days. Held-out will likely revert
their naked toward the cohort mean (+£1.28/pair × ~50 pairs ≈ £64/day,
not £1500/3-day).

**Lay-first short-price (laying favourites; "type 3"):**
```
5e4bb03c   gen 2   locked +£69    naked +£1222   total +£1275   lay_price 6.0  agg_back 36%
b3f4fb2a   gen 2   locked +£302   naked +£830    total +£1131   lay_price 5.7  agg_back 39%
3a91f162   gen 3   locked +£17    naked +£526    total +£533    lay_price 9.0  agg_back 0%
```
The pure-lay-first variant. Locked floor is much thinner (the back
side doesn't fire) so they live or die by naked outcomes. Higher
variance phenotype, lower base rate.

## Vs predecessor (different price regions, complementary)

The predecessor cohort (`scalping-race-confidence-gate`,
+£39.40/day held-out, 3/5 profitable) had ~92% lay-first agents,
laying outsiders at **avg price 43**. Its top-5 had locked floor
~£87/day plus volatile naked component.

This cohort's lay-first agents lay at **avg price 7**. The cap at 20
made the predecessor's price region (>20) inaccessible. So:

- Predecessor harvests "this outsider won't win" predictor edge at high
  prices (rare big wins, rare leveraged losses).
- This cohort harvests "this favourite/second-favourite has 20-50%
  chance" predictor edge at short prices (frequent small wins).

**These are structurally different bets** on different parts of the
predictor's edge surface. Day-by-day correlation between the two
should be low — a candidate for diversified live deployment.

## Methodology caveats (read before reusing this analysis)

1. **Bet-log sweep used DEFAULT env reward overrides for all agents**
   (not the per-agent gene values). The cohort's actual env per agent
   has gene-specific `stop_loss_pnl_threshold`, `naked_loss_scale`,
   `mark_to_market_weight`, etc. Most of these only affect REWARDS, not
   match/gate behaviour, so the policy's *actions* (back-vs-lay decisions,
   runner picks, prices, timing) are faithful.

   But `stop_loss_pnl_threshold` is an ENV gene (changes when env
   force-closes mid-race on per-pair MTM threshold). My sweep had it
   at default (0 = disabled), so **pairs that the cohort's env would
   have stop-closed instead went to naked in my sweep**. This inflates
   `n_naked` and zeros out `n_sc`. The scoreboard's actual sc count
   per agent is 5-20, vs 0 in my sweep.

   Practical consequence: `n_naked` and `naked_pnl` in my sweep are
   noisier upper-bounds than the cohort's real numbers. `n_mat`,
   `matured_pnl`, side-mix, prices, pwins are faithful.

2. **The Phase 2a bet-logging wiring bug** — I passed env as `day`
   parameter to `_build_eval_bet_records`. Fix landed in worker.py
   (commit pending) but the running cohort imported the broken code,
   so its on-disk bet_logs are empty. The sweep tool
   `tools/sweep_bet_capture.py` regenerates them ad-hoc. Future
   cohort runs (post-fix) will write bet_logs natively.

3. **One-day eval per agent×day combo** — each agent ran one
   deterministic rollout per day. Naked-pnl noise is real (σ £2.69/pair
   across the cohort). For ranking purposes the 3-day total smooths it
   somewhat but not entirely. The held-out reeval (Phase 6) is the
   load-bearing measurement.

## Why we capped lay_price_max at 20 (decision audit)

From the Phase 1 probe on the held-out window with the OLD gate
(`pwin_lay = 0.40`, no cap):

| Lay-price bucket | n | Win rate | EV/£ | Avg loss when lost |
|---|---:|---:|---:|---:|
| 2-5 | 77 | 80.5% | +£0.17 | −£3.25 |
| 5-10 | 287 | 86.4% | −£0.03 | −£6.57 |
| 10-20 | 310 | 93.2% | +£0.01 | −£13.62 |
| **20-50** | **308** | **95.5%** | **−£0.39** | **−£29.57** |
| >50 | 191 | 99.0% | +£0.37 | −£59 |

The 20-50 bucket was the dominant bleed (n=308, EV −£0.39/£) —
breakeven at avg price 30 needs 96.7% win rate, actual was 95.5%, a
1.2pp shortfall × £29.57 leverage = the loss. The cap at 20 dropped
this bucket entirely.

**The cap also dropped the >50 bucket** which the probe said was +EV
+£0.37. But that bucket's win rate (99%) is fragile — 1pp of
calibration shift → −£0.7/£. Not safe enough to bank on.

**Alternative we could try in a future plan: a "donut filter"** that
accepts lay in `[2, 20] ∪ [>X]` and rejects only `[20, X]` for some
boundary X. This preserves the predecessor's lay-outsider phenotype
while still dropping the trap. The single-cap choice was the simplest
implementation that cleared the §3 EV threshold; donut filter is
plan-2 material.

## Files in this plan dir

- `probe_2026-04-28_30.txt` — original Phase 1 probe (old gate, EV −£0.035)
- `probe_2026-04-28_30_FULL_GATE.txt` — re-probe with new gate (EV +£0.098)
- `probe_2026-05-04_06_FULL_GATE.txt` — training-window probe (EV +£0.368)
- `smoke_2026-05-04.txt` — original (failing) smoke v1
- `smoke_2026-05-04_v2_PASS.txt` — fixed smoke (PASS)
- `phenotype_analysis.md` — this file
- (Phase 6) `findings.md` — official verdict after held-out reeval

## Files in the cohort dir

- `registry/_predictor_SCALPING_layq_1778712871/scoreboard.jsonl` — one row
  per completed agent (96 at finish)
- `registry/_predictor_SCALPING_layq_1778712871/agents_rollup.csv` —
  scoreboard summary CSV
- `registry/_predictor_SCALPING_layq_1778712871/phenotypes.csv` —
  per-agent phenotype rollup from this sweep
- `registry/_predictor_SCALPING_layq_1778712871/all_bets.parquet` — every
  bet from every agent's sweep capture (175k rows-ish)
- `registry/_predictor_SCALPING_layq_1778712871/bet_logs/adhoc_<agent>/<date>.parquet`
  — per-agent×day parquets (172 files)
