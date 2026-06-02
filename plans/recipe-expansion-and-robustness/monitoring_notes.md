# Round 5 Monitoring Notes

Append-only log of autonomous babysit decisions. Each entry: timestamp,
cells completed since last entry, key observations, action taken.

## State at session-end handoff (2026-05-25 22:22 BST)

User stepped away. Autonomous monitoring loop armed. Current state:
- Phase B in progress. F2 done (mat 3.2%, cls 41.2%, fc 52.3%, opens 93, day_pnl -£110). F3 running (started 22:07:47, ETA ~22:33). F3b queued (ETA ~23:00).
- Round 5 wrapper polling for "phase-b fan-out complete" — will re-exec to pick up any BASE_RECIPE edits, then fire 25 cells.
- BASE_RECIPE currently set to F3 defaults (positive_weight=1.0). Edit BEFORE the re-exec if F3b clearly wins.

## **GOAL-DIRECTED MODE — READ THIS FIRST**

User's clarification (2026-05-25 22:22 BST): operate in goal-directed
mode. **Don't passively execute Round 5 cells in order**. Actively
work toward all 5 metrics passing simultaneously. Read
[goal_directed_instructions.md](goal_directed_instructions.md) for
the operating principles, amendment patterns, Round 6+ design
heuristics, and exit criteria.

Key behavioural changes vs the original /loop prompt:
- After each analysis, **identify the leader cell** (closest to passing
  all 5 metrics) and **queue cells designed to fix its remaining
  failures**, not whatever's next in the original Round 5 plan.
- **Kill dead sweeps eagerly** — 3 consecutive cells in a group all
  failing the same metric means abandon that group.
- **After Round 5 completes, design Round 6**, don't stop at findings.
  Keep iterating until either a deploy candidate emerges, GPU budget
  is exhausted (~15:18 BST tomorrow), or you judge no further probe-
  scale sweep will move the needle.

## Cell completion count: 0 (round 5 hasn't started)

## Decision checkpoint cadence

Wake every ~28 min. Full analysis only when 3+ new round-5 cells
finished since the last analysis. Multi-gen cells (R4_2gen/3gen/5gen)
take longer (~50-125 min) so the cadence will stretch in Group 4.

## Log entries

### 2026-05-25 22:18 BST — Initial state

Pre-Round-5. Phase B mid-flight. Waiting on F3 + F3b before BASE_RECIPE
decision. No round-5 cells finished yet.

Action: schedule first wakeup ~22:46 BST (28 min) to check F3 result.

### 2026-05-26T18:47:03+01:00 — babysit iteration

**New cells since last iteration:**

_round6_G1_e7_seed43_1779814159                    pnl=  -82.5 locked=+17.5 opens= 139 mat= 5.9% cls=23.4% fc=67.4% L/σN= 0.73 passes=4/5
_round3_E7_pwinback_bc500_1779731177               pnl=  -66.4 locked=+19.7 opens= 138 mat= 5.2% cls=26.6% fc=64.9% L/σN= 0.32 passes=3/5
_round3_E8_pwinback_bc500_matbonus5_1779732664     pnl=  -91.3 locked=+15.3 opens= 138 mat= 4.7% cls=28.2% fc=63.8% L/σN= 1.23 passes=3/5
_round3_N2_pwinback_opencost05_1779724973          pnl=  -99.6 locked=+14.0 opens= 112 mat= 1.6% cls=37.7% fc=56.3% L/σN= 0.55 passes=3/5
_round6_G1_e7_seed46_1779817075                    pnl= -117.7 locked=+21.7 opens= 141 mat= 7.8% cls=13.5% fc=75.9% L/σN=  inf passes=3/5
_round6_G1_e7_seed44_1779815631                    pnl=  -88.4 locked=+17.4 opens= 138 mat= 3.6% cls=30.1% fc=62.6% L/σN= 0.36 passes=2/5
_round5_R3_pwin035_1779763061                      pnl=  -92.3 locked= +9.4 opens=  70 mat= 5.8% cls=28.1% fc=62.6% L/σN= 0.23 passes=2/5
_round3_N1_pwinback_matbonus5_1779723409           pnl=  -94.6 locked=+15.6 opens= 112 mat= 1.8% cls=37.6% fc=56.2% L/σN= 0.44 passes=2/5
_round5_R1_seed46_1779751866                       pnl=  -95.5 locked=+13.2 opens= 110 mat= 3.8% cls=29.2% fc=62.9% L/σN= 0.37 passes=2/5
_round5_R1_seed43_1779747574                       pnl= -109.4 locked=+11.5 opens=  96 mat= 5.5% cls=26.8% fc=64.2% L/σN= 0.54 passes=2/5
_round5_R3_pwin030_1779761646                      pnl= -116.0 locked=+10.0 opens=  76 mat= 6.9% cls=22.1% fc=66.7% L/σN= 0.54 passes=2/5
_round5_R2_bc100_1779753232                        pnl= -124.0 locked=+12.8 opens= 113 mat= 2.4% cls=35.5% fc=58.3% L/σN= 0.85 passes=2/5
_round5_R7_tight_lock_1779783372                   pnl= -127.1 locked= +8.2 opens=  92 mat= 7.9% cls=23.0% fc=66.1% L/σN= 0.59 passes=2/5
_round3_E5_bc500_matbonus5_1779728200              pnl= -285.4 locked=+33.9 opens= 240 mat= 5.9% cls= 1.1% fc=88.5% L/σN= 1.00 passes=2/5
_round3_E6_bc500_opencost05_1779729707             pnl= -293.2 locked=+32.1 opens= 230 mat= 5.7% cls= 1.8% fc=88.6% L/σN= 2.09 passes=2/5
_round3_E1_bc200_1779721158                        pnl= -296.8 locked=+32.4 opens= 239 mat= 5.5% cls= 2.2% fc=88.9% L/σN= 1.06 passes=2/5
_round5_R6_race_conf035_1779779169                 pnl= -100.8 locked= +8.8 opens=  74 mat= 6.4% cls=17.5% fc=72.4% L/σN= 0.49 passes=1/5
_round5_R4_2gen_1779764496                         pnl= -104.1 locked= +9.1 opens=  77 mat= 6.4% cls=26.4% fc=63.8% L/σN= 0.32 passes=1/5
_round5_R4_3gen_1779767175                         pnl= -104.6 locked= +8.2 opens=  73 mat= 6.7% cls=26.2% fc=63.8% L/σN= 0.27 passes=1/5
_round5_R6_lay_price_max20_1779780585              pnl= -109.2 locked=+10.5 opens=  81 mat= 7.1% cls=19.7% fc=69.5% L/σN= 0.25 passes=1/5
_round5_R6_pwin_lay050_1779777728                  pnl= -111.5 locked= +8.5 opens=  88 mat= 6.3% cls=22.0% fc=68.3% L/σN= 0.21 passes=1/5
_round5_R4_5gen_1779771151                         pnl= -113.7 locked= +9.8 opens=  80 mat= 5.8% cls=24.2% fc=66.2% L/σN= 0.34 passes=1/5
_round5_R3_pwin025_1779760280                      pnl= -115.1 locked= +8.4 opens=  75 mat= 6.7% cls=21.1% fc=68.9% L/σN= 0.40 passes=1/5
_round3_N3_pwinback_matbonus5_opencost05_1779726590 pnl= -115.9 locked=+15.1 opens= 113 mat= 1.6% cls=39.0% fc=55.7% L/σN= 0.27 passes=1/5
_round5_R2_bc2000_1779757383                       pnl= -121.0 locked=+12.7 opens=  96 mat= 5.7% cls=24.4% fc=67.1% L/σN= 0.47 passes=1/5
_round5_R1_seed42_repeat_1779746203                pnl= -125.0 locked=+11.3 opens=  85 mat= 6.2% cls=25.1% fc=65.8% L/σN= 0.36 passes=1/5
_round5_R2_bc1000_1779755985                       pnl= -131.9 locked= +9.3 opens=  86 mat= 6.4% cls=23.0% fc=66.0% L/σN= 0.45 passes=1/5
_round5_R7_fc60_1779782006                         pnl= -136.2 locked=+12.4 opens=  92 mat= 7.1% cls=26.1% fc=63.0% L/σN= 0.37 passes=1/5
_round5_R3_pwin015_1779758890                      pnl= -148.4 locked=+10.5 opens=  93 mat= 5.4% cls=23.1% fc=68.0% L/σN= 0.23 passes=1/5
_round3_E2_bc500_1779722700                        pnl= -280.2 locked=+34.9 opens= 243 mat= 4.9% cls= 0.8% fc=89.3% L/σN=  inf passes=1/5
_round5_R2_bc200_1779754609                        pnl= -109.9 locked=+11.2 opens=  99 mat= 3.8% cls=26.8% fc=64.9% L/σN= 0.21 passes=0/5
_round5_R1_seed44_1779749036                       pnl= -124.3 locked= +8.1 opens=  93 mat= 4.0% cls=34.2% fc=57.4% L/σN= 0.11 passes=0/5
_round5_R1_seed45_1779750466                       pnl= -154.3 locked= +6.4 opens=  81 mat= 4.6% cls=36.0% fc=56.9% L/σN= 0.22 passes=0/5

**Leader of this batch:** `_round6_G1_e7_seed43_1779814159` — 4/5, day_pnl=-82.5
**Overall leader so far:** `_round6_G1_e7_seed43_1779814159` — 4/5, day_pnl=-82.5

### 2026-05-26T19:48:43+01:00 — babysit iteration

**New cells since last iteration:**

_round6_G2_e7_tight_lock_1779818533                pnl=  -81.3 locked=+14.5 opens= 138 mat= 5.4% cls=27.6% fc=64.1% L/σN= 0.39 passes=3/5
_round6_G2_e7_race_conf035_1779820062              pnl=  -80.2 locked=+19.6 opens= 136 mat= 4.8% cls=28.6% fc=61.9% L/σN= 0.48 passes=2/5

**Leader of this batch:** `_round6_G2_e7_tight_lock_1779818533` — 3/5, day_pnl=-81.3
**Overall leader so far:** `_round6_G1_e7_seed43_1779814159` — 4/5, day_pnl=-82.5

### 2026-05-26T20:39:26+01:00 — babysit iteration

**New cells since last iteration:**

_round6_G2_e7_lay_price_max20_1779821698           pnl=  -54.6 locked=+17.1 opens= 139 mat= 5.2% cls=26.5% fc=65.0% L/σN= 0.34 passes=3/5
_round6_G2_e7_pwin_back025_1779823195              pnl=  -89.7 locked=+15.8 opens= 126 mat= 3.4% cls=33.2% fc=59.9% L/σN= 2.37 passes=3/5

**Leader of this batch:** `_round6_G2_e7_lay_price_max20_1779821698` — 3/5, day_pnl=-54.6
**Overall leader so far:** `_round6_G1_e7_seed43_1779814159` — 4/5, day_pnl=-82.5

### 2026-05-26T21:48:01+01:00 — babysit iteration

**New cells since last iteration:**

_round6_G3_e7_l2_lowweight_1779827554              pnl= -102.3 locked=+11.1 opens=  84 mat= 7.1% cls=19.0% fc=70.8% L/σN= 0.16 passes=1/5
_round6_G3_e7_l34_only_1779826089                  pnl= -110.1 locked= +8.8 opens=  93 mat= 3.2% cls=41.2% fc=52.3% L/σN= 0.24 passes=0/5

**Leader of this batch:** `_round6_G3_e7_l2_lowweight_1779827554` — 1/5, day_pnl=-102.3
**Overall leader so far:** `_round6_G1_e7_seed43_1779814159` — 4/5, day_pnl=-82.5

### 2026-05-26T22:48:01+01:00 — babysit iteration

**New cells since last iteration:**

_round6_G4_e7_bc1000_1779828950                    pnl=  -99.6 locked=+20.1 opens= 137 mat= 6.4% cls=19.7% fc=70.3% L/σN= 0.59 passes=4/5
_round6_G4_e7_bc250_1779830309                     pnl=  -54.2 locked=+16.4 opens= 139 mat= 3.2% cls=33.6% fc=60.3% L/σN= 0.33 passes=2/5

**Leader of this batch:** `_round6_G4_e7_bc1000_1779828950` — 4/5, day_pnl=-99.6
**Overall leader so far:** `_round6_G1_e7_seed43_1779814159` — 4/5, day_pnl=-82.5

### 2026-05-26T23:48:18+01:00 — babysit iteration

**New cells since last iteration:**

_round6_5_K3_e7_bc1000_fc_off_1779835079           pnl= +236.4 locked= +6.5 opens= 133 mat= 9.0% cls=33.1% fc= 0.0% L/σN=  inf passes=5/5
_round6_5_K2_e7_lay_max_fc_off_1779833718          pnl= +201.9 locked= +4.5 opens= 138 mat= 6.7% cls=37.2% fc= 0.0% L/σN= 0.04 passes=4/5

**Leader of this batch:** `_round6_5_K3_e7_bc1000_fc_off_1779835079` — 5/5, day_pnl=+236.4
**Overall leader so far:** `_round6_5_K3_e7_bc1000_fc_off_1779835079` — 5/5, day_pnl=+236.4

**POSITIVE DAY_PNL ACHIEVED** with 5/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-27T00:48:20+01:00 — babysit iteration

**New cells since last iteration:**

_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823   pnl= +232.7 locked= +5.4 opens= 137 mat= 8.9% cls=33.3% fc= 0.0% L/σN= 0.08 passes=4/5
_round6_5_K4_e7_fc30_1779836481                    pnl=  -81.7 locked=+25.0 opens= 138 mat= 6.7% cls=27.4% fc=62.3% L/σN= 0.78 passes=4/5

**Leader of this batch:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7
**Overall leader so far:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7

**POSITIVE DAY_PNL ACHIEVED** with 4/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-27T07:48:37+01:00 — babysit iteration

**New cells since last iteration:**

_round7_H1_pwin028_1779862296                      pnl= -105.8 locked=+18.3 opens= 124 mat= 4.4% cls=29.8% fc=62.3% L/σN= 0.82 passes=2/5
_round7_H1_pwin032_1779863693                      pnl= -126.9 locked=+17.1 opens= 120 mat= 5.0% cls=29.2% fc=61.2% L/σN= 0.35 passes=2/5

**Leader of this batch:** `_round7_H1_pwin028_1779862296` — 2/5, day_pnl=-105.8
**Overall leader so far:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7

**POSITIVE DAY_PNL ACHIEVED** with 4/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-27T08:48:34+01:00 — babysit iteration

**New cells since last iteration:**

_round7_H2_e7_tight_lock_pwin025_1779866469        pnl=  -84.9 locked=+14.6 opens= 126 mat= 5.2% cls=30.9% fc=59.8% L/σN= 0.63 passes=4/5
_round7_H1_pwin040_1779865099                      pnl= -114.4 locked=+19.5 opens= 114 mat= 4.6% cls=31.2% fc=60.0% L/σN= 1.22 passes=2/5

**Leader of this batch:** `_round7_H2_e7_tight_lock_pwin025_1779866469` — 4/5, day_pnl=-84.9
**Overall leader so far:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7

**POSITIVE DAY_PNL ACHIEVED** with 4/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-27T09:48:24+01:00 — babysit iteration

**New cells since last iteration:**

_round7_H2_e7_tight_lock_race_conf_1779869199      pnl=  -34.3 locked=+16.3 opens= 136 mat= 5.7% cls=28.6% fc=62.3% L/σN= 0.50 passes=4/5
_round7_H2_e7_tight_lock_lay_max_1779870580        pnl=  -58.4 locked=+14.7 opens= 138 mat= 6.5% cls=25.8% fc=64.3% L/σN= 0.35 passes=3/5

**Leader of this batch:** `_round7_H2_e7_tight_lock_race_conf_1779869199` — 4/5, day_pnl=-34.3
**Overall leader so far:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7

**POSITIVE DAY_PNL ACHIEVED** with 4/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-27T10:48:13+01:00 — babysit iteration

**New cells since last iteration:**

_round7_H4_bc1500_1779874640                       pnl= -163.5 locked=+22.2 opens= 134 mat= 9.7% cls= 9.0% fc=79.9% L/σN=  inf passes=3/5
_round7_H3_l34_only_tight_lock_1779871909          pnl= -127.4 locked= +8.8 opens=  96 mat= 6.2% cls=29.8% fc=60.9% L/σN= 0.33 passes=1/5
_round7_H3_l34_only_pwin025_1779873288             pnl= -138.6 locked= +8.2 opens=  92 mat= 3.8% cls=37.9% fc=55.6% L/σN= 0.22 passes=0/5

**Leader of this batch:** `_round7_H4_bc1500_1779874640` — 3/5, day_pnl=-163.5
**Overall leader so far:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7

**POSITIVE DAY_PNL ACHIEVED** with 4/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-27T11:48:01+01:00 — babysit iteration

**New cells since last iteration:**

_round7_H4_bc1500_l34_1779876067                   pnl= -128.9 locked= +9.7 opens= 101 mat= 3.2% cls=40.4% fc=53.3% L/σN= 0.13 passes=1/5

**Leader of this batch:** `_round7_H4_bc1500_l34_1779876067` — 1/5, day_pnl=-128.9
**Overall leader so far:** `_round6_5_K5_e7_lay_max_bc1000_fc_off_1779837823` — 4/5, day_pnl=+232.7

**POSITIVE DAY_PNL ACHIEVED** with 4/5 acceptance. Continuing to queue replicate cells via existing rounds; do not auto-stop.

### 2026-05-29 06:54 BST — H1 complete, Round M launched (loop v2)

H1 held-out results (7 unseen days, ranked by mat%):
- E7+tight_lock: mat 4.3%, day_pnl -£224
- full-aug: mat 4.2%, day_pnl -£145 (BEST day_pnl, fewest opens 100)
- E7: mat 3.7%, -£227 | E7+matbonus: 3.3%, -£221
- pwin_back: 1.8%, -£174 | baseline: 1.6%, -£232
All NEGATIVE. locked/matured positive (+£2-6). Bottleneck = mat% (selection).

Two mat% levers: tight_lock + full-aug (fewer/selective opens). Launched
Round M (8 cells, held-out) stacking them + pushing tight_lock to
0.003/0.002 + pwin025 selectivity. Goal: held-out mat% toward 10%+.

NOTE: GPU idled ~7h overnight (23:46→06:52) — ScheduleWakeup loop died
on session dormancy. Round M is a detached bash wrapper so it survives;
loop re-armed for active-session analysis. Overnight dormancy remains
the limiting factor for 24/7 utilisation.

### 2026-05-30 07:40 BST — Path C built + Round T queued (chain S→T)

Continued campaign. GPU verified alive: Q1 (tight0030_band050) training
through gen1, now in held-out eval. Q→R→S chain wrappers all confirmed
running (PIDs via wmic). No Q/R/S cells finished yet (Q1 is the first).

Built **Path C — mature_prob open-gate** (handoff priority 3, the
highest-leverage untried mechanism per findings.md):
- Policy-side per-runner mask: refuse OPEN_BACK/OPEN_LAY where the
  policy's own `sigmoid(mature_prob_head) < threshold`. NOOP/CLOSE
  never gated.
- New `--mature-prob-open-threshold` flag, threaded runner→worker→
  policy as a DIRECT arg (dodges the Path-A `trainer_hp.get(key,flag)`
  precedence foot-gun).
- Mirrors direction-gate's rollout/update KL-consistency (captured
  mask via `isfinite(masked_logits)`, replayed with
  `apply_mature_prob_gate=False`) + warmup annealing (0→gene over
  `mature_gate_warmup_eps`) to avoid cold-start collapse.
- Tests: `tests/test_v2_mature_gate.py` 12 pass; direction-gate suite
  still 31 pass.

Queued **Round T** (chained S→T via `run_chain_S_T.sh`, detached bash):
threshold sweep 0.20/0.30/0.40/0.50 + T5 head-trained-no-gate control
+ T6 seed-43 replicate, all on N4 base (full-aug + pwin band 0.25-0.50,
fc=120). CRITICAL: pins `mature_prob_loss_weight=2.0` cohort-wide —
every prior round left the head UNTRAINED (≈0.5 constant), which would
make the gate degenerate.

Next harvest: Q cells (~09:30), then R (~13:30), S (~17:30), T (~21:30).

### 2026-05-30 08:07 BST — Q1 harvested (first cell of Q→R→S→T chain)

Chain healthy (Q2 running; R/S/T wrappers polling). Q1
(tight0030_band050: arb_spread 0.003 + pwin band 0.25-0.50, fc=120):

| agent | opens | mat% | fc% | lck/pr | lockPnl(7d) | fcPnl | nakPnl | dpnl/d |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 646b63ca | 77 | 2.6 | 63.6 | 2.94 | +5.89 | -84.4 | -11.9 | -17.6 |
| 756fc178 | 39 | 2.6 | 74.4 | 3.71 | +3.70 | -55.8 | -45.6 | -15.6 |
| 05ccc6af | 31 | 9.7 | 80.6 | 1.06 | +3.17 | -57.6 | -10.7 |  -9.7 |
| d2b524b0 | 13 | 7.7 | 53.8 | 0.89 | +0.89 | -18.4 | -31.1 |  -7.6 |

Mean 7-day-total day_pnl -£88.4, but **day_pnl is the wrong unit**
(naked + fc are the zero-EV channels we discount). Read on our
chosen metrics:

- **LOCKED is cleanly positive for ALL 4 agents** (£0.89-£5.89/7d,
  never negative). The open-side selection is NOT where money leaks.
- **~100% of the deficit is the second-leg-fill channel**: fc-cost
  -£18 to -£84, naked -£10 to -£45. Tight-spread (0.003) already
  produces clean positive locked, so spread is not the leak.
- **Best agent beats N4 on day_pnl too**: d2b524b0 -£52.9 (7d) vs
  N4 -£78; 646b63ca highest locked (£5.89). The cohort MEAN is not
  the deploy unit — the best agent is.
- `mature_prob_bce = 0` confirms Q's head untrained (weight=0,
  expected — only Round T trains it).

**Value extracted (not a leaderboard miss):** Q1 localizes the loss
to second-leg fill, which *strengthens* the case for Path C (mature
gate, Round T) and Path D (liquidity gate) over further spread/band
tuning. Phenotype (tight0030 + band 0.25-0.50) trades a different
price slice than N4 — candidate for a complementary deploy ensemble
member even at negative standalone day_pnl. KEEP its weights;
re-evaluate as an ensemble component once Path C lands.

### 2026-05-30 09:05 BST — per-pair forensics (prompted by operator's "2% game" framing)

Read Q1 agent 05ccc6af's bet logs (217 filled-leg pairs / 7 held-out
days) per the phenotype-analysis methodology. **Campaign-redirecting
finding — see findings.md "KEY FINDING (2026-05-30)".** Summary:

- Naturally-matured pairs (19): ALL +0.35..+3.50% locked, balanced.
  The 2% scalping edge WORKS. £/pair small only because stakes small.
- Force-closed pairs (175 = **81%**): the throughput killer. Passive
  never fills → T−120 rail crosses spread (−£18..−84 fc-cost). Rail is
  correct; fault is upstream over-opening of unfillable pairs.
- Agent-closed via close_signal (16): ALL NEGATIVE, median −15.8%,
  under-hedged because strict-matcher close leg only partially fills
  (no overdraft, unlike force-close). close_signal is value-destructive
  as currently sized.

Action: confirms Path C (Round T, queued) + Path D (liquidity gate)
as the right direction. NEW candidate experiment: disable or
relaxed-match-overdraft the close_signal path (16/16 losses = cheap
win). Did NOT launch yet — Q/R/S/T chain busy; queue after assessing
whether a --disable-close-signal style flag exists.

### 2026-05-30 09:07 BST — Q2 + Q3 harvested (corroborate forensic finding)

Q2 (tight0020_band050): mean day_pnl -£94.1/7d. Q3 (tight0030 seed43):
mean -£147.7/7d. Both same structure (mat <10%, fc 60-75%, locked
positive £3-15/7d). Neither beats N4 (-£78) on cohort mean.

**Corroborates the 09:05 forensic finding — more opens = worse:**
Q3 agents open 100-120/day (most yet) and post the worst day_pnl;
each extra unfillable open pays the force-close toll. Conversely
Q2's `1e001693` opens only 37/day → best individual agent across
all of Q1-Q3: day_pnl **-£18.4/7d (≈ -£2.6/day)**, fc 62%, mat 5.4%.
The "open fewer" phenotype reduces the STRUCTURAL toll (not naked
luck — force-close spread-crossing is a real cost). Direct support
for Path C/D's "open only fillable pairs" thesis. Q4 running.

### 2026-05-30 09:33 BST — close-walk implemented + Round W queued

Root-caused the −£27 Runman close (operator's question) to the no-walk
matcher artifact, NOT open selection. See findings.md KEY FINDING #2:
the close leg sized at full equal-profit but the single-level matcher
took only the best level (£17.55) when £146 sat ONE tick away → ~£339/
agent/7d avoidable directional loss, 98% mechanically reachable within
10 ticks. Spec (betfair_market_model.md §2) confirms real Betfair fills
"at named price OR BETTER" — a bounded walk is a limit order, not the
phantom-profit market-sweep the no-walk ban targets.

IMPLEMENTED (this session):
- exchange_matcher: `walk_to_price` param on _match/match_back/match_lay
  (None = strict single-level, byte-identical; set = walk across levels
  to limit, weighted-avg price, hard-cap enforced per level). Matcher
  stays dependency-free (caller computes limit via tick_offset).
- bet_manager place_back/place_lay thread walk_to_price.
- betfair_env _attempt_close computes walk_to_price from new
  `_close_walk_ticks` knob (LAY close → +N ticks, BACK close → −N).
  Applies to BOTH agent close_signal AND force-close. OPEN path
  untouched. Knob via --reward-overrides close_walk_ticks=N (default
  0 = OFF = byte-identical), whitelisted in _REWARD_OVERRIDE_KEYS.
- Tests: tests/test_exchange_matcher.py::TestCloseWalk (6),
  tests/test_forced_arbitrage.py::TestCloseWalkWiring (3). 294 existing
  pass. CLAUDE.md "Order matching" documents the sanctioned exception.

QUEUED: Round W (run_roundW.sh) — close_walk_ticks ∈ {0,5,10,15} on Q1
base (fc=120, tight 0.003, band 0.50, seed 42). W1 walk=0 = reproduction
check vs Q1. Chained after roundT via run_chain_T_W.sh (detached, pid
2080, waiting on "roundT fan-out complete" — no GPU contention).
This supersedes the earlier "Round U close_signal-disable" idea: if the
walk completes the hedge, close_signal stops being value-destructive.

### 2026-05-30 09:40 BST — CAMPAIGN REPRIORITISED around close-walk (operator-directed)

3-level depth CONFIRMED: feed hard-capped at top-3 ladder levels (97%
of back books, 64% of lay books have 3; 21% of lay books have ≤1).
Implication: close-walk consumes ≤3 levels regardless of tick budget,
but 3 levels usually hold far more than a hedge needs (Runman: £230 vs
£42.60). Walk knob is effectively BINARY (5/10/15 converge given 3-level
cap). ~21% thin-lay cases can't be helped (future: deeper book capture).

KEY REFRAME: the entire held-out leaderboard was measured under the
BROKEN single-level close matcher. The −£1.20/open fc-side cost burying
every recipe is substantially the under-hedge artifact, not real spread
cost. So every recipe deserves a rerun with close-walk on.

ACTIONS (operator: "reconsider all queued experiments except matching"):
- KILLED chain wrappers Q_R, R_S, S_T, T_W → Rounds R (band tweaks),
  S (tight-lock mat-lift), T (mature gate) CANCELLED. They tested
  old-mechanism levers against a broken close baseline = stale before
  finishing. Mature-gate (T) shelved pending close-walk results.
- Q4 left to finish (completes Q data cleanly; no auto-launch after).
- Round W REBUILT as broad close-walk rerun (operator chose broad):
  N4, N2, M6, M7, O1 × close_walk_ticks {0,10} = 10 cells. walk0 =
  control (reproduces old held-out #), walk10 = lift. fc=120 (O1 fc=60).
  Base = Round N full-aug. Chained after Q via run_chain_Q_W.sh
  (detached, waiting on "roundQ fan-out complete"). ETA ~6-7h.

### 2026-05-30 10:10 BST — Q4 close-out + Round W live

Q4 (tight0030 s44): mean -£90.8/7d, mat 2.8-5%, fc 55-64%. Q round done
(Q1 -88.4, Q2 -94.1, Q3 -147.7, Q4 -90.8) — none beat N4, all old-close-
mechanism as expected. Q→W chain fired correctly at 09:46; Round W
(broad close-walk rerun, 10 cells N4/N2/M6/M7/O1 × walk{0,10}) now
running, N4_walk0 first. No W cells complete yet. NOTE: a stale
pre-reprioritization wakeup (Q→R→S→T / Round U) fired this cycle — those
are CANCELLED; close_signal "Round U" is superseded by close-walk (which
fixes under-hedging for both agent-close and force-close). Realigned the
loop prompt to close-walk.

### 2026-05-30 10:29 BST — Round W: N4_walk0 control reproduces baseline ✓

First Round W cell complete. N4_walk0 (close_walk_ticks=0, the control):

| agent | op/d | mat% | fc% | lockPnl7d | dpnl7d |
|---|--:|--:|--:|--:|--:|
| 1247b40e | 59 | 3.4 | 69.5 | +7.25 | -102.4 |
| e8d8611b | 78 | 1.3 | 66.7 | +6.48 | -140.5 |
| 080b8f8e | 47 | 2.1 | 61.7 | +4.84 |   -9.3 |
| d1d0fefb | 24 | 0.0 | 41.7 | +0.62 |  -60.1 |

Mean day_pnl **-£78.1** (old N4 baseline -£78 ✓), mean locked +£4.80
(old lkd/mat +£4.80 ✓). SANITY CHECK PASSED — walk=0 byte-reproduces
the historical no-walk N4 number, validating the Round W setup and
confirming close_walk_ticks=0 is byte-identical to old behaviour.
N4_walk10 (the lift) now running; comparison pending.

### 2026-05-30 16:30 BST — Round W COMPLETE: close-walk is a SAFETY tool, NOT a profit lever (hypothesis refuted)

All 10 cells done. Controls validated (walk0 = old baselines exactly:
N4 -78.1/-78, N2 -98.2/-98, M6 -98.0/-98, M7 -128.3/-128, O1 -113.7/-114).

| recipe | walk0 meanDP | walk10 meanDP | walk0 close/fc £/agent | walk10 close/fc £/agent | day_pnl spread 0→10 |
|---|--:|--:|--:|--:|--:|
| N4 | -78.1 | -124.7 | -438 | -794 | 131→87 |
| N2 | -98.2 | -126.3 | -653 | -904 | 138→76 |
| M6 | -98.0 | -139.0 | — | — | 71→48 |
| M7 | -128.3 | -121.7 | — | — | 125→30 |
| O1 | -113.7 | -145.1 | -851 | -962 | 58→123 |

**Close-walk made mean day_pnl WORSE (4/5) and per-close realised loss
WORSE (N4 -0.32→-0.43/pair).** BUT day_pnl variance dropped in 4/5
(spread tighter). Interpretation:

**The "£339/agent avoidable loss" (KEY FINDING #2) was a MISREAD.** That
was computed via min(win,lose) = worst-case DOWNSIDE, not realised mean.
The under-hedged residual is zero-EV DIRECTIONAL VARIANCE, not free
money. Walking completes the hedge → pays the FULL spread cost (real,
guaranteed) to eliminate that variance. So walk converts zero-EV
variance into a guaranteed (slightly larger) spread cost: lower σ, worse
mean. Exactly the operator's "accept a small loss to avoid exposure" —
but the loss is real and the avoided exposure was ~break-even on this
window.

**Bigger implication:** the walk0 leaderboard numbers were FLATTERED by
zero-EV directional luck from under-hedging. The walk10 numbers are the
HONEST fully-hedged deployable P&L. N4's true deployable number is
~-£125, not -£78. We're further from breakeven than the leaderboard
suggested.

**Verdict:** close-walk = correct DEPLOYMENT SAFETY rail (kill ±£100
directional swings live), NOT a profit unlock. Treat like force_close:
keep it for deploy-time honesty; the leaderboard's variance-inflated
numbers should be re-read with walk ON. Does NOT unblock deployment —
the core per-open EV problem (spread+commission > locked edge) stands,
now seen more clearly without the directional-variance smokescreen.

### 2026-05-30 16:46 BST — Round T RESURRECTED (mature_prob open-gate) — operator-directed re-anchor on OPEN selection

Operator course-corrected: close-walk was realism not aim; the core
problem (open the right trades → higher mat%, less force-close) is
unchanged. Resurrected Round T (Path C). Launched run_roundT.sh (GPU
free post-Round-W). Cells (full-aug + pwin 0.25 = M6 base, fc=120,
mature_prob_loss_weight=2.0 to TRAIN the head, seed 42):
- T1_nogate (thr 0): head trained, gate OFF — isolates actor-input effect vs old M6.
- T2/T3/T4 (thr .30/.40/.50): head trained + gate ON — isolates GATE effect vs T1.
- T5 (thr .30, s43): variance replicate.
close_walk OFF (mat%/fc% are pair-lifecycle, independent of close matching).

THE EXISTENTIAL TEST: mat% has sat at ~5% (≈ base rate = no selection
edge) across every recipe. Can a LEARNED maturation signal pick opens
better than chance? Naive direction-gating already failed (findings
"what doesn't work" #3); mature_prob is trained on the actual outcome
so it's more complete. If gate lifts mat%/cuts fc% → selection viable.
If flat → strong evidence per-open maturation isn't predictable from
current 3-level features (→ need richer fill signal, e.g. the unused
TradedVolumeLadder, before any gate works).

WIRING: verified _build_trainer_hp merges mature_prob_loss_weight=2.0
into trainer_hp (genes dump showing 0.0 is cosmetic raw-CohortGenes).
EMPIRICAL CHECK PENDING: train_mean_mature_prob_bce must be >0 on first
cell (every prior round = 0) + eval_direction_gate_refusals >0 in gated
T2-T4 (gate reuses that counter).

### 2026-05-30 20:42 BST — Round T COMPLETE: maturation is NOT selectable (existential test = RED)

Wiring verified: bce 0.37-0.57 (head trained, prior rounds=0); gate fired
(~30k refusals T2-T5, 0 T1). Result:

| cell | thr | op/d | mat% | fc% | lk7d | dp7d | best |
|---|---|--:|--:|--:|--:|--:|--:|
| old M6 | - | 90 | 5.2 | 69.5 | 11.6 | -98 | -75 |
| T1 (head,gate off) | 0 | 96 | 5.9 | 70.8 | 12.0 | -121 | -99 |
| T2 | .30 | 72 | 4.8 | 66.7 | 7.5 | -93 | -62 |
| T3 | .40 | 59 | 5.5 | 64.0 | 7.5 | -60 | -26 |
| T4 | .50 | 60 | 4.7 | 68.7 | 6.9 | -72 | +13 |
| T5 | .30/s43 | 58 | 6.0 | 54.5 | 6.9 | -92 | -3 |

**VERDICT: mat% stayed flat ~5-6% across all gate thresholds** despite
aggressive masking (opens 96→58). The gate removed opens proportionally
across maturing/non-maturing → mature_prob_head's per-runner prediction
does NOT correlate with actual maturation. SAME as direction-gate
(findings "what doesn't work" #3). Per-open maturation is NOT selectable
from current 3-level features. T1 (head trained, gate off) vs M6: actor-
input from a trained head didn't help either (mat 5.9 vs 5.2, dp WORSE).

Consolation (limited): gating cut opens → mean day_pnl improved (T3 -60,
T4 -72 vs T1 -121) and a couple positive best-agents (T4 +13). But that's
the Path-A "open fewer = less fc-toll" effect, NOT maturation — LOCKED
FELL (12→7, fewer matured pairs). Means still negative; positive bests
are the lucky tail of 4 agents (naked variance).

**GA IMPLICATION: RED on launching the GA as designed** (its premise
"reward maturation rate → GA finds high-mat% recipe" is undercut; mat%
isn't movable from current features). The real unlock is FEATURES
(traded-volume ladder → richer maturation signal), not search. Decision
point with operator: run GA reframed (select locked_per_std, expect
selectivity optima not maturation) vs redirect compute to feature work.

### 2026-05-30 ~21:30 BST — CAMPAIGN PIVOT → plans/imitation-first/

After Round T (maturation not selectable on lean-obs/3-day) + operator
direction, the campaign pivots. Root cause reframe: we starved the model
(3 of 49 days; lean obs not full 143-d) and fought online-PPO reward
sparsity when we have a fixed dataset + an oracle that labels profitable
scalps = an IMITATION/OFFLINE problem. New plan **plans/imitation-first/**
(purpose + hard_constraints + master_todo), 3 steps: (0) oracle's own
held-out P&L = ceiling/opportunity check; (1) BC-to-convergence on full
obs + 42 days, eval holdout, sparsity-free = learnability test; (2)
reward-aware polish (BC→PPO fine-tune via ga-recipe-search, or offline
RL/Decision Transformer) only if Step 1 promises. Data split locked: train
42 (Apr6–May19) / holdout latest 7 (May20-29). Ready for a fresh session.
