# Env-side recipe sensitivity sweep — findings

Wrapper: `run_after_gradient_sweep.sh`
Ran: 2026-05-25 08:19 → 11:42 BST. 8 cells × 4 agents × 1 gen × 3 train
days × 5 eval days. Total wall ~3h 23m.

All cells share: BC pretrain off, frozen C11 direction head loaded,
all 3 predictors active, lean obs, force_close=120s baseline,
close_feasibility_max_spread_pct=0.05, matured_arb_expected_random=0.0.

## Top-line table

Mean across 4 agents × 5 eval days per cell:

| cell                 | day_pnl | locked | naked  | force_closed | closed | bets | opens | mat%  | cls%  | fc%   | d_gate_ref |
|----------------------|--------:|-------:|-------:|-------------:|-------:|-----:|------:|------:|------:|------:|-----------:|
| **C0_baseline**      | -166.6  | +19.1  | -28.2  | -106.5       | -51.1  |  263 |   134 | 1.3%  | 31.7% | 63.1% |          0 |
| **C1_fc60**          | -163.4  | +21.2  | -17.0  | -116.1       | -51.4  |  274 |   140 | 1.8%  | 30.5% | 63.4% |          0 |
| **C2_pwin_back_020** | **-102.4** | +14.6 | +16.2 | -82.8     | -50.5  |  219 |   112 | 1.6%  | 38.7% | 55.7% |          0 |
| **C3_pwin_lay_050**  | -189.7  | +16.2  | -43.9  | -114.7       | -47.2  |  266 |   135 | 1.3%  | 29.8% | 65.1% |          0 |
| **C4_race_conf_035** | -205.5  | +13.7  | -46.6  | -122.9       | -49.7  |  259 |   132 | 1.5%  | 31.1% | 63.8% |          0 |
| **C5_dir_gate_030**  | -220.1  | +26.1  |  -5.5  | -164.3       | -76.4  |  430 |   217 | 0.8%  | 29.6% | 67.6% |     35,274 |
| **C6_dir_gate_045**  | -242.6  | +22.6  | -39.9  | -142.6       | -82.6  |  398 |   201 | 0.5%  | 33.3% | 64.1% |     46,190 |
| **C7_all_on**        | -73.5   | +8.5   | +59.1  | -65.1        | -76.0  |  176 |    88 | 0.0%  | 63.8% | 35.6% |      1,887 |

`day_pnl` = held-out cash P&L per agent per day. `mat%` = matured
naturally / pairs_opened. `cls%` = agent-closed via close_signal /
pairs_opened. `fc%` = env force-closed / pairs_opened. `d_gate_ref` =
policy-side direction-gate refusals per day.

## Headline findings

### 1. Back-side pwin gate (C2) is the only single env-side prior that helps

C2 (`--predictor-p-win-back-threshold 0.20`) reduces day_pnl loss
**by £64/day vs C0** (-£102 vs -£167). Mechanism:

- Opens drop 134 → 112 (−16%). Bets 263 → 219 (−17%).
- Close rate climbs 32% → 39%; force-close rate drops 63% → 56%.
- Naked term **flips sign**: -£28 (C0) → +£16 (C2).
- Force-closed loss shrinks from -£107 to -£83.

The mechanism — env refuses OPEN_BACK on runners whose champion
p_win < 0.20 — kills speculative back picks at the source. The
agent has fewer opens to defend, more of them mature or close
cleanly, and the survivors don't blow up the naked term. This is
the "remove decisions beats teaching them" pattern from the
2026-05-20 lessons-learnt memo, repeating cleanly.

This is the only env-side knob that bit at probe scale.

### 2. Lay-side pwin gate (C3) and race-confidence (C4) actively hurt

C3 (lay p_win cap at 0.50, refuses OPEN_LAY on runners whose
p_win > 0.50) and C4 (`--race-confidence-threshold 0.35`, refuses
opens in low-confidence races) BOTH regress vs C0:

- C3: -£190/day (−£23 vs C0). Naked **worsens to -£44** (vs -£28).
  Same open count, same fc rate — the lay gate didn't reduce
  activity but it changed _which_ lay opens went through, and
  the surviving lays were worse.
- C4: -£206/day (−£39 vs C0). Same pattern — naked -£47, opens
  unchanged, fc rate unchanged.

The lay-side p_win cap was designed against an earlier diagnosis
(lay-quality-gate) where high-p_win lays were structural -EV. At
the current cohort scale, with the C11 direction head live in
obs, the previously-bad lay set seems to have already been
filtered by the policy. Applying the env-side cap removes opens
the policy already had a useful gradient on, leaving only the
hard-to-trade tail.

### 3. Policy-side direction gate confirms clamp fix works but the gate HURTS

C5 (threshold 0.30) and C6 (threshold 0.45) both have non-zero
`d_gate_ref` (35,274 and 46,190 refusals/day respectively),
proving the
2026-05-25 `DIRECTION_GATE_THRESHOLD_MIN` clamp fix landed and
the policy-side gate is actually firing. Pre-fix this counter
was 0 across the gradient sweep regardless of gene draw.

**But the gate makes things worse, not better:**

- C5: day_pnl -£220 (worst −£53 vs C0); mat% drops to 0.8%; opens
  PARADOXICALLY RISE to 217 (vs C0's 134).
- C6: day_pnl -£243 (worst −£76); mat% drops to 0.5%; opens 201.

The paradoxical open increase suggests the policy compensates for
gate refusals by trying open actions more often, since each
attempt has a higher chance of being rejected. The agent ends up
attempting MORE opens to land FEWER pairs, with each pair worse
on average. The direction signal at the calibrated C11 head
threshold isn't useful at the moment of OPEN decision.

This corroborates the price-band findings: the direction
predictor is anti-informative at favourites (price 1-2) where
the agent is currently opening 90% of its activity. Gating on
direction signal _at the price band the agent actually trades_
filters the wrong opens.

### 4. "Stack everything on" (C7) is the headline anomaly

C7 day_pnl -£73/day is the best of the eight, BUT:

- `mat% = 0.0%` across all 4 agents (zero natural maturations).
- Naked term **+£59/day** dominates the result.
- Opens collapse to 88 (lowest in the sweep, -34% vs C0).
- `d_gate_ref` is only ~1.9k (vs 35k in C5) because the upstream
  pwin/race_conf gates filter most opens before direction gate
  sees them.
- Close rate jumps to 63.8%.

This is the "policy gave up and rode naked variance lucky" shape
from the 2026-05-14 phenotype lessons. At n=4 agents and one
generation, +£59 nakeds across 88 opens is just gambling — the
σ is large and we have no held-out re-eval to discount luck. The
mat=0% kills any deployment story this cell could have told.

Do not chase this configuration.

### 5. Force-close window (C1: 60s vs C0: 120s) is approximately a no-op

C1 day_pnl -£163 vs C0 -£167 (Δ −£3, well within noise at n=4).
Tightening the force-close window from 120s to 60s gives the
agent more pre-off time to mature pairs naturally but doesn't
meaningfully change behaviour at the current open/close cadence.
Worth carrying forward as a held-out flag (lower deploy-time
exposure) but not a training-stage lever.

## Important methodology caveats

### Env-side gate refusal counters are placeholders

`pwin_back_gate_refusals`, `pwin_lay_gate_refusals`, and any
race-confidence refusal counter are **never incremented** in the
code path. The gates themselves ARE wired (`agents_v2/action_space
.py:366-368` consume the thresholds; C2's outcome diverges
measurably from C0). The COUNTERS are placeholders for a future
plan — the comment at `env/betfair_env.py:1186` is explicit:
"placeholders for the future `--predictor-p-win-back-threshold`
gates".

Only the policy-side `direction_gate_refusals` increments cleanly
(via `training_v2/discrete_ppo/rollout.py:638`).

**Implication:** we cannot quantify per-episode env-side gate
ACTIVITY from these scoreboard rows. We can only observe gate
EFFECT through the cash P&L and pair-lifecycle counters. For
future probes, wire the missing increments.

### N=4 per cell, single generation

Every cell ran 4 agents × 1 gen × 5 eval days. Per-agent
within-cell consistency was high (see per-agent breakdown
earlier in scratchpad), so the cell-level means are not driven
by a single outlier. But cross-cell Δs of ±£20–40 per day
should be treated as suggestive — solid signals need
re-validation at higher agent counts.

### Day-pnl is a noisy headline

Day_pnl mixes locked, naked, closed, and force_closed terms.
The naked term is dominated by directional luck and `mean(locked)
/ σ(naked_leg)` is the better selection metric for deployment
(memory: `naked_variance_primary_metric`). C7's headline -£73
is the textbook example of the trap.

## Cross-cutting interpretation

Three patterns line up across the eight cells:

1. **Remove-bad-decisions wins. Teach-better-decisions lost.**
   C2's selection-removal env prior bit by £64/day. C5/C6's
   policy-side gradient on direction signal cost £53–£76/day.
   This is the third repetition of the
   `remove_decisions_beats_teaching` pattern at probe scale (was
   E3/E4/E5 in the 2026-05-20 cohort, now C2 here).

2. **Adding gates is not additive.** C7 ≠ C2 + C3 + C4 + C5/6.
   Stacking the four gates that helped or hurt individually
   collapsed the policy to bets=88 / mat=0%. Each gate filters
   a different decision; their intersection is too narrow for
   the policy to learn within in one generation.

3. **The direction predictor's information helps in obs but
   hurts at the gate.** C0 (direction in obs, no gate) outperforms
   C5/C6 (direction in obs, gate ON). This pairs with the price-
   band finding from `recipe-sensitivity-sweep/price_band_findings.md`:
   the predictor is informative at price 3-10 but the policy
   doesn't trade there; gating uniformly at the calibrated
   threshold filters opens that the predictor isn't qualified
   to score (price 1-2).

## Production-recipe recommendation

Carry forward into next cohort:

- ✅ `--predictor-p-win-back-threshold 0.20` (C2's bite is the
  only clean win).
- ✅ `force_close_before_off_seconds=120` (default; C1's 60s
  isn't structurally better but is fine for held-out re-eval).
- ❌ Do NOT enable `--predictor-p-win-lay-threshold` (C3
  regressed; previous lay-gate findings may not survive the
  current obs schema).
- ❌ Do NOT enable `--race-confidence-threshold` at 0.35 (C4
  regressed; threshold may be miscalibrated for current
  predictor distribution).
- ❌ Do NOT enable `--direction-gate-enabled` at the calibrated
  thresholds (C5/C6 regress). The D-cells experiment will
  determine if any lower threshold (0.20) helps; until then, the
  policy-side direction gate is off-by-default.
- ❌ Do NOT stack multiple gates (C7's mat=0% collapse).

## Pending experiments downstream of this sweep

Two probe pipelines chained on this wrapper's completion:

1. **Close-penalty + BC-pretrain probes** (`plans/oracle-alignment-
   investigation/run_after_env_sweep.sh`): PC0 baseline, PC1
   close_signal_bonus=-2, PC2 close_signal_bonus=-5, PC3
   bc_pretrain_steps=1000, PC4 BC + penalty. Tests the "selection
   failure (price-band) + holding failure (close_signal as
   stop-loss)" diagnosis from `oracle-alignment-investigation/
   purpose.md`.
2. **D-cells (direction gate threshold response)** (`plans/
   direction-predictor-mechanism/run_after_close_penalty.sh`):
   D0 gate off, D2 t=0.30, D2b t=0.45, D3 t=0.20. First proper
   threshold-response test with the clamp fix. Compares to C0,
   C5, C6 here — direct comparison since same train/eval days
   and same flags otherwise.

## Follow-up tooling

- Wire `pwin_back_gate_refusals` / `pwin_lay_gate_refusals` /
  add a `race_confidence_gate_refusals` counter so we can
  quantify env-side gate activity in future sweeps. The
  placeholder comment in `env/betfair_env.py:1186` calls this out.
  Effort: ~30 min (3 increments in action_space.py mask builder
  + tests).
