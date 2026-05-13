# Autonomous run log — scalping-race-confidence-gate

One entry per `/loop` iteration. Append-only. The loop's stop
conditions are evaluated against this file.

## 2026-05-13 07:53 — Session 01, iteration 1

**State entering iteration:** plan freshly scaffolded; no code
changes yet. Sibling direction-gate plan's smoke previously FAILED;
this plan substitutes a per-race confidence filter built on the
champion `p_win` cache (already populated by the predictor-
integration path).

**Work done (full Session 01 — single iteration):**
- `env/betfair_env.py`: added `race_confidence_threshold: float =
  0.0` kwarg, validation block (loud-fail when threshold > 0 but
  `use_race_outcome_predictor=False` or `predictor_bundle is None`,
  range guard `[0, 1]`), `_race_confidence_gate_active` flag, and
  `_race_is_confident_by_race: list[bool]` cache. Populated in
  `_precompute` immediately after `_race_p_win_by_race.append(...)`
  using `max(p_wins.values()) >= threshold`.
- `agents_v2/action_space.py::compute_mask`: added race-level
  short-circuit immediately after the `bm is None` / `race_idx` /
  `tick_idx` early-outs. Default-off via `getattr(..., False)` so
  the mask is byte-identical to pre-plan when the flag is unset.
  When active and the current race is non-confident, returns the
  NOOP-only mask before any per-slot iteration.
- `training_v2/cohort/worker.py`: threaded
  `race_confidence_threshold` through `_build_env_for_day`,
  `train_one_agent`, and all three `_build_env_for_day` call sites
  (sizing env, per-day train env rebuild, held-out eval env).
- `training_v2/cohort/runner.py`: added `--race-confidence-threshold`
  CLI flag and threaded through `run_cohort()` → `train_one_agent`
  call.
- `tools/reevaluate_cohort.py`: added matching CLI flag + threaded
  through both `_build_env_for_day` calls (sizing env + per-eval-day
  env).
- `tests/test_agents_v2_action_space.py`: appended
  `TestRaceConfidenceGate` class with 7 tests:
  - `test_gate_disabled_by_default`
  - `test_confident_race_passes_through_unchanged`
  - `test_non_confident_race_masks_all_opens_and_closes`
  - `test_byte_identical_when_disabled`
  - `test_raises_without_use_race_outcome_predictor`
  - `test_invalid_threshold_raises`
  - `test_composes_with_pwin_gate` (additive composition with the
    pwin gate — verifies non-confident races mask OPEN_LAY
    regardless of pwin, AND confident races still see the pwin gate
    apply on high-pwin runners).

**Tests run:**
- `tests/test_agents_v2_action_space.py` — 45 passed (38 pre-existing
  + 7 new).
- `tests/test_betfair_env.py` — 62 passed (no regressions).
- `python -m training_v2.cohort.runner --help` confirms
  `--race-confidence-threshold` registers with the expected help
  text.

**Decisions made:** applied defaults from `master_todo.md`
(`race_confidence_threshold=0.30` is the cohort flag value, but the
code-side default stays 0.0 = disabled per hard_constraints §1).

**Outstanding for this session:** none — Session 01 complete.

**Next iteration's focus:** Session 02 (pre-flight smoke). Write
`tools/smoke_race_confidence_gate.py` against 2026-05-04. Three
threshold checks per hard_constraints §3.

## 2026-05-13 07:59 — Session 02, iteration 1 — SMOKE FAIL — STOP

**State entering iteration:** Session 01 committed (`cccb8ad`).
Gate code complete + tested. Move to pre-flight smoke per
hard_constraints §3.

**Work done:**
- Wrote `tools/smoke_race_confidence_gate.py` modelled on
  `tools/smoke_direction_gate.py`. Adapts the population statistic
  to per-race `max(p_win) >= threshold` and the legal-ratio
  threshold to ≤ 80% (per `hard_constraints.md §3`).
- Ran the smoke on `2026-05-04` with the locked defaults
  (`--race-confidence-threshold 0.30`, pwin back/lay = 0.20/0.40,
  device cuda). Bundle: champion + ranker + direction production
  manifests.

**Smoke result on 2026-05-04 (118 races, 8925 ticks):**

```
POPULATION:
  total races ........................... 118
  races confident (max p_win >= 0.30) ... 118 (100.00%)
  races skipped ......................... 0

LEGAL ACTIONS (post-mask):
  baseline (pwin only):
    OPEN_BACK ........................... 13890
    OPEN_LAY  ........................... 71700
  with race-confidence gate:
    OPEN_BACK ........................... 13890 (delta: +0)
    OPEN_LAY  ........................... 71700 (delta: +0)
    legal-tick ratio (with/no race gate)  100.00%

POLICY ROLLOUT (uniform-random over legal, 3 races):
  attempted opens BACK / LAY ............ 1296 / 6176
  matched bets .......................... 178
  → bets/race ........................... 59.33
  → full-day estimate (118 races) ....... 7001

VERDICT vs hard_constraints §3:
  race_qualification_rate >= 30%        PASS (actual 100.00%)
  legal_ratio <= 80% (material work)    FAIL (actual 100.00%)
  bets_matched >= 50 (full day est.)    PASS (estimate 7001)

OVERALL: FAIL — STOP loop
```

**Why this fails:** Every single one of the 118 races on
2026-05-04 has at least one runner the champion predictor scored
`p_win ≥ 0.30`. The race-confidence gate at threshold 0.30
refuses zero actions — it is structurally inert on this day. Per
hard_constraints §3, `legal_ratio ≤ 80%` is required for the
gate to do material work; we're at 100%. The gate is a no-op
under these inputs.

**Decisions made:**
- Per hard_constraints §3, `legal_ratio > 80%` is a binary FAIL.
- Per hard_constraints §9, `race_confidence_threshold` is LOCKED
  at 0.30 — the autonomous loop cannot tune it. Any threshold
  change is a new plan.
- Per stop condition §1 (`session_prompts/00_autonomous_full_
  run.md`): "Session 02 pre-flight smoke FAILS any of three
  thresholds (hard_constraints §3). Stop, write diagnostic. Do
  NOT launch the 12h cohort."
- STOP the autonomous loop. Do NOT schedule the next wakeup.
  Surface the headline to the operator.

**Outstanding for this session:** none — gated correctly by the
smoke discipline. No 12h cohort wasted. Same outcome shape as the
direction-gate plan's smoke save.

**Next plan (operator-decision, NOT autonomous):** the data tells
us that on a representative day, racing favorites virtually always
have champion `p_win ≥ 0.30`. The threshold is too permissive to
function as a race-selection filter. Two branches the operator
might consider:

1. **Higher threshold** (e.g. 0.40 or 0.50). Per the smoke's
   POPULATION block, a tighter threshold would actually shrink
   the legal-action surface and let the gate function as intended.
   This requires a new plan because §9 locks 0.30 for this plan;
   re-opening "race-confidence-gate-v2" with a higher default
   would not violate the spirit of the locked threshold.
2. **Different per-race signal**. `segment_strong_flag` from the
   deterministic baseline already encodes "this race has a runner
   the predictor strongly favors" — it might capture race-level
   confidence more selectively than a raw `max(p_win)` cutoff.
   Per §9 this is also a new plan, not a mid-flight tweak.

Recommendation: a follow-on `scalping-race-confidence-gate-v2`
plan with `threshold=0.50` default, modelled on this plan but
using the smoke's per-day max-p_win distribution to set a value
that produces a meaningful split. The current cohort budget (12 ×
8 × 6) is unchanged — only the gate value differs.

## 2026-05-13 08:51 — operator override: pick a new threshold in-plan

**Operator instruction** (this conversation): "yes, use different
defaults. See if you can find a sensible figure that works".

The follow-on-plan path is dropped; the threshold is re-tuned
in-place. The autonomous loop resumes with the operator-authorised
new default.

## 2026-05-13 08:51 — Session 02b, iteration 1 (post-override probe)

**State entering iteration:** smoke FAILED at 0.30 (gate inert).
Operator authorised a new threshold; I need to pick one supported
by the per-race max-p_win distribution.

**Work done:**
- Wrote `tools/probe_race_confidence_distribution.py`. Reads
  `_race_p_win_by_race` after env construction across N days,
  prints quantiles + threshold candidate table.
- Ran the probe across 5 days
  (2026-05-01, 02, 04, 05, 06 — 434 races total).
- Distribution (max per-race champion p_win):

```
QUANTILES:
  p10 = 0.3865
  p25 = 0.4258
  p50 = 0.5338
  p75 = 0.6140
  p90 = 0.7069

THRESHOLD → races SKIPPED (qualification rate):
  0.30  →   0%  (100% qualify) ← FAIL'd smoke; gate inert
  0.40  →  16%  ( 84%)
  0.45  →  30%  ( 70%)
  0.47  →  33%  ( 68%) ← probe's suggestion
  0.50  →  40%  ( 60%) ← chosen default
  0.55  →  58%  ( 42%)
  0.60  →  72%  ( 28%) ← would FAIL §3 qualification floor
```

- Per-day medians confirm the distribution is stable across days:
  2026-05-01 = 0.59, 2026-05-02 = 0.52, 2026-05-04 = 0.53,
  2026-05-05 = 0.51, 2026-05-06 = 0.52.

**Decision: `race_confidence_threshold = 0.50`.** Sits at the
median of the observed distribution — natural "above-median
favorite" cut. Skips ~40% of races (legal_ratio ~60%, comfortably
under §3's 80% bar) without starving training (race
qualification rate ~60%, well above the 30% floor).

The probe also tested 0.47 (its automatic suggestion based on
landing in the 25-40% target band) but 0.50 is a cleaner
round-number choice and sits exactly at the population median.

**Plan documents updated (in-place):**
- `README.md` — threshold-choice section rewritten with the probe
  evidence + new default.
- `hard_constraints.md` §4 — `--race-confidence-threshold 0.30`
  → `0.50`.
- `master_todo.md` — Session 03 launch command threshold.
- `session_prompts/00_autonomous_full_run.md` — defaults table
  entry + "What NOT to do" lock.
- `session_prompts/03_launch_cohort.md` — replaced all 0.30
  references with 0.50 + revision note.

**Outstanding:** re-run smoke at the new threshold.

**Next iteration's focus:** smoke at 0.50, verify all three §3
PASS, then Session 03 launch.

## 2026-05-13 08:53 — Session 02b, iteration 2 — SMOKE PASS

**State entering iteration:** plan docs updated to threshold 0.50.

**Work done:**
- Ran `tools.smoke_race_confidence_gate --day 2026-05-04
  --race-confidence-threshold 0.50 --device cuda`.

**Smoke result on 2026-05-04 (118 races, 8925 ticks) at
threshold 0.50:**

```
POPULATION:
  total races ........................... 118
  races confident (max p_win >= 0.50) ... 65 (55.08%)
  races skipped ......................... 53

LEGAL ACTIONS (post-mask):
  baseline (pwin only):
    OPEN_BACK ........................... 13890
    OPEN_LAY  ........................... 71700
  with race-confidence gate:
    OPEN_BACK ........................... 8133  (delta: -5757)
    OPEN_LAY  ........................... 33287 (delta: -38413)
    legal-tick ratio (with/no race gate)  48.39%

POLICY ROLLOUT (uniform-random over legal, 3 races):
  attempted opens BACK / LAY ............ 750 / 3109
  matched bets .......................... 98
  → bets/race ........................... 32.67
  → full-day estimate (118 races) ....... 3854

VERDICT vs hard_constraints §3:
  race_qualification_rate >= 30%        PASS (actual 55.08%)
  legal_ratio <= 80% (material work)    PASS (actual 48.39%)
  bets_matched >= 50 (full day est.)    PASS (estimate 3854)

OVERALL: PASS — proceed to Session 03
```

All three §3 thresholds PASS with healthy margin:
- 55% qualification rate (1.8× the 30% floor)
- 48% legal-ratio (gate refusing 52% of action surface — substantial work)
- 3854 estimated full-day matched bets (76× the 50 floor)

**Decisions made:** proceed to Session 03 launch.

**Outstanding for this session:** none — Session 02b complete.

**Next iteration's focus:** Session 03 — launch the 12h cohort
with the locked-by-operator threshold 0.50. Verify Gen 1 starts
within 5 min, arm watcher, sleep into 1h heartbeat mode.

## 2026-05-13 09:32 — Session 03, iteration 1 — COHORT LAUNCHED

**State entering iteration:** smoke PASS at threshold 0.50;
operator confirmed launch via AskUserQuestion.

**Work done:**
- Pre-checks: GPU 22 GB free (RTX 3090), all three predictor
  manifests present, prior cohort code untouched.
- Launched cohort:
  - TAG = `_predictor_SCALPING_raceconf_1778661062`
  - PID 635
  - Log: `registry/_predictor_SCALPING_raceconf_1778661062.log`
  - Command: `training_v2.cohort.runner --n-agents 12
    --generations 8 --days 6 --seed 42 --mutation-rate 0.2
    --strategy-mode arb --predictor-p-win-back-threshold 0.20
    --predictor-p-win-lay-threshold 0.40
    --race-confidence-threshold 0.50` + the 6 safety genes +
    three production manifests + lean obs.
- Verified Gen 1 started (log shows "Cohort: 12 agents × 8
  generations on 3 training days (eval=2026-05-04…2026-05-06 (3
  days)); device=cuda" and agents 1-12 enumerated with their
  starting genes).
- Wrote watcher `/tmp/auto_reeval_raceconf.sh` (mirror of the
  pwin-gate watcher template; threshold 0.50 in the reeval call;
  tag substituted). Launched in background, PID 657. Output to
  `/tmp/auto_reeval_raceconf.log`; first lines confirm "waiting
  for 96 rows".
- Launched `tools.show_cohort_status --watch 60`, PID 668.
- Verified cohort artefacts present: `models.db`,
  `scoreboard.jsonl` (empty so far), `status.txt`
  ("Progress: 0/96 agents"), `weights/`, `bet_logs/`.

**Tests run:** none in this iteration — cohort itself is the
test surface. Pre-flight smoke covered correctness.

**Decisions made:** none new; mirroring predecessor pwin-gate
launch verbatim except for the new threshold.

**Outstanding for this session:** ~12h wall-time waiting for the
cohort to fill 96 scoreboard rows. Sleep into 1h heartbeat mode.

**Next iteration's focus:** heartbeat — read `status.txt`, log a
single-line entry "heartbeat N/96 rows", check for Traceback. If
N=96, jump to Session 04.

## 2026-05-13 10:33 — Session 03, heartbeat 1/N — 7/96 (healthy)

- Rows: 7/96 (~7% in ~1h; on track for ~13h total wall-clock).
- Traceback count in cohort log: 0.
- Watcher PID 657 alive, logging "still running" every 5 min.
- Generation 0 partial (7/12 agents complete): mean eval pnl
  **+£125**, median +£166, best +£297; 6/7 profitable. For
  context: predecessor pwin-gate cohort verdict was mean
  **−£13** held-out, 3/5 profitable. Mid-flight in-sample
  doesn't satisfy hard_constraints §7 (no premature stop) — the
  verdict will be set by the held-out reeval at 96 rows. But the
  early signal is consistent with the hypothesis.
- Top eval pnl: f80af8b3 +£297, 2a878902 +£256, 9bc51046 +£247.
  Maturation rates 0.30–0.38 (vs pwin-gate's 0.334 baseline).
- No action this iteration. Sleep 1h.

## 2026-05-13 11:34 — Session 03, heartbeat 2/N — 14/96 (healthy)

- Rows: 14/96 (+7 since last hb; avg 7.7 min/agent on track).
- Traceback count: 0.
- Gen 0 closed at mean **+£114, 10/12 profitable** (median +£105,
  best +£406, mean_locked +£86, mean_naked +£44).
- Gen 1 partial (2/12): d5899277 +£94, 24c4d51b −£160. The −£160
  is naked −£216 — same variance shape as the worst Gen 0 agents
  (bbe37901 −£170 via naked −£236). The locked floor on the loser
  is +£71, confirming the pattern: locked is the *steady* signal
  (~+£70-110 across virtually every agent), naked is the *variance*
  (driving both top of leaderboard and bottom). Selection pressure
  should over the next 6 generations push survivors toward
  locked-dominance.
- No action. Sleep 1h.

## 2026-05-13 12:35 — Session 03, heartbeat 3/N — 22/96 (healthy)

- Rows: 22/96 (+8 since last hb; avg 7.5 min/agent; ETA 9.2 h).
- Traceback count: 0.
- Gen 1 partial (10/12): mean +£42, median +£11, 5/10 profitable.
  Breakdown: locked +£81, naked −£23. Locked floor stable across
  Gen 0 vs Gen 1 (+£86 → +£81); naked flipped from +£44 to −£23
  — the variance pattern confirms (locked = signal, naked =
  variance, mean-reverts).
- New strong performers in Gen 1: 8f9e2c96 +£310, 1664c56e +£210,
  76eb3675 +£196 — selection pressure already picking up.
- Gen 1 mean pulled up by the +£310 outlier; median +£11 is the
  cleaner signal of typical Gen 1 performance.
- No action. Sleep 1h.

## 2026-05-13 13:36 — Session 03, heartbeat 4/N — 30/96 (healthy)

- Rows: 30/96 (+8 since last hb; ETA 8.1 h).
- Traceback count: 0.
- Gen 1 closed (12/12): mean +£56, median +£84, 7/12 profitable
  (locked +£81, naked −£9).
- Gen 2 partial (6/12): mean +£60, median +£64, 4/6 profitable
  (locked **+£90** — highest of any gen so far, naked −£13).
- Locked floor trajectory across Gens 0/1/2: +£86 → +£81 → +£90.
  Steady-to-slightly-improving — exactly what PPO learning locked
  spread extraction should look like.
- Naked trajectory: +£44 → −£9 → −£13. Mean-reverting toward
  small negative (as predicted; pre-cohort hypothesis).
- New top-10 entrant from Gen 2: 4b4101b5 +£230 (locked +£109,
  naked +£133). Bottom-3 unchanged since hb 2.
- No action. Sleep 1h.

## 2026-05-13 14:38 — Session 03, heartbeat 5/N — 39/96 (healthy)

- Rows: 39/96 (+9 since last hb; ETA 6.9 h).
- Traceback count: 0.
- Gen 2 finalised (12/12): mean +£2, median +£27, 7/12 profitable
  (locked +£82, naked **−£63**). Gen 2 was the worst naked draw
  of the run so far.
- Gen 3 partial (3/12): mean +£74, median +£68, 3/3 profitable
  (locked +£84, naked +£8). Small n; too early to call selection
  effect, but the locked floor remains rock-solid.
- Locked trajectory across gens 0/1/2/3: +£86/+£81/+£82/+£84.
  σ across the four gen means is ~£2 — by far the tightest
  statistic in this entire cohort.
- New bottom-3: d2a3cae5 (Gen 2) and af560e14 (Gen 2) both at
  naked < −£249. Both have locked > +£66, so locked floor holding
  even on the worst agents.
- No action. Sleep 1h.

## 2026-05-13 15:40 — Session 03, heartbeat 6/N — 47/96 (healthy)

- Rows: 47/96 (+8 since hb 5; ETA 5.9 h).
- Traceback count: 0.
- Gen 3 essentially complete (11/12): mean +£36, median +£32,
  **9/11 profitable (82%)**, locked +£85, naked −£32. The hb-5
  read (n=3 then) was selection bias from early-finishers; with
  n=11 the gen settles at +£36 mean rather than +£74. Still a
  good gen — highest profitability rate of any complete gen, but
  not a standout in absolute terms.
- The "tight cluster, low peak, high floor" shape holds with the
  full Gen 3 data: best agent only +£152 (no naked windfalls),
  worst is positive (no naked disasters either), median ≈ mean
  (Δ=4). Top-10 leaderboard still Gen 0/1/2 dominated; Gen 3 is
  consistently mid-pack with high floor. This is what locked-
  driven trading should look like.
- Watch Gen 4 in the next hb — if the tight-distribution pattern
  persists, GA+PPO selection is genuinely converging on locked
  agents. If Gen 4 reverts to high-variance shape, Gen 3 was a
  lucky draw.
- No action. Sleep 1h.

## 2026-05-13 16:41 — Session 03, heartbeat 7/N — 55/96 (healthy)

- Rows: 55/96 (+8 since hb 6; ETA 5.0 h).
- Traceback count: 0.
- Gen 3 final (12/12): mean +£40, median +£33, **10/12
  profitable (= Gen 0 for highest %)**, locked +£85, naked −£27.
  The tight-distribution shape held with full Gen 3 data.
- Gen 4 partial (7/12): mean +£93, median **+£104** (much higher
  than Gen 3's +£33), best +£223, **naked back to +£28** (positive
  again, first since Gen 0). 6/7 profitable.
- The Gen 3 "convergence" call looks shakier with Gen 4 data —
  Gen 4 is wider than Gen 3, with one naked-£160-driven top-10
  entrant (4cb5f7fb). Either (a) Gen 4 is reverting to high-
  variance shape because the GA selected on raw pnl (which
  rewards lucky-naked Gen 0/1 agents) and their offspring are
  rolling the same dice; or (b) the small Gen 4 sample (n=7) is
  selection-biased toward early-finishers and the median will
  drop when n=12. Hb 8 with full Gen 4 will tell us.
- Locked trajectory across gens 0-4: 86/81/82/85/83. σ across
  the five gen means is ~£2 — still the tightest stat in the
  cohort. Locked floor is genuinely real and stable; the entire
  cohort's per-agent variance is in the naked channel.
- No action. Sleep 1h.

## 2026-05-13 17:42 — Session 03, heartbeat 8/N — 62/96 (healthy, watch Gen 5)

- Rows: 62/96 (+7 since hb 7; ETA 4.1 h).
- Traceback count: 0.
- Gen 4 final (12/12): mean +£69, median **+£104**, best +£326,
  **8/12 profitable (67%)**, locked +£84, **naked +£2** — first
  non-Gen-0 generation to land non-negative naked mean. Two Gen-4
  agents reached top-10: 0eba6ae1 (+£326, naked +£251) and
  4cb5f7fb (+£223, naked +£160). Both top entries naked-driven,
  echoing the Gen 0 lucky-tail pattern. Gen 4 mean higher than
  Gen 3 (+£69 vs +£40) but profitability rate LOWER (67% vs 83%)
  — wider distribution.
- Gen 5 partial (2/12): mean −£37, **locked +£75 (lowest of any
  gen)**, naked −£91, 0/2 profitable. Very early sample; could be
  the worst-finishing-first selection bias, or a real shift.
  Watch Gen 5 carefully next hb.
- Locked trajectory across gens 0-5: 86/81/82/85/84/**75**. The
  rock-solid floor cracked on Gen 5. If Gen 5's locked stays
  ≤£78 with full n=12, the GA-selected genes have shifted toward
  agents that don't extract locked spread as effectively. This is
  recoverable in Gens 6/7 if PPO training and selection pull the
  cohort back, but flag the data point.
- No action this iteration. Sleep 1h.

## 2026-05-13 18:43 — Session 03, heartbeat 9/N — 70/96 (healthy)

- Rows: 70/96 (+8 since hb 8; ETA 3.2 h).
- Traceback count: 0.
- Gen 5 partial (10/12): mean +£41, median +£93, 5/10 profitable,
  **locked +£79** (lowest gen mean so far), naked −£21. The hb-8
  alarm (locked +£75 on n=2) was selection bias; with 10/12 the
  locked floor settles a few £ below the £81-86 range but doesn't
  collapse. Individual agents still report locked >£70 throughout
  the gen.
- Two new Gen-5 leaderboard entries: ff7cdbe6 +£300 in top-10
  (naked +£219, naked-driven again) and b44d53f3 −£214 in
  bottom-3 (naked −£287). Both are within the established
  naked-variance pattern.
- Locked trajectory across gens 0-5: 86/81/82/85/84/79. Drift is
  small (~£7 range) but the trend over the last 3 gens is
  downward (85 → 84 → 79). Worth watching whether Gen 6/7
  recover or continue drifting. Could indicate that GA selection
  on raw pnl is gradually selecting AGAINST the most
  locked-disciplined agents (because their pnl loses to lucky-
  naked agents in-sample).
- Top-10 composition: Gen 0×5, 1×2, 2×1, 4×2, 5×1. Cross-gen
  spread; no single gen dominates. Still naked-driven on average
  (the top-10 mean naked is +£200, top-10 mean locked is +£88).
- No action. Sleep 1h.

## 2026-05-13 19:44 — Session 03, heartbeat 10/N — 78/96 (healthy, Gen 6 strong)

- Rows: 78/96 (+8 since hb 9; ETA 2.2 h — final reeval should
  fire around 21:55 wall-clock).
- Traceback count: 0.
- Gen 5 final (12/12): mean +£54, median +£93, 7/12 profitable,
  locked **+£75**, naked −£4. Locked confirmed as the lowest gen
  mean of the run; the "watch Gen 5" call from hb 8 was warranted.
- Gen 6 partial (6/12): mean **+£105**, median **+£132**, best
  **+£420** (new all-time best — 0a3cc000), 4/6 profitable,
  locked **+£89** (new high), naked +£35. Gen 6 strongly recovers
  the locked floor that dipped in Gen 5.
- Locked trajectory across gens 0-6: 86/81/82/85/84/75/89.
  Gen 5 dip was real but transient. Could be GA's stochastic
  per-gen draw (mutations introduce variance); the underlying
  signal looks stable.
- 0a3cc000 (Gen 6, +£420, locked +£106, naked +£330) — new
  all-time best agent. Locked component is the second-highest
  locked of any agent (after 609bf1a7 at +£121). Hopeful sign
  if its locked dominates on held-out reeval where the +£330
  naked windfall WON'T replicate.
- Top-10 now: Gen 0×5, 1×2, 2×1, 4×2, 5×1, 6×1. Cross-gen
  spread. Bottom-3 unchanged.
- Next hb (20:44) is likely the last full one before completion;
  the one after (21:44) will probably catch the reeval running.
- No action. Sleep 1h.

## 2026-05-13 20:45 — Session 03, heartbeat 11/N — 86/96 (healthy, Gen 6/7 strong)

- Rows: 86/96 (+8 since hb 10; ETA 1.2 h; cohort completion
  ~21:55, reeval will fire shortly after).
- Traceback count: 0.
- Gen 6 final (12/12): mean **+£105** (second-best of run after
  Gen 0's lucky +£114), median **+£134** (best of any gen), best
  +£420, **9/12 profitable (75%)**, locked +£85, naked +£39.
  Critically, the +£39 naked is positive but modest — not a
  lucky-tail-driven generation.
- Gen 7 partial (2/12): mean +£132, locked **+£104** (highest
  locked of any gen, suggesting strong selection toward locked-
  extractive agents), naked +£41. Status display shows median =
  best = +£334 for n=2 which looks like a display quirk; sample
  too small for the moment.
- Locked trajectory across gens 0-7: 86/81/82/85/84/75/85/104.
  Gen 7's +£104 is genuinely remarkable. The Gen 5 dip was a
  one-gen anomaly; the underlying trend is locked-floor strong
  and improving in later gens.
- New top-10 entries: 35297cd3 (Gen 6) +£230 and 0de125f5
  (Gen 7) +£334, both with locked >£94 — the late-gen agents
  are bringing real locked extraction skill.
- Next hb (21:45) will catch the cohort at completion or
  shortly after; watcher should fire reeval automatically when
  scoreboard hits 96 rows.
- No action. Sleep 1h.

## 2026-05-13 22:17 — Session 03, heartbeat 12 / Session 04 — PLAN COMPLETE

- Rows: 96/96. Cohort complete at 21:57 (12 h 26 min from launch).
- Watcher detected at 21:57:04, fired reeval. Reeval completed
  22:16:23 (19 min 19 s).
- Traceback count: 0 across entire run.
- Reeval output: `registry/_predictor_SCALPING_raceconf_
  1778661062/reeval_held_out_2026-04-28_30.jsonl` (5 top agents
  on 2026-04-28/29/30).

### HELD-OUT VERDICT

```
agent           gen  mean pnl/day  locked  naked   closed  stop    mr
8ab4204f-0b4    6    +£137.42     +£102.60 +£54.20 −£4.57  −£14.82 0.337
d3471cae-dc6    7    +£103.45     +£82.23  +£39.58 −£3.84  −£14.52 0.326
35297cd3-4e1    6    +£92.61      +£90.13  +£30.17 −£15.96 −£11.73 0.372
0de125f5-648    7    −£2.02       +£86.72  −£61.62 −£14.75 −£12.37 0.376
f5001118-0e5    7    −£134.45     +£77.89  −£197.02 −£2.92  −£12.40 0.298
```

- profitable: 3/5 (PASS — bar ≥ 3/5)
- mean per-day pnl: **+£39.40** (PASS — beats pwin-gate's −£13
  by **+£52.40**)
- median per-day pnl: +£92.61

Per the README "What success looks like" table: **MODEST
SUCCESS** (mean > 0 AND ≥ 3/5 profitable; below "strong" which
needs ≥ 4/5).

### Plan output

`findings.md` written and committed. Detailed per-day, per-agent,
per-channel breakdown + lessons learnt + recommended direction
for the next plan included.

### Stop the loop

Per `session_prompts/00_autonomous_full_run.md` stop condition #1
(Session 04 findings.md is written and committed → plan complete).
No next wakeup scheduled.
