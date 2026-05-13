# Autonomous run log — scalping-lay-quality-gate

Per-iteration log of the autonomous run. Each iteration writes
one entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

## 2026-05-13 — Phase 0, iteration 1

**State entering iteration:** plan folder existed with only
`session_prompts/00_autonomous_full_run.md`; predecessor
`scalping-race-confidence-gate` complete (mean +£39.40/day,
3/5 profitable held-out).
**Work done:**
- Verified no active cohort process is running
  (`Get-Process python` showed only the predecessor watcher).
- Created `plans/scalping-lay-quality-gate/README.md`.
- Created `plans/scalping-lay-quality-gate/hard_constraints.md`.
- Created `plans/scalping-lay-quality-gate/master_todo.md`.
- Created `session_prompts/01_probe.md` through
  `06_compare_and_verdict.md`.
- Created this `autonomous_run_log.md`.
**Tests run:** none (scaffolding only).
**Decisions made:**
- Inherit `race_confidence_threshold = 0.50` (predecessor's
  smoke-PASS value).
- Phase 1 will set `predictor_p_win_lay_threshold` and
  `lay_price_max` empirically.
- Two reeval watchers (force_close=0 + 120) per
  `memory/project_force_close_train_vs_deploy.md`.
- Phase 2a and 2b will be committed separately so variables
  are separable for analysis.
**Outstanding for this phase:** commit scaffold with message
`plan(scalping-lay-quality-gate): scaffold next stack-on plan`.
**Next iteration's focus:** Phase 1 — re-run
`tools/probe_lay_outcome_distribution.py` on 2026-04-28/29/30
to set Phase 3 defaults.

## 2026-05-13 22:31 — Phase 1, iteration 1

**State entering iteration:** Phase 0 scaffold committed
(`d8edc53`).
**Work done:**
- Ran `python -m tools.probe_lay_outcome_distribution
  --days 2026-04-28 2026-04-29 2026-04-30
  --race-confidence-threshold 0.50 --lay-threshold 0.40
  --device cuda` (output `/tmp/layq/probe_2026-04-28_30.txt`).

**Probe output (key tables):**

OVERALL: 1173 gate-eligible tuples, EV/£ stake = **−£0.0350**
(matches 2026-05-13 baseline; profile has NOT shifted).

LAY-PRICE BUCKETS:

| Bucket | n | lay_winrate | EV/£stake | avg_loss_when_lost |
|---|---:|---:|---:|---:|
| 2-5 | 77 | 80.5% | +£0.1727 | −£3.25 |
| 5-10 | 287 | 86.4% | −£0.0293 | −£6.57 |
| 10-20 | 310 | 93.2% | +£0.0097 | −£13.62 |
| 20-50 | 308 | 95.5% | **−£0.3896** | **−£29.57** |
| > 50 | 191 | 99.0% | +£0.3717 | −£59.00 |

PREDICTOR-CALIBRATION CHECK:

| pwin band | n | realised | avg_lay_price | EV/£stake |
|---|---:|---:|---:|---:|
| 0.00-0.10 | 866 | 5.9% | 56.31 | −£0.0801 |
| 0.10-0.20 | 205 | 10.7% | 7.91 | **+£0.3741** |
| 0.20-0.30 | 80 | 20.0% | 6.94 | **−£0.7362** |
| 0.30-0.40 | 22 | 9.1% | 8.15 | +£0.4773 |

**Decisions made (Phase 3 defaults locked):**

- `lay_price_max = 20` — drops the 20-50 leverage trap
  (n=308, EV −£0.39). The 5-10 bucket sits at −£0.029
  which is within the §3 −£0.05 bar; the > 50 bucket
  (n=191, EV +£0.37) looks attractive but rests on a 99%
  win rate at avg leverage £59 per loss — a 1% shift in
  win rate would crater it. Capping at 20 keeps the
  reliable buckets (2-5, 5-10, 10-20) and drops the
  documented bleed bucket.
- `predictor_p_win_lay_threshold = 0.20` — drops the
  0.20-0.30 calibration hole (n=80, EV −£0.74) and the
  0.30-0.40 outlier (n=22, too small to trust). At
  threshold 0.20 the blended EV is approximately
  (866 × −£0.08 + 205 × +£0.37) / 1071 ≈ **+£0.006/£**
  across n=1071 (down from 1173).

**Sanity check on the "lowest bucket where EV ≥ 0 and
n ≥ 100" rule:**
- threshold 0.10 → admits 866 at EV −£0.08 → negative
- threshold 0.20 → admits 1071 at EV ~+£0.006 → positive ✓
- threshold 0.30 → admits 1151 at EV −£0.046 → back
  negative
- threshold 0.40 (current) → admits 1173 at EV −£0.035

→ 0.20 is uniquely optimal.

**Cross-intersection note:** Phase 4 smoke will measure the
EV of the actual admitted set (pwin ≤ 0.20 AND
lay_price ≤ 20). The per-bucket marginals above suggest
this set will sit at EV ≥ 0; the smoke verifies.

**Tests run:** none (probe + analysis only).
**Outstanding for this phase:** commit Phase 1 with
`autonomous_run_log.md` updated + probe output reference.
**Next iteration's focus:** Phase 2a — wire per-bet logging
during training-eval rollouts.

## 2026-05-13 22:46 — Phase 2a, iteration 1

**State entering iteration:** Phase 1 committed (`f200d6f`);
defaults `predictor_p_win_lay_threshold = 0.20`,
`lay_price_max = 20`.
**Work done:**
- Extended `EvaluationBetRecord` (registry/model_store.py)
  with 4 nullable fields: `stop_close`, `runner_champion_p_win`,
  `race_max_pwin`, `final_outcome`.
- Updated `write_bet_logs_parquet` to write the new columns.
- Added `_build_eval_bet_records` helper in
  `training_v2/cohort/worker.py` mirroring v1's
  `training/evaluator.py` pattern.
- Wired bet-log capture into `train_one_agent`'s eval-day loop:
  records captured pre-overwrite of `eval_shim`; written via
  `model_store.write_bet_logs_parquet` after
  `create_evaluation_run` returns `run_id`.
- 10 new tests in `TestPerBetLogCapture` (final_outcome
  categorisation, predictor-context capture, parquet round-trip).
- Patched brittle `test_parquet_schema_correct` to include the
  4 new columns.
**Tests run:** 19/19 in v2_cohort_worker (fast) + 60/60 in
model_store + integration tests untouched. All pass.
**Decisions made:**
- Parquet writer + `EvaluationBetRecord` re-used (vs writing
  a new JSONL writer per the driver's nominal schema). The
  v1 path already covers most fields and is the v2 stack's
  natural choice; adding bespoke JSONL would duplicate the
  per-bet-detail infrastructure for no benefit.
- `final_outcome` derived per-pair at write time
  (matured / agent_closed / force_closed / stop_closed /
  naked / directional) so post-hoc joins to scoreboard.jsonl
  can answer "what happened to this pair" without re-running
  the categorisation logic.
**Outstanding for this phase:** commit Phase 2a, then Phase 2b.
**Next iteration's focus:** Phase 2b — extend
SCALPING_POSITION_DIM by 4 leverage/close-cost obs fields.

## 2026-05-13 22:50 — Phase 2b, iteration 1

**State entering iteration:** Phase 2a committed (`ac6cff1`).
**Work done:**
- Bumped `SCALPING_POSITION_DIM` from 4 to 8 in
  `env/betfair_env.py`.
- Implemented 4 new per-runner features in
  `_get_position_vector`:
  `naked_downside_if_runner_wins`,
  `naked_downside_if_runner_loses`, `cost_to_close_now`,
  `worst_case_naked_pnl`.
  Naked legs = matched bets with `pair_id` whose passive
  partner is still in `bm.passive_book.orders`. All four
  normalised by `starting_budget` and clipped to [-1, 1].
- 7 new tests in
  `TestLeverageObsFeatures::test_betfair_env.py` (all 7
  driver-mandated names + cross-load failure guard).
- Updated `test_obs_space_grows_when_scalping` in
  `test_forced_arbitrage.py` from 58 → 114 extra dims.
**Tests run:** 292/292 in env / forced_arbitrage / v2 cohort
worker / model_store / bc_pretrainer pass.
**Decisions made:**
- Architecture-hash break is correct-by-default; pre-plan
  weights cannot cross-load via `load_state_dict(...,
  strict=True)`. Mirrors `fill_prob_in_actor` /
  `mature_prob_in_actor` pattern.
- `cost_to_close_now` uses the simple price-difference form
  `S * (P_back - LTP)` (not the equal-profit hedge sizing) so
  the feature stays interpretable as "price drift since entry".
**Outstanding for this phase:** commit Phase 2b, then Phase 3.
**Next iteration's focus:** Phase 3 — `lay_price_max` env
kwarg + plumbing.

## 2026-05-13 22:56 — Phase 3, iteration 1

**State entering iteration:** Phase 2b committed (`a9007e1`).
**Work done:**
- Added `lay_price_max: float = 0.0` kwarg to
  `BetfairEnv.__init__`. Validation: in [0, 1000]; loud-fail
  if > 0 but `use_race_outcome_predictor = False`.
- `_lay_price_cap_active` flag short-circuits the mask path
  when disabled (byte-identical when off).
- `agents_v2/action_space.compute_mask` clears OPEN_LAY when
  cap active and `runner.last_traded_price > lay_price_max`.
  Asymmetric — OPEN_BACK and CLOSE untouched.
- Plumbed through `training_v2/cohort/{worker,runner}.py` and
  `tools/reevaluate_cohort.py`. CLI flag `--lay-price-max`
  visible in `python -m training_v2.cohort.runner --help`.
- 6 new tests in `TestLayPriceCapGate` mirroring
  `TestRaceConfidenceGate`.
**Tests run:** 139/139 in agents_v2_action_space +
betfair_env + v2_cohort_worker pass.
**Decisions made:**
- Cap composes additively with the pwin lay threshold and
  race-confidence gate (locked semantics per
  hard_constraints.md §12).
- Cap reads CURRENT LTP at the decision tick (not off-N
  proxy) so the policy can adapt to live price drift.
**Outstanding for this phase:** commit Phase 3, then
Phase 4 smoke.
**Next iteration's focus:** Phase 4 — write
`tools/smoke_lay_quality_gate.py` and run on 2026-05-04.

## 2026-05-13 23:25 — Phase 4, iteration 1 — STOP

**State entering iteration:** Phase 3 committed (`bebdff2`).
**Work done:**
- Wrote `tools/smoke_lay_quality_gate.py` mirroring
  `tools/smoke_race_confidence_gate.py` shape. Adds the 4th
  §3 EV threshold: lay-side EV per £1 stake on the
  gate-admitted set (LTP at off-30s proxy, mirrors the Phase
  1 probe's methodology).
- Ran the smoke on 2026-05-04 with the locked Phase 3
  defaults (`race_confidence_threshold=0.50`,
  `predictor_p_win_lay_threshold=0.20`, `lay_price_max=20`).

**Smoke result:**

| Threshold | Bar | Actual | Verdict |
|---|---|---|---|
| `race_qualification_rate` | ≥ 30% | 55.08% | PASS |
| `legal_ratio` | ≤ 80% | 63.30% | PASS |
| **`expected_per_£_lay_EV`** | **≥ −£0.05** | **−£0.2361** | **FAIL** |
| `bets_matched` (full day est.) | ≥ 50 | 3854 | PASS |

OVERALL: **FAIL — STOP loop**.

Cohort NOT launched. Per `hard_constraints.md §3` and §10.2,
the smoke EV threshold is load-bearing — "if the gate-tuned
admitted set isn't +EV (or close), the lay-quality-gate
hypothesis is wrong."

**Smoke detail (2026-05-04, 118 races, full gate config):**

- 327 (race, runner) tuples admitted by the FULL gate.
- Lay win rate: 91.74% (8.26% of admitted runners actually
  won the race).
- Avg lay price (LTP at off-30s on admitted set): 15.99 —
  near the cap, consistent with the cap doing material work.
- 27 losers (admitted runners that won the race) had avg
  loss-price ≈ 14, leveraging the rare loss to dominate the
  small per-bet gain.

**Cross-check vs Phase 1 held-out probe** (same probe
methodology, different window):

| | Held-out 2026-04-28/29/30 (OLD gate, lay 0.40) | Smoke 2026-05-04 (NEW gate, lay 0.20 + cap 20) |
|---|---|---|
| n | 1173 | 327 |
| lay winrate | 92.2% | 91.74% |
| avg lay price | 43.58 | 15.99 |
| EV/£ stake | **−£0.0350** | **−£0.2361** |

**Why the disagreement:**

1. **Sample size variance.** n=327 with 27 losers carries
   a ~£0.22 1-sigma error on EV/£ stake (back-of-envelope:
   √(0.08 × 0.92 × E[(P-1)²]) / √n with mean P-1 ≈ 14).
   The observed −£0.236 is within 1 sigma of the held-out
   probe's −£0.035 across all three days. Statistically
   NOT significantly different from the held-out probe.

2. **Phase 1 marginal-based estimate was likely optimistic.**
   The Phase 1 commit estimated post-cap EV at ~+£0.006 by
   compositing pwin-band marginals and lay-price-bucket
   marginals as if independent. They aren't — pwin and
   lay-price are negatively correlated (low pwin → high
   implied price → high lay price). Tightening pwin to 0.20
   AND capping price at 20 keeps a specific subset whose
   joint distribution may differ from the per-axis
   marginals.

3. **Day-specific calibration.** 2026-05-04 was the
   predecessor cohort's training-eval window, not the
   held-out window. The predictor's lay-side calibration can
   vary day-to-day; the held-out probe averaged over 3
   days, the smoke captures 1.

**Decisions made:**

- STOP the loop per the driver's hard-constraint discipline.
- Surface the diagnostic in this log so the operator can
  decide the next step from informed evidence rather than a
  binary "smoke failed, what now?"
- Do NOT lower the EV threshold mid-flight. Do NOT change
  Phase 3 defaults mid-flight. Either of those is a new plan,
  not an in-loop tweak.

**Tests run:** smoke tool — runs to completion, exit 1 on
FAIL as designed.

**Outstanding for this phase:** none — the smoke runs cleanly
and the verdict is FAIL. The plan exits at this gate.

**Recommended next plan (operator's call):**

A. **Multi-day smoke** — re-run the smoke across 2026-05-04,
   05, 06 (the predecessor's full training-eval window) to
   get n~1000 and a tighter EV estimate. If the 3-day
   aggregate EV is within −£0.05 of zero, the single-day
   smoke was just sample noise — launch the cohort with the
   current defaults.

B. **Re-probe with new caps applied** — re-run
   `probe_lay_outcome_distribution.py` but with
   `--lay-threshold 0.20` AND a new `--lay-price-max` flag
   so the probe measures the FULL gate's admitted-set EV
   on the held-out window. This gives the cleanest
   load-bearing evidence: does the NEW gate produce a +EV
   admitted set on the SAME data the cohort will be
   evaluated against?

C. **Open `scalping-lay-quality-gate-v2`** — accept the
   smoke verdict and design a different fix. Options:
   - Tighten further (e.g. lay 0.15 + cap 15).
   - Add a per-race lay-side filter (only lay in races where
     a specific shape, e.g. one-favorite races, holds).
   - Add a per-tick book-quality filter (refuse OPEN_LAY
     when the book is thin / volatile).

C is the most expensive (new plan, full re-implementation
loop). A and B are cheap follow-up probes that can run in
~30 min total. Either A or B is the recommended starting
point for the next iteration of this plan.

**Loop terminated.** Phase 4 smoke FAIL; cohort NOT
launched.
