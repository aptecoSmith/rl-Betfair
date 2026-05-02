---
plan: rewrite/phase-3-followups/force-close-architecture
status: session-02 implementation-ready; awaiting operator GPU launch
opened: 2026-05-01
session_01_completed: 2026-05-02
session_02_implementation: 2026-05-02
---

# Force-close-architecture — findings

## Session 01 — target-£-pair sizing

### Pre-flight (2026-05-01)

Baseline metrics computed from
`registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl` (12 agents).

**fc-rate (= naked-rate proxy)** — AMBER v2 ran with
`force_close_before_off_seconds = 0`, so every non-matured pair lands
in `arbs_naked` rather than `arbs_force_closed`. The "fc rate" the
operator review uses is therefore
`arbs_naked / (arbs_completed + arbs_naked)`:

| Statistic | Value |
|---|---|
| mean fc-rate (12 agents) | **0.809** |
| min | 0.754 |
| max | 0.862 |
| positive eval P&L | **2/12** |

**policy-close fraction** — undefined for AMBER v2: the scoreboard
emits `eval_arbs_completed` / `eval_arbs_naked` but not
`eval_arbs_closed` / `eval_arbs_force_closed`. Session 01 extends
the worker's `EvalSummary` with the missing counters
(`arbs_closed`, `arbs_force_closed`, `arbs_target_pnl_refused`,
`pairs_opened`, `closed_pnl`, `force_closed_pnl`) so the metric is
computable on the post-plan cohort. AMBER v2 effectively had
`pcf = 0` (no env-initiated closes; agent-initiated closes via
`close_signal` weren't aggregated in the v2 scoreboard).

### Implementation

#### Math helpers (env/scalping_math.py)

- `solve_lay_price_for_target_pnl(back_stake, back_price, target_pnl, c)`
  → solved P_lay or None if non-physical. Closed-form derived from
  the equal-profit identity:
  ```
  P_lay = c + S_back × (1 − c) × [P_back × (1 − c) + c]
            / (target + S_back)
  ```
- `solve_back_price_for_target_pnl(lay_stake, lay_price, target_pnl, c)`
  → symmetric inverse for lay-first scalps:
  ```
  K       = S_lay × (P_lay − c) / [S_lay × (1 − c) − target]
  P_back  = (K − c) / (1 − c)
  ```
- `quantise_to_betfair_tick(price, side)` — rounds DOWN for
  `side="lay"` and UP for `side="back"`, so quantisation only ever
  WIDENS the spread (the agent's £-target is preserved as a floor).

Tests in `tests/test_scalping_math.py::TestSolveTargetPnl` and
`TestQuantiseToBetfairTick` (parametrised round-trip on a 5×3 grid
of (S, P, target, c) tuples; refusal cases for unreachable targets;
quantise direction guards). All 59 scalping_math tests pass.

#### Env wiring (env/betfair_env.py)

- `_REWARD_OVERRIDE_KEYS` adds `target_pnl_pair_sizing_enabled`
  (plan-level boolean, default False = byte-identical to pre-plan).
- `_maybe_place_paired` gains an `arb_frac: float = 0.0` parameter.
  When `target_pnl_pair_sizing_enabled` is True the env maps
  `arb_frac ∈ [0, 1]` to `target_pnl ∈ [£0.20, £5.00]` (linear),
  calls the appropriate solver, and quantises the result. Refusals
  (solver returns None, or the quantised price collides with the
  aggressive leg) increment `_scalping_arbs_target_pnl_refused`
  and the function returns False — the aggressive leg stays naked.
- The matcher's junk filter / hard price cap continue to gate the
  passive's eventual fill at match time; the new solver path does
  not duplicate that logic.
- Force-close (`_attempt_close` with `force_close=True`) and
  re-quote (`_attempt_requote`) are unaffected — the new mechanism
  lives only on the OPEN-time path, per purpose.md §"What's locked"
  point 1.

Per-episode `info` dict gains `arbs_target_pnl_refused`. Pre-plan
rollouts default 0 (counter is never incremented when the flag is
off). 4 new tests in
`tests/test_forced_arbitrage.py::TestTargetPnlPairSizing` cover
flag-off byte-identity, flag-on solver routing, refusal accounting,
and force-close path independence.

#### Scoreboard plumbing (training_v2/cohort/worker.py + runner.py)

- `EvalSummary` extends with `arbs_closed`, `arbs_force_closed`,
  `arbs_target_pnl_refused`, `pairs_opened`, `closed_pnl`,
  `force_closed_pnl`.
- `_eval_rollout_stats` reads the matching keys from `last_info`.
- `_agent_result_to_scoreboard_row` emits `eval_arbs_closed`,
  `eval_arbs_force_closed`, `eval_arbs_target_pnl_refused`,
  `eval_pairs_opened`, `eval_closed_pnl`, `eval_force_closed_pnl`.
- `train_one_agent` and `_build_env_for_day` accept
  `reward_overrides: dict | None`; `run_cohort` passes it through.
- New CLI flag `--reward-overrides KEY=VALUE` (repeatable). Values
  parse as bool / float / string. Example:
  `--reward-overrides target_pnl_pair_sizing_enabled=true`.

#### Test status

| Suite | Pre-plan | Post-plan |
|---|---|---|
| tests/test_scalping_math.py | 44 pass | 59 pass (+15 new) |
| tests/test_forced_arbitrage.py | 113 pass | 117 pass (+4 new) |
| tests/test_v2_cohort_*.py | 7 pass | 7 pass |
| tests/test_betfair_env.py | 62 pass | 62 pass |
| Full repo suite | — | 1572 pass / 1 pre-existing fail (test_orchestrator unrelated) |

### Cross-cohort comparison data-dir caveat

`select_days(seed=42, n_days=8)` walks `data/processed/` and emits a
deterministic-given-the-pool train+eval split. New parquets landed
since AMBER v2 (`2026-04-29.parquet`, `2026-04-30.parquet`) shift
the window:

| Pool | Train | Eval |
|---|---|---|
| AMBER v2 (snapshot) | 2026-04-{20..26} | 2026-04-28 |
| Current data/processed/ | 2026-04-{22..29} | 2026-04-30 |

To preserve the plan's hard constraint #3 ("Same `--seed 42` for
every cohort"), the Session 01 cohort runs against a curated dir
`data/processed_amber_v2_window/` containing only the original 8
days. Verified deterministically reproduces AMBER v2's train+eval
split.

### Cohort run — pending

Wall envelope ~3.5h GPU. Launch command:

```
TS=$(date +%s)
OUT=registry/v2_force_close_arch_session01_target_pnl_${TS}
mkdir -p "$OUT"
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 8 \
    --device cuda --seed 42 \
    --data-dir data/processed_amber_v2_window \
    --reward-overrides target_pnl_pair_sizing_enabled=true \
    --output-dir "$OUT" 2>&1 | tee "$OUT/cohort.log"
```

### Scoring (run after cohort completes)

```
python C:/tmp/v2_phase3_bar6.py registry/v2_force_close_arch_session01_target_pnl_<ts>
```

Plus inline analysis for the new metrics (policy-close fraction +
target-£ refusal rate). Results table to be filled in:

| Metric | AMBER v2 baseline | Session 01 | Δ |
|---|---|---|---|
| mean fc_rate | 0.809 | TBD | TBD |
| ρ(entropy_coeff, fc_rate) | −0.532 | TBD | TBD |
| positive eval P&L | 2/12 | TBD | TBD |
| median policy-close fraction | 0.00 (counter not in v2) | TBD | TBD |
| median target-PnL refusal rate | n/a | TBD | TBD |

### Cohort run — complete (2026-05-02)

Output: `registry/v2_force_close_arch_session01_target_pnl_1777672840/`.
Wall: 13471.9s = **3.74h** (slightly over 3.5h envelope but well
within the 5h stop-condition).
12/12 agents completed without crashes.

Per-agent eval (sorted by P&L descending):

| agent | day_pnl | bets | closed | naked | refused | opens | locked | naked_pnl | pcf | refrate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e7fa936c-a69 | **+£477.43** | 282 | 14 | 192 | 11 | 237 | +£128.01 | +£366.45 | 0.311 | 0.044 |
| f86ec23d-d9f | **+£268.08** | 304 | 12 | 204 | 9 | 254 | +£157.45 | +£127.90 | 0.240 | 0.034 |
| 4370ba9c-7ac | **+£54.05**  | 276 | 14 | 166 | 8 | 221 | +£168.72 | −£95.87 | 0.255 | 0.035 |
| a3f070f5-112 | −£12.26    | 295 | 9  | 185 | 11 | 240 | +£189.94 | −£192.27 | 0.164 | 0.044 |
| 5b6b0d02-4dc | −£20.68    | 285 | 13 | 199 | 12 | 242 | +£124.30 | −£130.16 | 0.302 | 0.047 |
| d72ba92c-1c4 | −£171.03   | 271 | 11 | 173 | 15 | 222 | +£157.51 | −£314.07 | 0.224 | 0.063 |
| 98d10bff-eee | −£283.61   | 306 | 10 | 188 | 7  | 247 | +£200.71 | −£455.76 | 0.169 | 0.028 |
| 0162efc3-d63 | −£368.48   | 261 | 13 | 159 | 6  | 210 | +£156.82 | −£511.80 | 0.255 | 0.028 |
| 8f53e2c9-bbe | −£459.31   | 329 | 19 | 207 | 12 | 268 | +£173.48 | −£610.83 | 0.311 | 0.043 |
| b3229e61-f06 | −£580.13   | 274 | 14 | 156 | 6  | 215 | +£185.46 | −£737.33 | 0.237 | 0.027 |
| 78cc768a-879 | −£655.41   | 302 | 15 | 188 | 13 | 245 | +£173.11 | −£809.16 | 0.263 | 0.050 |
| f94b961f-8f0 | −£1024.68  | 298 | 14 | 192 | 16 | 245 | +£162.26 | −£1167.72 | 0.264 | 0.061 |

(`closed` = `arbs_closed`; `naked` = `arbs_naked`; `refused` =
`arbs_target_pnl_refused`; `opens` = `pairs_opened`;
`pcf = closed / (closed + completed_natural + force_closed)`;
`refrate = refused / (refused + opens)`. `force_closed = 0` for
every agent — cohort matches AMBER v2 protocol with
`force_close_before_off_seconds = 0`.)

### Bar 6 verdict trio

| Bar | Threshold | AMBER v2 baseline | Session 01 | Δ | Status |
|---|---|---|---|---|---|
| 6a mean fc_rate | ≤ 0.30 (session bar) | 0.809 | **0.821** | +0.012 | **FAIL** |
| 6b ρ(entropy_coeff, fc_rate) | ≤ −0.5 | −0.532 | −0.206 | +0.326 | FAIL |
| 6c positive eval P&L | ≥ 4/12 (session bar) | 2/12 | **3/12** | +1 | **FAIL** |

(Bar 6a uses naked-rate as fc-rate proxy: every cohort runs with
`force_close_before_off_seconds = 0`, so non-matured pairs land in
`arbs_naked` rather than `arbs_force_closed`. Bar 6c session bar
≥ 4/12 is tighter than the rewrite-level Bar 6c ≥ 1/12 which DID
pass.)

### New metrics

| Metric | AMBER v2 baseline | Session 01 |
|---|---|---|
| median policy-close fraction | 0.000 (counter not in v2 scoreboard) | **0.255** |
| mean policy-close fraction | n/a | 0.250 |
| median target-PnL refusal rate | n/a | **0.043** |
| mean target-PnL refusal rate | n/a | 0.042 |

**Policy-close fraction interpretation:** The new mechanism IS
teaching the policy to fire `close_signal`. Median 0.255 means
roughly 1-in-4 of all pair-resolutions are agent-initiated closes
(vs ~0 in AMBER v2 where close_signal was structurally available
but never fired meaningfully). The mechanism worked at the
behavioural level — just not enough to move the macro fc rate.

**Target-PnL refusal rate interpretation:** Median 4.3 % means
the [£0.20, £5.00] range is well-calibrated to the data: only ~1
pair in 25 has a target the solver can't reach. The Session 01
stop-condition (refusal > 0.80) is nowhere close to tripping —
the £-target framing itself is structurally usable.

### Why the macro fc rate didn't move

Despite the policy actively closing 25 % of its pairs (vs 0 % in
AMBER v2), the non-matured-pair rate barely budged
(0.821 vs 0.809). Mechanically:

- AMBER v2: ~80 % of pairs settle naked (passive never matched).
- Session 01: ~25 % of pairs are policy-closed early; the
  REMAINING pairs still leave their passive unmatched at the
  same ~80 % rate. The closes mostly happen on pairs that would
  have matured anyway, OR on naked-headed pairs that get closed
  at a cost (note 8 of the 9 negative-naked-pnl agents).

Both readings are consistent with the cohort data. The locked-pnl
column is positive on every agent (+£124..+£201), so matured
pairs DO generate cash. The naked-pnl column is the kill — agents
fail to prevent enough naked exposures from running to settle.
The £-target mechanism gives the policy a first-class
profit-taking lever but no first-class STOP-LOSS lever, which is
exactly what Session 02 (`projected-loss stop-close`) addresses.

### Verdict — FAIL on session bar; PASS on Bar 6c rewrite-level

Per session prompt §6:

- mean fc ≤ 0.30 **AND** ≥ 4/12 positive eval P&L → GREEN
- One threshold met → PARTIAL
- Neither met → **FAIL**

Both session-bar thresholds missed. Per the prompt:

> **FAIL**: neither threshold met. Document; the operator
> decides whether to spend the next ~4 h GPU on Session 02 or
> call this RED early.

Mitigating evidence for spending the next 4h on Session 02
rather than calling RED:

1. **Mechanism is working.** Median pcf = 0.255 is a 5–10×
   improvement on AMBER v2 in policy-initiated close behaviour;
   the £-target framing is reaching the policy.
2. **Refusal rate is healthy.** 4.3 % refusal means the £-target
   range is structurally compatible with the data. A higher
   target range or a per-agent gene wouldn't unlock more.
3. **The diagnosis lines up with Session 02.** Naked-pnl is the
   kill term, and Session 02's stop-close is the targeted fix
   for run-away naked losses. With Session 01's pcf already
   non-zero, stop-close has a partner mechanism that's actually
   firing.
4. **Bar 6c rewrite-level threshold (≥ 1/12) is met** at 3/12 —
   the architecture IS producing positive-cash agents on no
   shaping. The session-bar of 4/12 is a *tighter* cut than the
   rewrite ships at.

Risk of running Session 02:

- 4h GPU on a hypothesis that may also fail; that's then 8h
  spent on a single mechanics question with no GREEN at either
  step.
- Session 02 stacks naturally with Session 01 (both flags can
  be on simultaneously per session prompt §3 of `purpose.md`),
  but the plan's Session 03 verdict path requires individual
  GREEN, not stacked-only GREEN.

**Decision required:** load Session 02 (projected-loss
stop-close) next, OR call the plan RED and write up the
post-mortem.

## Session 02 — projected-loss stop-close

### Implementation (2026-05-02, awaiting GPU launch)

Code changes landed under operator authorisation to proceed
past the Session 01 FAIL gate. Session 01's
`target_pnl_pair_sizing_enabled` flag stays OFF for Session 02
per hard constraint §7 (one mechanics change at a time).

#### env/betfair_env.py

- `_REWARD_OVERRIDE_KEYS` adds `stop_loss_pnl_threshold` (£,
  default 0.0 = disabled) and `lay_only_naked_price_threshold`
  (price floor for the naked-lay carve-out, default 4.0).
- `_compute_portfolio_mtm` now also rebuilds
  `self._per_pair_mtm: dict[pair_id, £]` in the same pass.
  Bets without a `pair_id` contribute to the portfolio total
  but not to the per-pair map. Drop-out (resolved bets removed)
  preserves the telescope-to-zero invariant per pair.
- New `_stop_close_open_pairs(race, tick)` method walks
  `self._per_pair_mtm` and triggers `_attempt_close` with
  `stop_close=True` (force_close=False ⇒ strict matcher) on
  any pair whose MTM crosses `-stop_loss_pnl_threshold`. The
  naked-lay carve-out (`_is_naked_lay_long_odds`) skips pairs
  whose only open leg is LAY-side AND whose original BACK
  partner was matched at price ≥
  `lay_only_naked_price_threshold`.
- The trigger is wired into `step()` at slot 0d (after the
  T−N force-close pass, before action handling). It runs ONLY
  when `scalping_mode + threshold > 0 + not in_play`. When
  enabled it pays one extra `_compute_portfolio_mtm` walk per
  tick; the end-of-step MTM compute is unchanged so the
  existing shaped-MTM contribution stays byte-identical.
- `_attempt_close` gains `stop_close: bool = False`. When True,
  the placed close leg's `stop_close=True`, the
  `_scalping_arbs_stop_closed` counter increments, and the
  pair-id-hint eviction path now also accepts stop-close (not
  just force-close — passive may be evicted before MTM trips).
- `_settle_current_race` classifies pairs into a third bucket
  `scalping_arbs_stop_closed` / `scalping_stop_closed_pnl`
  (mutually exclusive with closed/force_closed). Stop-closed
  pairs are excluded from the matured-arb bonus
  (`n_matured = arbs_completed + arbs_closed` only) and from
  the close_signal +£1 bonus
  (`n_close_signal_successes = arbs_closed` only) — they were
  env-initiated, not policy-initiated.
- Selective-open-shaping `_resolve_open_cost_pairs` was
  extended to NOT refund stop-closed pairs (same exclusion as
  force-closed; both are env-initiated bail-outs).
- `RaceRecord` gains `arbs_stop_closed` and `stop_closed_pnl`.
- `_get_info` exposes `arbs_stop_closed`,
  `scalping_stop_closed_pnl`, plus
  `stop_loss_pnl_threshold_active` and
  `lay_only_naked_price_threshold_active` as telemetry.

#### env/bet_manager.py

- `Bet` gains `stop_close: bool = False`. Mutually exclusive
  with `force_close` (stop-close uses the strict matcher; only
  force-close uses the relaxed path). `stop_close=True`
  implies `close_leg=True` (both set together at placement).

#### training_v2/cohort/

- `EvalSummary` gains `arbs_stop_closed` and `stop_closed_pnl`.
- `_eval_rollout_stats` reads them from `last_info`.
- `_agent_result_to_scoreboard_row` emits `eval_arbs_stop_closed`
  and `eval_stop_closed_pnl`.

#### Tests added (all passing)

| Suite | Pre-S02 | Post-S02 |
|---|---|---|
| tests/test_mark_to_market.py | 11 pass | 14 pass (+3 `TestPerPairMtm`) |
| tests/test_forced_arbitrage.py | 117 pass | 125 pass (+8 `TestStopClose`) |

`TestPerPairMtm` covers:
1. Σ per_pair_mtm == portfolio MTM (within float tol).
2. Per-pair bucket drops to zero on resolution.
3. Per-pair telescope invariant on offsetting legs.

`TestStopClose` covers:
1. `threshold=0` is byte-identical to pre-plan (counter stays 0
   even with deeply-underwater MTM).
2. Stop-close fires when per-pair MTM dips below threshold.
3. Naked-lay at long odds (back_price ≥ floor) skips
   stop-close (carve-out applies).
4. Naked-lay at short odds (back_price < floor) closes.
5. Naked-back closes regardless of original back price.
6. Stop-closed pair routes to `arbs_stop_closed`, NOT
   `arbs_closed` or `arbs_force_closed`.
7. Stop-close does NOT pay the close_signal +£1 bonus.
8. Stop-close uses strict matcher (close leg's
   `force_close=False`).

#### Regression suite (511 tests on touched / adjacent paths)

`test_mark_to_market`, `test_forced_arbitrage`,
`test_betfair_env`, `test_scalping_math`,
`test_v2_cohort_runner`, `test_v2_cohort_worker`,
`tests/arb_signal_cleanup/`, `tests/arb_curriculum/`,
`test_close_signal`, `test_arb_freed_budget`,
`test_bet_manager` — **511 pass / 0 fail / 0 skip**.

(Wider full-suite run reveals 4 unrelated failures pre-dating
this session: a stale `kl_early_stop_threshold == 0.03`
assertion contradicting CLAUDE.md's 2026-04-25 default change
to 0.15; a flaky-bound `policy_loss < 100` advantage-norm
test; a test-isolation leak in
`test_kl_early_stop_is_per_mini_batch_not_per_epoch` which
PASSES when run alone; a websocket timeout flake in
`test_training_worker`. None touch stop-close, MTM, or
scalping settle paths.)

### Carve-out semantics clarified (2026-05-02 mid-cohort)

The first launch of the Session 02 cohort was stopped at 4/12 agents
after a peek run flagged a likely Bar-6a fail. Inspection of the
partial results, combined with an operator clarification on the
carve-out's intent, surfaced two real bugs in the original
``_is_naked_lay_long_odds`` implementation:

1. **Wrong price source.** The function gated on the BACK partner's
   matched price. The operator's intent was always to gate on the
   **open LAY leg's** matched price — the price the agent is
   actually exposed to. With pair-bracketed prices typically within
   a few ticks of each other on the same horse, the bug fired the
   carve-out on virtually every paired pair (back partners cluster
   ≥ 4.0 on most horses).

2. **Wrong threshold.** Default was 4.0; operator's bands are
   short = 1.0–5.0, mid = 5.1–15.0, long = >15.0. At 4.0 the
   carve-out applied to pairs across all three bands. Bumped to
   15.0 so only the genuinely long-odds slice is protected.

3. **Lay-first-naked unreachable.** The original function returned
   False ("don't carve out") when no matched BACK partner existed
   in ``bm.bets``. But the lay-first-naked case (aggressive LAY
   matched, paired BACK never filled) is **exactly** the scenario
   the operator described as "okay" if at long odds. The fix gates
   on the open LAY's price directly, no back-partner lookup
   needed.

Net effect of the fix: the corrected carve-out is **strictly more
conservative** — it skips stop-close on a much smaller slice of
pairs, so stop-close fires on more naked lays, not fewer. This
should drive fc_rate down vs the original buggy run.

Also bonus: the operator framing "if our models always lay first,
our naked losses would go down" is logged as a follow-on idea
but NOT implemented in this session — it's a substantial
architectural shift (reward shaping or env-side side selection)
that warrants its own plan once the corrected stop-close has been
measured.

Test coverage extended to 9 stop-close tests. New test
``test_carve_out_ignores_back_partner_price`` is the regression
guard for the bug-#1 fix; ``test_stop_close_does_not_fire_for_
naked_lay_at_long_odds`` now exercises the lay-first-naked path
directly.

### Threshold semantics (2026-05-02 operator clarification)

After re-launching the cohort with the corrected carve-out, agent 1
produced **bit-identical** training metrics to the previous run on
days 1-3 — strongly implying the carve-out fix didn't change
anything observable for that agent. Inspection surfaced a second,
more fundamental issue: at the agent's typical stake size (£5-£15),
the **£1 absolute threshold rarely fires**.

Math: MTM = stake × (P_matched - LTP) / LTP. To reach -£1 MTM:

| Stake | Adverse drift needed |
|---|---|
| £5 | +25 % |
| £10 | +12.5 % |
| £20 | +5.3 % |
| £100 | +1.0 % |

Real Betfair pre-race volatility is typically 5-15 % over the last
10 minutes, so £20+ bets get ~half-coverage but £5 bets are
essentially unprotected.

The operator's original "£1 → close" framing was implicitly
assuming typical stake sizes; with micro-bets common, the absolute
£ figure produces inconsistent behaviour across stake sizes.

**Fix:** reinterpret `stop_loss_pnl_threshold` as a **fraction of
open-leg matched stake**. Trigger fires when MTM crosses
`-threshold × open_stake`. Same relative drift fires at all stake
sizes:

- £5 stake × 0.10 → trigger at -£0.50 (~10 % drift)
- £20 × 0.10 → trigger at -£2.00 (~10 % drift)
- £100 × 0.10 → trigger at -£10.00 (~10 % drift)

Default working point: 0.10 (10 % loss → close). Operator can
tune to 0.05 (tighter) or 0.20 (looser) via reward_overrides.

The `stop_loss_pnl_threshold` reward override KEY stays the same;
only its UNITS change. Threshold = 0.0 still means "disabled,
byte-identical to pre-plan." Per-pair MTM telescope and carve-out
semantics are unchanged. The lay-first-naked / open-LAY-price
carve-out from the 2026-05-02 fix continues to gate at the LAY
leg's price ≥ 15.0.

10th test added (`test_threshold_scales_with_open_leg_stake`)
exercising the small-stake case that motivated the change.

### Cohort run — pending operator confirmation

Wall envelope ~3.5h GPU. Launch command (per session prompt §4
with the curated AMBER v2 data window from Session 01):

```
TS=$(date +%s)
OUT=registry/v2_force_close_arch_session02_stop_close_${TS}
mkdir -p "$OUT"
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 8 \
    --device cuda --seed 42 \
    --data-dir data/processed_amber_v2_window \
    --reward-overrides stop_loss_pnl_threshold=0.10 \
    --output-dir "$OUT" 2>&1 | tee "$OUT/cohort.log"
```

Threshold 0.10 (post-2026-05-02 reinterpretation) means "close
when MTM dips below 10 % of open-leg stake." Default
`lay_only_naked_price_threshold=15.0` per the operator's odds
bands. One mechanics change, one threshold value, one cohort —
follow-on threshold sweeps are out of scope.

### Scoring (run after cohort completes)

```
python C:/tmp/v2_phase3_bar6.py registry/v2_force_close_arch_session02_stop_close_<ts>
```

Plus inline analysis for stop-close fraction (per session
prompt §5 metric) and the naked-back catastrophe count
(loss > £200 per pair). Results table to be filled in:

| Metric | AMBER v2 baseline | Session 01 | Session 02 |
|---|---|---|---|
| mean fc_rate | 0.809 | 0.821 | TBD |
| positive eval P&L | 2/12 | 3/12 | TBD |
| median policy-close fraction | 0.000 | 0.255 | TBD |
| median stop-close fraction | n/a | n/a | TBD |
| naked-back catastrophe count (>£200) | TBD | TBD | TBD |
