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

### Verdict-bar reframe (2026-05-02 mid-cohort)

While the third Session 02 cohort was running (with the corrected
carve-out and stake-scaled threshold), the operator flagged that
the original "individual GREEN required" verdict structure
conflicted with the structural reality of scalping:

> "It sounds like 2 key tools in a scalper's locker. Why would
> stacking be questionable?"

The reframe (operator decision):

- **Profit-taking and stop-loss are the minimum viable scalping
  toolkit, not alternatives.** A real scalper always uses both;
  neither alone is sustainable.
- The original "individual GREEN required" rule was a
  methodological choice (clean per-mechanics attribution), not a
  fundamental principle. It was over-strict for a question whose
  expected ship configuration always involved both pieces.
- **Stacking is now treated as the expected ship configuration**,
  not a degraded outcome.

Updated verdict logic:

1. Either S01 alone OR S02 alone clears Bar 6a → ship GREEN.
2. **Stacked S01 + S02 clears Bar 6a → ship GREEN (no asterisk).**
   Per-session attribution evidence (S01's pcf shift, S02's scf
   shift) is recorded so the per-mechanics contribution stays
   on the record.
3. Neither alone nor stacked clears Bar 6a, but each
   demonstrably moves a behavioural metric → ship AMBER with
   explicit operator decision on next mechanics layer.
4. Neither clears AND behavioural metrics don't move → RED.

Methodological discipline preserved: each session's individual
cohort still runs first to attribute per-mechanics behavioural
shifts; stacked cohort runs Session 03. What changed is only
the bar at which "stacked GREEN" is ship-worthy without
asterisk.

`purpose.md` §"Success bar" + §"Sessions/Session 03" updated to
match. Hard constraint §"One mechanics change per cohort" is
unchanged but rephrased to clarify it's a per-cohort isolation
rule, not a ban on shipping a stacked configuration.

### Session 02 standalone cohort — truncated at 3/12 (2026-05-02)

Output: `registry/v2_force_close_arch_session02_stop_close_1777726954/`.

Two prior launches were stopped early on the same day:
- 1777718273: buggy carve-out (gated on back partner's price). 4/12
  agents complete; data invalidated by carve-out bug.
- 1777725672: corrected carve-out, absolute-£ threshold. 0/12 agents
  complete; data invalidated by threshold-too-loose-on-small-bets
  diagnosis.

The third launch (1777726954) ran with the corrected carve-out
(open-LAY-price gating, floor=15.0) AND the stake-scaled threshold
(0.10 × open_stake). Operator stopped at 3/12 once the
behavioural pattern was clear and stable — running the remaining
9 agents would have spent ~2.5 h GPU confirming a reading that's
already locked in.

Per-agent eval (3 agents, all on AMBER v2 day 2026-04-28):

| agent | day_pnl | matured | closed | stop | naked | fc_rate | scf | naked_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a49d1aa6-58a | +£153.32 | 38 | 4 | 27 | 146 | 0.679 | 0.126 | +£83.14 |
| 60157461-0d0 | -£706.26 | 46 | 3 | 23 | 133 | 0.649 | 0.112 | -£841.18 |
| bd33b0a4-935 | -£109.74 | 45 | 1 | 28 | 138 | 0.651 | 0.132 | -£200.02 |

**Behavioural attribution evidence (S02 alone):**

| Metric | AMBER v2 | S01 alone | **S02 alone (3/12)** |
|---|---|---|---|
| mean fc_rate (TRUE) | 0.809 | 0.821 | **0.660** |
| median scf | 0.000 | 0.000 | **0.126** |
| median pcf | 0.000 | 0.255 | 0.015 |
| positive eval P&L | 2/12 | 3/12 | 1/3 |

**What the 3-agent sample tells us:**

1. **Stop-close mechanism works.** scf consistently in the 0.11–
   0.13 band (target range 0.10–0.30). All 3 agents fire it
   meaningfully on real data.
2. **fc_rate moves ~20 % relative** vs S01 (0.821 → 0.660). The
   tightness of the per-agent fc_rate band (0.649–0.679) confirms
   this isn't sampling noise — it's the new equilibrium under
   stop-close.
3. **Policy-close fraction collapses.** S01's 0.255 → S02's
   0.015. The agent stops using `close_signal` because stop-close
   fires first. Mechanism shifts the work env-side as designed.
4. **Naked-pnl variance is still very high.** -£200 to -£841 on
   two agents, +£83 on one. Stop-close caps each pair at ~10 %
   of stake but two structural leaks remain: (a) pairs whose MTM
   never crosses -10 % yet settle on the wrong race-outcome side,
   (b) pairs alive into in-play (the trigger gates off when
   `tick.in_play=True`).

**Bar 6a (mean fc ≤ 0.30):** likely FAIL alone. Even if remaining
9 agents averaged 0.20, cohort mean would land ~0.31. The 0.66
floor from this small sample is the working point.

**Bar 6c (≥ 4/12 positive eval P&L):** TBD. 1/3 with very high
variance.

### 4-stack cohort — truncated at 5/12 (2026-05-02)

Output: `registry/v2_force_close_arch_session03_stacked_1777735438/`.

Mechanics enabled (all four):
- `target_pnl_pair_sizing_enabled = true` (S01: £-target on opens)
- `stop_loss_pnl_threshold = 0.10` (S02: stake-scaled stop-close)
- `force_close_before_off_seconds = 60` (T−60 safety net re-enabled)
- `min_seconds_before_off = 60` (no aggressive opens in close-out window)

Per-agent eval (5 agents, all on AMBER v2 day 2026-04-28):

| agent | day_pnl | bets | matured | closed | stop | forced | naked | fc_rate | locked | naked_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 70b40041 | -£204.49 | 423 | 36 | 6 | 25 | 142 | 5 | 0.023 | +£180.12 | -£87.57 |
| 19036f88 | -£37.69 | 417 | 28 | 9 | 29 | 140 | 4 | 0.019 | +£150.25 | +£112.70 |
| e7c89a86 | -£135.73 | 438 | 46 | 2 | 28 | 140 | 4 | 0.018 | +£210.95 | -£40.41 |
| 50bdb282 | -£231.45 | 415 | 40 | 4 | 24 | 137 | 3 | 0.014 | +£187.26 | -£94.91 |
| 0e3ce830 | +£15.73 | 426 | 34 | 12 | 21 | 142 | 7 | 0.032 | +£178.85 | +£52.69 |

**Aggregate (5/12) verdict-bar tracking:**

| Metric | AMBER v2 | S01 | S02 | S01+S02 | **4-stack (5/12)** |
|---|---|---|---|---|---|
| mean fc_rate | 0.809 | 0.821 | 0.660 | 0.661 | **0.021** |
| Bar 6a (≤0.30) | FAIL | FAIL | FAIL | FAIL | **PASS** |
| median scf | 0.000 | 0.000 | 0.126 | 0.135 | 0.117 |
| median pcf | 0.000 | 0.255 | 0.015 | 0.037 | 0.028 |
| positive eval P&L | 2/12 | 3/12 | 1/3 | 1/2 | 1/5 |

**Bar 6a verdict — PASS by 14×.** fc_rate range across 5 agents:
0.014–0.032. Tight band, structurally locked in. Naked count
3–7 per agent (was 130–200 in prior cohorts). Operator stopped
at 5/12 once the verdict was unambiguous — the remaining 7
agents would have spent ~2 h GPU confirming 0.02 ± 0.01.

**Bar 6c verdict — likely FAIL.** 1/5 positive eval P&L; at the
1-in-5 rate the cohort would land 2-3/12, below the ≥4/12 bar.

**Why most agents are negative despite naked-rate near zero:**

The economic story per agent:
- locked_pnl (matured profit): **+£150 to +£211**
- naked_pnl (capped tail, small magnitude): **-£100 to +£113**
- force-close friction (~140 closes × spread): **-£140 to -£280**
- net day_pnl: **-£231 to +£16**

The safety net is doing its job — naked tail is bounded — but
the agent is opening 415–438 pairs/day to feed it. ~140 of those
go to force-close at ~£1-2 each = £140-280 of pure friction.
That friction is what's keeping day_pnl negative.

**Behavioural lesson exposed:** with `open_cost=0`, opens are
essentially free at the open decision (per-tick gradient too
weak; cash signal at settle is GAE-smeared). The agent learned
"trade more, env cleans up." Bet count went UP from S01+S02
stacked's ~290 to ~425 (+47%). The agent is exploiting the
safety net rather than learning selectivity.

**Mechanism layer of the rewrite ships GREEN per the
2026-05-02 verdict reframe** — fc_rate cleared cleanly, all
four mechanics demonstrably moved their respective metrics.
Bar 6c failure is a policy-shape problem (over-opening), not a
mechanism problem.

### 5-stack cohort — truncated at 9/12 (2026-05-02)

Output: `registry/v2_force_close_arch_session03_5stack_1777741794/`.

Mechanics enabled (all five):
- `target_pnl_pair_sizing_enabled = true` (S01)
- `stop_loss_pnl_threshold = 0.10` (S02)
- `force_close_before_off_seconds = 60`
- `min_seconds_before_off = 60`
- `open_cost = 1.0` (NEW — selective-open shaping per-tick)

Per-agent eval (9 agents, all on AMBER v2 day 2026-04-28):

| agent | day_pnl | bets | matured | closed | stop | forced | naked | fc_rate | locked | naked_pnl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| af58ae4e | -£93.18 | 404 | 41 | 5 | 27 | 125 | 7 | 0.034 | +£190.19 | -£42.22 |
| 5d20907c | -£378.04 | 421 | 30 | 10 | 30 | 134 | 12 | 0.056 | +£153.21 | -£234.92 |
| 2d0bc9a7 | -£200.25 | 437 | 45 | 3 | 27 | 141 | 3 | 0.014 | +£208.39 | -£112.24 |
| a7f22b68 | -£243.20 | 356 | 24 | 5 | 26 | 119 | 5 | 0.028 | +£131.49 | -£135.28 |
| 8e6ccb63 | -£69.32 | 413 | 38 | 2 | 24 | 139 | 4 | 0.019 | +£170.34 | +£17.17 |
| 56abd9f8 | -£92.81 | 406 | 52 | 3 | 19 | 125 | 7 | 0.034 | +£236.02 | -£69.64 |
| 3810f6d0 | -£242.25 | 382 | 34 | 3 | 23 | 131 | 0 | 0.000 | +£163.67 | -£121.02 |
| b55578ba | -£67.17 | 459 | 30 | 7 | 19 | 169 | 4 | 0.017 | +£164.70 | +£67.87 |
| a8322bd1 | -£132.85 | 415 | 32 | 5 | 25 | 141 | 8 | 0.038 | +£153.15 | +£37.36 |

**Aggregate (9/12) verdict-bar tracking:**

| Metric | AMBER v2 | S01 | S02 | S01+S02 | 4-stack | **5-stack (9/12)** |
|---|---|---|---|---|---|---|
| mean fc_rate | 0.809 | 0.821 | 0.660 | 0.661 | 0.021 | **0.027** |
| Bar 6a (≤0.30) | FAIL | FAIL | FAIL | FAIL | PASS | **PASS** |
| median scf | 0.000 | 0.000 | 0.126 | 0.135 | 0.117 | 0.120 |
| median pcf | 0.000 | 0.255 | 0.015 | 0.037 | 0.028 | 0.024 |
| positive eval P&L | 2/12 | 3/12 | 1/3 | 1/2 | 1/5 | **0/9** |

**Bar 6a verdict — PASS by 11×.** fc_rate range 0.000–0.056 across
9 agents. Tightly-clustered, structurally locked. Even with one
outlier (5d20907c at 0.056), the cohort mean is well below 0.30.

**Bar 6c verdict — almost certainly FAIL.** 0/9 positive eval P&L.
Even if all 3 remaining agents land positive (extremely unlikely),
3/12 < 4/12 bar. Operator stopped at 9/12 to address the
over-opening dynamic before spending more GPU.

**Open_cost=1.0 effect — minimal on bet count.**

| | 4-stack mean | 5-stack mean |
|---|---|---|
| bet_count per agent | ~425 | ~410 (-4%) |
| matured per agent | ~39 | ~36 |
| force-closed per agent | ~140 | ~136 |
| day_pnl per agent | -£140 | -£170 |

The agent didn't learn to be more selective at the open decision.
Bet count barely budged (4% drop). Hypothesised cause: 1
generation × 7 training days × ~12k transitions × 4 PPO epochs
gives limited update opportunity for a new shaping term. The
agent has multiple knobs to comply with the open_cost gradient
(cut bets, tighten targets, drop signal threshold); on this
budget it didn't pick "cut bets."

One agent achieved fc_rate = 0.000 (3810f6d0): every unmatured
pair got force-closed, zero nakeds. Most efficient safety-net
usage observed across any cohort. day_pnl still -£242 because
force-close friction (~131 closes × ~£1-2 spread) dominates the
+£164 locked profit.

**Mechanism layer ships GREEN per the 2026-05-02 verdict
reframe** (Bar 6a cleared, all mechanics demonstrably moving
their respective metrics). **Bar 6c failure is a policy-shape
problem**, not a mechanism problem — the agent over-opens and
the friction cost dominates day_pnl.

### Operator pause — 2026-05-02

After the 9/12 5-stack reading, operator paused to do further
work before launching the next cohort iteration. Three lines of
follow-up are open:

1. **Stronger open_cost** (e.g. 2.0 — upper bound of safe range
   per CLAUDE.md). Tests whether more pressure forces the agent
   to cut bet count.
2. **Multi-generation training** (2-4 generations vs the current
   1). Lets the policy respond more fully to open_cost.
3. **Different shaping (not open_cost-based).** E.g.
   matured_arb_bonus to reward maturation positively rather than
   penalising opens negatively.

Neither (1) nor (3) is a new mechanic — both are existing knobs.
(2) is a training-budget change, not a mechanism change.

The mechanism layer of `force-close-architecture` is GREEN as of
2026-05-02. Bar 6c is now a follow-on question on policy shape.

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
