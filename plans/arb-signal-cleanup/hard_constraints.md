# Hard constraints — Arb Signal Cleanup

Non-negotiable rules. Anything that violates one gets
rejected in review.

## Scope

**§1** This plan makes these coordinated changes, in
session order:

1. Force-close mechanism: env-initiated best-available
   market-close on any open position with an unfilled
   second leg once `time_to_off ≤
   force_close_before_off_seconds`. Naked accounting
   handles the residual unpriceable cases.
2. Entropy controller velocity widening: `alpha_lr`
   promoted from hardcoded constant to GA gene with a
   plan-level range.
3. Transformer context-window ceiling raise: widen
   `PPOTransformerPolicy`'s supported
   `transformer_ctx_ticks` range from `{32, 64, 128}`
   to `{32, 64, 128, 256}`. Strictly additive — lower
   values still valid; this adds an option.
4. Shaped-penalty warmup: plan-level
   `shaped_penalty_warmup_eps` that linearly scales
   `efficiency_cost` and `precision_reward` from 0 → 1
   across the first N episodes.
5. Plan redraft + validator update + launch.

Anything NOT in that list is out of scope. Explicit
out-of-scope examples: matcher rewrites, action/obs schema
bumps, new aux heads, controller-architecture changes
(PI/Adam), BC pretrainer changes, matured-arb bonus
formula changes, naked-loss annealing changes, curriculum
day-order changes, adaptive mutation, live-inference work.

**§2** No changes to matcher semantics.
`env/exchange_matcher.py` stays single-price, no-walking,
LTP-filtered. Force-close calls the existing matcher
through the existing aggressive/close code paths; it does
NOT introduce a "close at any price" mode, a new filter
rule, or ladder walking.

**§3** No changes to PPO numerical stability defences.
Ratio clamp ±5 stays. KL early-stop 0.03 stays. Per-arch
LR stays. 5-update LR warmup stays. Advantage normalisation
stays. Reward centering EMA stays.

**§4** The target-entropy controller's *structure* stays
wired as-is. SGD (not Adam) momentum 0. `log_alpha` clamp
`[log(1e-5), log(0.1)]`. `target_entropy = 150`. BC↔
controller handshake via `bc_target_entropy_warmup_eps`.
Only the `alpha_lr` value changes — from hardcoded
constant to per-agent gene. Everything else is untouched.

**§5** MTM shaping stays at config default 0.05. No MTM
changes.

**§6** BC pretrain semantics (per-agent, never shared;
signal + arb_spread heads only; separate optimiser from
PPO Adam) are unchanged. Genes `bc_pretrain_steps`,
`bc_learning_rate`, `bc_target_entropy_warmup_eps` keep
their current ranges and semantics.

**§7** Matured-arb bonus formula is unchanged.
`n_matured = arbs_completed + arbs_closed` with zero-mean
correction. **Force-close completions are NOT counted as
matured.** The agent didn't choose them. New counter:
`scalping_arbs_force_closed` (see §12) lives alongside
the existing counts and feeds telemetry only, not the
bonus.

**§8** Naked-loss annealing stays as a gene. With
force-close active, natural nakeds become rare — residual
nakeds (unpriceable runner at T−30s) are still subject to
`naked_loss_scale` the same way they were before. No
change to the anneal schedule.

## Force-close mechanism (Session 01)

**§9** New env config key
`constraints.force_close_before_off_seconds: int`, default
`0` = disabled = byte-identical to pre-change. When > 0
and `scalping_mode == True`, triggers the force-close loop
at each env step BEFORE the action-handling loop runs. Pre-
scalping behaviour (`scalping_mode == False`) ignores the
knob entirely — naked accounting doesn't apply in non-
scalping mode anyway.

**§10** Force-close uses the existing `_attempt_close`
path as its execution mechanism. No new matching code.
Each open pair's force-close leg is marked
`close_leg = True` on the placed `Bet` object so the
existing `_settle_current_race` classification logic sees
it as a close. The force-close flag on the leg is
distinguishable from an agent-initiated close via a new
`force_close = True` attribute on `Bet` (defaults
`False`; backwards-compatible).

**§11** Force-close is best-effort, with a RELAXED
matcher path (2026-04-21 revision — supersedes the original
strict path which refused ~95% of attempts in the cohort-A
smoke run due to data sparsity / thin books near the off).
The env passes `force_close=True` through
`ExchangeMatcher.match_back / match_lay / pick_top_price`,
which then:

1. Drops the LTP requirement — a runner with no LTP can
   still be closed against its available book.
2. Drops the ±`max_price_deviation_pct` junk filter — any
   priceable level is a valid close target.
3. **Keeps** the hard `max_back_price` / `max_lay_price`
   cap — the cap protects against matching into parked
   £1–£1000 orders where the consequence of a match is
   catastrophic.
4. **Keeps** single-price / no-walking (the matcher takes
   ONE level, not a sweep). §2 unchanged.

Agent-initiated closes via `close_signal` keep the strict
match (`force_close=False`). Only env-initiated force-close
at T−N sees the relaxation.

A force-close the relaxed matcher still refuses — empty
opposite-side book (`force_close_refused_no_book`), best
price above hard cap (`force_close_refused_above_cap`), or
stake below `MIN_BET_STAKE` after self-depletion
(`force_close_refused_place`) — leaves the pair open; it
settles naked via the existing naked-term accounting,
subject to `naked_loss_scale`. These counters are exposed
on `info` and JSONL per §29.

Design justification: leaving a pair naked costs ±£100s of
directional variance; crossing a thin or unpriced book
costs ±£0.50–£5 of spread — the relaxed path is strictly
better on expectation while the hard cap protects the
catastrophic tail. The matcher's strict rules exist to
prevent bad OPENS (where the agent pays the spread with no
offsetting matched leg); closes have an already-matched
aggressive partner whose exposure we're flattening, so the
strictness is the wrong trade-off for env-initiated closes.

**Budget (overdraft):** `place_back` / `place_lay` with
`force_close=True` bypass the per-race budget gate (`capped =
min(stake, available_budget)` and the lay liability scale-down).
`bm.budget` / `_open_liability` can go past `starting_budget`;
the assumption is the live trader has more than a single race's
worth of capital in the bank, so an overdraft to flatten a
matched position at T−N is always reachable. The cost still
flows through `race_pnl` → `raw_pnl_reward` at settle so the
agent sees it as real cash cost. `MIN_BET_STAKE` (£2) still
applies — Betfair's real minimum, not a sim-only knob.

**Sizing (revised 2026-04-22):** force-close uses **equal-
profit sizing**, same helpers as `close_signal`
(`equal_profit_lay_stake` / `equal_profit_back_stake`).
Equal-profit produces a hedge whose net P&L at settle is
the same on race-win vs race-lose — bounded by
`~spread × stake`, no race-outcome variance.

An earlier revision (2026-04-21) tried 1:1 stake matching
under the argument that equal-profit stakes didn't fit the
per-race budget. The cohort-A probe showed the flaw: at
drifted prices 1:1 hedges are highly asymmetric. Back £50
@ 5.0 + 1:1 lay £50 @ 8.0 settles at −£160 on race-win
but −£2 on race-lose; summed over ~600 force-closes per
episode the race-outcome variance produced −£800 to −£1900
episode rewards, blew up PPO log-prob ratios (approx_kl
39,786 vs the 0.03 early-stop threshold), and collapsed
agents to bets=0 by ep10 (worker.log 2026-04-21T22:37).

The "stake doesn't fit budget" concern is now handled by
the overdraft above — equal-profit stake lands in the
overdraft and the hedge is bounded by construction.
MIN_BET_STAKE (£2) still applies; if equal-profit can't
match at least £2 the pair stays naked, same as any other
refusal. Real-world rules stay intact.

**§12** New `BetManager` counter
`scalping_arbs_force_closed` and new per-race stat
`arbs_force_closed`. Exposed in `info` dict, per-race
trace, and `episodes.jsonl`. Separate from `arbs_closed`
(which remains agent-initiated). Matured-arb bonus counts
only `arbs_completed + arbs_closed`, NEVER
`arbs_force_closed` (per §7).

**§13** Force-close P&L lands in `race_pnl` via the normal
settlement path — same as any other close. The
`scalping_closed_pnl` aggregate gains a
`scalping_force_closed_pnl` sibling for telemetry; the
raw-reward formula changes:

```
race_pnl = scalping_locked_pnl
         + scalping_closed_pnl
         + scalping_force_closed_pnl
         + scaled_naked_sum
```

where `scaled_naked_sum` is the existing formula over
pairs that ended naked despite the force-close attempt.

**§14** Force-close does NOT contribute to the
`close_signal`-success shaped bonus
(`+£1 per agent-initiated close`). That bonus stays
agent-only. Force-closed pairs contribute 0 to it,
`arbs_closed` to it, and `arbs_force_closed` to it.

## Transformer context-window ceiling raise (Session 01)

**§14a** `PPOTransformerPolicy` (in
`agents/policy_network.py`) currently documents and
accepts `transformer_ctx_ticks ∈ {32, 64, 128}`. This
plan widens the allowed set to
`{32, 64, 128, 256}` — strictly additive. Nothing is
removed; lower values stay valid. Default behaviour
(no gene override) keeps whatever the plan data model
currently defaults to (32, per the existing class
docstring).

**§14b** The widening is purely a range edit in the
class docstring + any enumerated validation list. No
architectural change:
- `self.position_embedding = nn.Embedding(ctx_ticks,
  d_model)` already sizes off the gene value.
- The causal mask (`torch.triu` at ctx × ctx) already
  sizes off the gene value.
- No hardcoded ceiling exists in the policy code.

Confirm the above by reading
`agents/policy_network.py::PPOTransformerPolicy.__init__`
BEFORE editing. If an unexpected ceiling turns up (e.g.
a validation assert, a test that enumerates
`{32, 64, 128}`, an architecture_registry range), that
goes in-scope and gets fixed as part of Session 01.

**§14c** One new smoke test:
`test_transformer_builds_and_forwards_at_ctx_256`.
Instantiates `PPOTransformerPolicy` with
`transformer_ctx_ticks=256`, runs a single forward
pass on a synthetic obs tensor, asserts the output
shapes match the documented contract, and records GPU
memory delta so a regression into OOM territory would
be obvious. Must pass on CPU too (tests run CPU-only
by default).

**§14d** Weight-file cross-loading: a model trained
at `ctx_ticks=128` cannot be loaded into a policy
built at `ctx_ticks=256` (different
`position_embedding` shape). This is already true for
`ctx ∈ {32, 64, 128}` pairs and is governed by
`registry/model_store.py`'s architecture-hash check —
no new constraint needed, but Session 01's test should
confirm the existing hash machinery treats ctx=256 as
a distinct architecture variant (no silent
coercion).

## Entropy controller velocity (Session 01)

**§15** New gene `alpha_lr: float`, range
`[1e-2, 1e-1]`. Replaces the hardcoded `1e-2` constant in
`PPOTrainer`. Default (plan-level, for runs without a
gene override) stays at `1e-2` so existing reference runs
are byte-identical when `alpha_lr` isn't in the gene
schema.

**§16** `alpha_lr` is per-agent. Set at `PPOTrainer`
construction time and NEVER mutated during the agent's
lifetime. Construction takes the gene value, creates the
`self._alpha_optimizer = torch.optim.SGD(...,
lr=alpha_lr, momentum=0)`. That's it.

**§17** `alpha_lr` is whitelisted in `_REWARD_OVERRIDE_KEYS`
**not applicable** — it's a trainer-side gene, not a
reward-config gene. Instead, `_TRAINER_GENE_MAP` (or the
equivalent per-trainer override path) is the whitelist
layer. If no such map exists, add one; document it.

**§18** The smoke-test tracking-error gate from
`plans/naked-clip-and-stability` Session 04 stays wired.
The gate asserts that entropy is moving toward target
(not away) across the smoke agent's episodes. Widening
`alpha_lr` should make the gate *easier* to pass, not
harder — if a smoke run regresses, that's a real signal
worth investigating before the full launch.

## Shaped-penalty warmup (Session 02)

**§19** New plan-level config key
`training.shaped_penalty_warmup_eps: int`, default `0`
= disabled = byte-identical to pre-change. Value applies
uniformly across the population; NOT a gene. Typical
values: 5–15.

**§20** Warmup scales ONLY `efficiency_cost` and
`precision_reward`. All other shaping terms
(early_pick_bonus, drawdown_term, spread_cost_term,
inactivity_term, naked_penalty_term, early_lock_term,
matured_arb_term, MTM per-step) stay at full strength
from ep 1.

**§21** Warmup shape is linear:

```
if episode_idx < warmup_eps:
    scale = episode_idx / warmup_eps
else:
    scale = 1.0
```

`episode_idx` is 0-based (ep1 = index 0) and counts
episodes since training started (BC pretrain doesn't
count; PPO rollout episodes do). After warmup the
effective scale is exactly 1.0 — no residual damping.

**§22** `_log_episode` records
`shaped_penalty_warmup_scale` as an optional JSONL field.
Pre-change rows lack it; downstream readers must tolerate
absence.

**§23** The invariant `raw + shaped ≈ total` holds at
every warmup step. `shaped_bonus` in the info dict and
JSONL row reflects the SCALED value (the one that
actually contributes to reward), not the unscaled
component sum. Integration test covers scale = 0 / 0.5 /
1.0.

## Plan draft + validator (Session 03)

**§24** New training plan
`registry/training_plans/arb-signal-cleanup-probe.json`:

- Population 48 agents (3 architectures × 16 each).
- 4 generations, `auto_continue: true`,
  `generations_per_session: 1` so each gen is its own
  resumable session (matches the existing pattern).
- **Cohort split via gene-range blocks** — see §28.
- `reward_overrides`: matured-arb bonus and naked-loss
  scale carry over from current config.yaml defaults.
- `hp_ranges` INCLUDES (new for this plan):
  - `alpha_lr` — range `[1e-2, 1e-1]`.
  - `transformer_ctx_ticks` — PINNED to 256 (range
    `{min: 256, max: 256}` or whatever the schema uses
    to express "no variation"). Reason: per the
    2026-04-21 transformer audit (see
    `lessons_learnt.md` entry), the transformer's
    default `ctx_ticks=32` covers only ~13 % of a
    typical race; 128 covers ~54 %. Session 01 widens
    the arch's supported range to include 256 (§14a-
    §14d), which covers the full race for the median
    case (~238 ticks). Longer races (300+ ticks,
    typically Wolverhampton / multi-race parades)
    still truncate at the earliest ticks; this is a
    strict improvement on 128, not a guarantee of
    full-race coverage.
  - All other genes carry over from
    `arb-curriculum-probe` unchanged.
- `naked_loss_anneal: {start_gen: 0, end_gen: 2}` —
  unchanged from `arb-curriculum-probe`.
- `training.curriculum_day_order: "density_desc"` —
  unchanged.
- `training.shaped_penalty_warmup_eps: 10` — new.
- `constraints.force_close_before_off_seconds: 30` —
  new.
- `seed` different from `arb-curriculum-probe` (pick
  8101).
- `status: "draft"`; all runtime fields null.

**§25** The three-cohort ablation:

- **Cohort A (16 agents):** all three mechanisms active.
  `alpha_lr` drawn from gene range,
  `force_close_before_off_seconds = 30`,
  `shaped_penalty_warmup_eps = 10`.
- **Cohort B (16 agents):** entropy velocity only.
  `alpha_lr` drawn from gene range,
  `force_close_before_off_seconds = 0`,
  `shaped_penalty_warmup_eps = 0`.
- **Cohort C (16 agents):** warmup + force-close only.
  `alpha_lr = 1e-2` (pinned),
  `force_close_before_off_seconds = 30`,
  `shaped_penalty_warmup_eps = 10`.

**§26** If the plan data model doesn't support cohort-
level overrides, Session 03 falls back to THREE plan
files (`arb-signal-cleanup-probe-A.json`,
`-B.json`, `-C.json`) run serially against the same
registry snapshot. Checked in Session 03's first step.
Either way, cohort identity is recorded on each episode's
JSONL row via a new `cohort` field (A / B / C).

**§27** Validator script
`scripts/validate_arb_signal_cleanup.py` reuses the 5
criteria from `plans/arb-curriculum/`. Adds a sixth
**diagnostic** (not pass/fail) column to the output
table:

| # | Criterion | Pass/Fail | All agents | Cohort A | B | C |

Cohort-wise pass/fail is informational. The headline
pass/fail is population-wide (all 48 agents).

## Telemetry and invariant

**§28** Existing invariant test
`test_invariant_raw_plus_shaped_equals_total_reward` MUST
stay green, parametrised across:

- `force_close_before_off_seconds ∈ {0, 30}`
- `shaped_penalty_warmup_eps ∈ {0, 5}` (with varying
  `episode_idx` to hit the ramp)
- `alpha_lr ∈ {1e-2, 5e-2}` (controller still stable)

**§29** Per-episode JSONL rows gain optional fields:

- `force_close_before_off_seconds: int` (Session 01)
- `arbs_force_closed: int` (Session 01)
- `scalping_force_closed_pnl: float` (Session 01)
- `alpha_lr_active: float` (Session 01)
- `shaped_penalty_warmup_scale: float` (Session 02)
- `shaped_penalty_warmup_eps: int` (Session 02)
- `cohort: str` (Session 03 — "A" / "B" / "C" or
  "ungrouped")
- `force_close_attempts: int` (Session 03b — total
  _attempt_close calls with force_close=True)
- `force_close_refused_no_book: int` (Session 03b —
  no priceable opposite-side level)
- `force_close_refused_place: int` (Session 03b —
  matcher refused despite a priceable peek price)
- `force_close_refused_above_cap: int` (Session 03b —
  subset of place_refused: top price > hard cap)
- `force_close_via_evicted: int` (Session 03b —
  attempts that hit the pair_id_hint fallback because
  the passive was already cancelled)
- `episode_idx_at_settle: int` (Session 03b — env-side
  _episode_idx at _get_info time; diagnostic for the
  ep1 warmup bug, safely removable once verified)

Pre-change rows lack them; downstream readers must
tolerate absence. Same backward-compat pattern as
`mtm_weight_active`, `alpha`, `naked_loss_scale_active`.

## Testing

**§30** Each session commit ships with new tests. Full
`pytest tests/ -q` MUST be green on every session commit.
NEVER run the full suite during active training (operator
directive from prior plans).

**§31** Session 01 tests (force-close + entropy velocity
+ transformer ctx widening):
- **Force-close fires at threshold.** Scripted race where
  an open pair exists at T−31s and T−29s; force-close
  triggers at T−29s only.
- **Force-close uses matcher.** Scripted race with a
  matchable opposite-side book → close lands; same race
  with an unpriceable runner → position stays naked and
  is counted in `arbs_naked`.
- **Force-close respects junk filter.** Opposite-side
  best price sits outside `max_price_deviation_pct` of
  LTP → close refused, position stays open (then naked
  at settle).
- **Force-close P&L in `race_pnl`.** Settlement accounting
  routes the forced close's P&L into `scalping_force_
  closed_pnl` and summed into `race_pnl`.
- **Matured-arb bonus excludes force-closes.** Scripted
  race with 5 force-closes and 2 natural matures →
  `n_matured = 2`, not 7.
- **Close_signal bonus excludes force-closes.** Same
  scripted race; agent-initiated close count = 0, so
  the `+£1 per close_signal success` bonus contributes
  0 on that race.
- **`alpha_lr` gene passthrough.** PPOTrainer built with
  gene override `{"alpha_lr": 0.05}` → optimiser LR is
  0.05.
- **`alpha_lr` doesn't mutate.** Scripted 5-episode run
  with gene value 0.05 → `_alpha_optimizer.param_groups
  [0]['lr']` is 0.05 at every checkpoint.
- **Invariant parametrised.** Invariant test covers
  `force_close_before_off_seconds ∈ {0, 30}` and
  `alpha_lr ∈ {1e-2, 5e-2}`.
- **Transformer builds at ctx=256** (per §14c). One
  test that instantiates `PPOTransformerPolicy` with
  `transformer_ctx_ticks=256`, forwards a synthetic
  obs, asserts output shapes match the contract.

**§32** Session 02 tests (shaped-penalty warmup):
- **Default (warmup=0) byte-identical.** Scripted
  rollout with `warmup_eps=0` matches pre-change
  per-episode (raw, shaped, total) to float-eps.
- **Linear ramp.** `warmup_eps=10`, episode_idx in
  {0, 5, 9, 10, 20}: scale is 0.0, 0.5, 0.9, 1.0, 1.0.
- **Only two terms affected.** Scripted race with
  known non-zero components; at scale=0.5 the shaped
  total is
  `sum_of_other_terms + 0.5 * (efficiency_cost +
  precision_reward)`.
- **JSONL field present.** Post-episode row carries
  `shaped_penalty_warmup_scale`.
- **No cliff at warmup+1.** Scripted 20-episode run
  with `warmup_eps=10`; reward trajectory is
  continuous at ep10 → ep11 (no step discontinuity).
- **Invariant parametrised.** Invariant test covers
  `shaped_penalty_warmup_eps ∈ {0, 5}` with varying
  `episode_idx`.

**§33** Session 03 tests (plan draft + validator): no new
env/trainer tests. The plan-file JSON validates via
`PlanRegistry('registry/training_plans').list()`. The
validator script is CLI-only; `pytest` doesn't cover it.
Instead, Session 03 runs the validator against the
`arb-curriculum-probe` logs as a sanity check that the 5
criteria are computed identically (same pass/fail on the
prior run's data).

## Reward-scale change protocol

**§34** This plan introduces reward-scale changes:
- Non-zero `force_close_before_off_seconds` changes what
  lands in `race_pnl` (forced closes replace nakeds on
  most pairs, bounded spread cost replaces directional
  variance).
- Non-zero `shaped_penalty_warmup_eps` scales
  `efficiency_cost` and `precision_reward` early.

Runs with either active are NOT byte-identical to
pre-change. Scoreboard rows pre-plan vs post-plan are
comparable only when `force_close_before_off_seconds = 0`
AND `shaped_penalty_warmup_eps = 0`.

**§35** CLAUDE.md "Reward function: raw vs shaped" and
"Bet accounting" sections gain TWO new dated paragraphs
(one per reward-scale change):

- "Force-close at T−N (2026-04-21)" — under "Bet
  accounting" since it changes what gets bet-accounted.
- "Shaped-penalty warmup (2026-04-21)" — under "Reward
  function: raw vs shaped".

Historical entries stay preserved.

## Cross-session

**§36** Sessions land as separate commits, in order
01 → 03. Session 03 is operator-gated (plan draft +
launch same pattern as `arb-curriculum` Session 06).

**§37** If Session 01's invariant tests fail, Session 02
does not start. Force-close accounting must be correct
before warmup stacks on top of it.

**§38** If Session 02's invariant tests fail, Session 03
does not start. Same principle.

**§39** Do NOT bundle the launch into the Session 03
commit. Session 03 is "plan draft + validator script +
prereq check + documentation"; the launch is a follow-on
operator action that writes back into `progress.md` as a
Validation entry.

**§40** The `277bbf49` `arb-curriculum-probe` plan stays
in the registry with `status = failed`. This plan does
NOT archive the registry — the current registry state is
valid (no new model weights to preserve; the
post-crash-fix worker will cleanly ignore the failed
plan). If operator wants a registry archive, Session 03
includes an optional step.

**§41** When each session lands, append a dated entry to
`progress.md` mirroring the format used in
`plans/arb-curriculum/progress.md`: What landed, Not
changed, Gotchas, Test suite, Next.
