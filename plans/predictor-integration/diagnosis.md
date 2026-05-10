# Diagnosis — v2 audit and the v3-vs-integrate verdict

## TL;DR

**Don't start a v3 repo.** The v2 stack already has the structural
hooks needed to consume the predictor outputs cleanly:

- `RUNNER_KEYS` is extensible (last extension was 2026-05-07 at
  Phase 14 S02 — added 10 direction-related features, bumped
  `OBS_SCHEMA_VERSION` 6 → 7).
- The `actor_input` concat pattern in v2's `DiscreteLSTMPolicy`
  (CLAUDE.md §"fill_prob feeds actor_head", §"mature_prob_head
  feeds actor_head") is exactly the shape required to consume
  per-runner external scalars: `concat([runner_emb, backbone,
  fill_prob, mature_prob], dim=-1)`. Adding 4 more columns
  (champion `p_win`, `p_placed`; ranker `softmax_share`,
  `top1_high_confidence_flag`) is structurally identical to the
  port already done.
- The env supports two action surfaces: `scalping_mode = True`
  (pair-based, current default) and `scalping_mode = False`
  (single-shot, used outside scalping). Adding a `value_mode`
  variant is a config-level switch over existing surface, not
  a new policy class.

**No genuine data gap.** Place-betting on Betfair is delivered
via the **each-way mechanic**, not a separate TO-BE-PLACED
market — and EW settlement is already complete in
`plans/ew-settlement/` (4 sessions, finished 2026-04-11). The
data already carries `each_way_divisor` and
`number_of_each_way_places` per race; `BetManager.settle_race`
correctly settles EW bets when `bet.is_each_way = True`. The
only env-side gap is an **action-surface flag** that lets the
policy elect to place a bet as each-way; Session 04 adds it,
~3–4 hours of work.

## What's load-bearing in v2 (must survive)

| Module | Why load-bearing |
|---|---|
| `env/betfair_env.py` | Ladder-correct matching, force-close, equal-profit pair sizing, day/race orchestration. The simulator is the moat. CLAUDE.md is dense with hard-won correctness facts that any successor would have to re-derive. |
| `env/exchange_matcher.py`, `env/bet_manager.py` | Single-price matching, junk filter, hard caps, force-close overdraft, MIN_BET_STAKE. CLAUDE.md §"Order matching" + §"Force-close at T−N". |
| `env/scalping_math.py` | Equal-profit pair sizing formula. The 2026-04-18 fix (`f7a09fc`) was load-bearing; pre-fix runs had a sizing bug. |
| `data/extractor.py`, `data/episode_builder.py` | Parquet pipeline. Same artefact `betfair-predictors` consumes. Stable, tested, indispensable. |
| `agents_v2/discrete_policy.py` (post-`plans/rewrite/phase-7`) | Per-runner aux-head pattern is the integration template. |
| `training_v2/discrete_ppo/trainer.py` | Discrete PPO with per-runner GAE; the trainer the integration plugs into. |
| `registry/model_store.py` + `models.db` | Architecture-hash check refuses pre-plan weights — the right behaviour when `RUNNER_DIM` changes. |
| Frontend (Vite/React + websocket scoreboard) | Schema-driven; the new RUNNER_KEYS appear automatically. No frontend work needed. |

## What's vestigial (the predictors supersede)

| Module | Why vestigial after this plan |
|---|---|
| `plans/rewrite/phase-0-supervised-scorer/` (the internal scorer) | Production champion in `betfair-predictors` is a strictly stronger version of the same idea — calibrated, test-set-validated, segment-performance-aware. Phase 0's standalone scorer can stay as a fallback or be retired in a follow-on plan. **Don't bundle the retirement here.** |
| `agents_v2/discrete_policy.py::fill_prob_head` (probably) | The fill-prob head was trained on env-derived labels (`count >= 2`). The champion's calibrated `p_win` × `p_placed` × the env's known commission gives a tighter, supervised version. KEEP the head for now (it's per-tick; champion is per-race), but its weight may default to 0 once the predictor signal is doing the discrimination work. |
| `agents_v2/discrete_policy.py::mature_prob_head` | Same argument. Phase 7's S03 leverage test shows the head doesn't move maturation_rate at weight 0.5 — likely because the supervised label space (race outcome) is what's needed and the head's BCE label (force-close-aware maturation) isn't strong enough. Keep, but expect weight to go to 0 once `p_win` enters. |

**No vestigial removal in this plan.** Vestigial-by-this-plan
is itself a hypothesis that the integration must verify; only
remove after Session 07's three-way comparison provides evidence.

## What's "mid" (works, would benefit from cleanup but not in this plan)

| Module | Why not in this plan |
|---|---|
| Phase 14 S02 engineered direction features (10 dims at v7) | They're already in the obs and don't conflict with a real direction-predictor model call. KEEP for now; the per-tick model call is additive, not replacement. Decide on consolidation in a follow-on plan AFTER seeing whether the per-tick model adds gradient. |
| Reward shaping accumulators (8+ terms, CLAUDE.md is half about them) | Plumbing-level legacy from arb-mode tuning. The new strategy modes (value-win, value-place) need their own simpler reward shapes (settle to win/lose; no scalp pair). New modes get new shapes; arb mode keeps its current shape. Don't touch arb mode's shaping in this plan. |
| The two-track `agents/` (v1) vs `agents_v2/` (v2) split | `plans/rewrite/` is dealing with this. Don't re-litigate. |

## What's genuinely missing (this plan's gaps)

| Gap | Scope | Session |
|---|---|---|
| Predictor loader (manifest read, per-race callable, segment_performance routing) | New module `predictors/loader.py`. ~200 LOC + tests. Imports stable inference code from `betfair-predictors/scripts/`. | 01 |
| Per-runner predictor outputs in observation tensor | RUNNER_KEYS extension (4 new keys + 1 segment-strong flag), `OBS_SCHEMA_VERSION` 7 → 8, `data/feature_engineer.py` injection. | 02 |
| Strategy-mode switch (`arb` / `value_win` / `value_each_way`) | New `training.strategy_mode` config key; env honours it for action surface + reward shape; trainer reads it for cohort label. | 03 |
| Each-way action surface (`each_way` action signal + `place_*(each_way=True)` kwarg + non-EW-race masking) | Small env + bet_manager addition; reuses complete EW settlement from `plans/ew-settlement/`. | 04 |
| Per-strategy reward shapes | Value modes: realised P&L only at settle; no scalping shaping. | 03 + 05 + 06 |
| Per-strategy cohort runs + comparison report | Three-way head-to-head with shared seed-day set. | 07 |

## How big a v3 would actually be

For the operator's "do we need v3" question, here's the honest
size-of-rewrite calculation if we did go v3 instead:

| Component | v3 cost |
|---|---|
| Re-derive env (matching, force-close, equal-profit, ladder semantics) | 2–3 weeks. CLAUDE.md is 700+ lines of correctness facts to preserve. |
| Re-derive data pipeline (parquet, episode_builder, day loading) | 1 week. Indistinguishable from current; nothing to gain. |
| Re-derive discrete-action policy + trainer (i.e. redo `plans/rewrite/`) | 3+ weeks of work that's already 7/16 phases done. Throwing it away is the largest single cost of going v3. |
| Re-derive registry, frontend, websocket schema, evaluator | 1–2 weeks. Same artefact in different repo. |
| **NEW thing v3 buys you** (predictor-first observation design) | ~2–3 days. The integration is small. |

So the v3 cost is ~6–8 weeks of mostly duplicated work for ~3 days
of genuine novelty. **The novelty is what this integration plan
delivers, without the duplication.**

## Why the existing `plans/rewrite/` is not the answer to the
predictor question

`plans/rewrite/` is an **action-space** rewrite — continuous
multi-head Gaussian → discrete categorical pick-which-runner. Its
motivation (`plans/greenfield_review.md`) is the 75% force-close
ceiling, not feature coverage. Its scope is policy + trainer,
explicitly **not** observation content. Phase 7 is currently
AMBER on the leverage test — that's about whether the new
discrete architecture can express selectivity given its current
features, NOT about whether the right features are available.

The two plans are orthogonal:

- `plans/rewrite/` answers "is the action space the bottleneck?"
- This plan answers "is the observation signal quality the bottleneck?"

They compose: this plan's predictor outputs feed v2's discrete
policy via the actor_input concat. **Both can be true** (action
space WAS too coarse AND observation signal WAS too weak) and
both fixes are needed. Phase 7's S03 result hints at exactly
this — the discrete architecture isn't getting the per-runner
discrimination signal even at high aux-head weight, which
suggests the missing factor is the signal itself.

## Pain points the operator has flagged (cross-reference)

From the operator's framing in this conversation:

> "We used to feed all the race data we could to the rl models, in
> the hope they would figure out what to use. Unfortunately we
> never increased the maturation rate enough to become profitable."

This describes the v1→v2 trajectory. Phase 7 in v2 confirms
the same ceiling persists. **The predictors are the
discrimination signal that the agent has been failing to
self-derive from raw market data.**

> "All that work [on attention heads, fill-prob, mature-prob heads]
> did not feel like it really worked."

Confirmed by Phase 7 S03 leverage test. The aux-head pattern
is correct architecturally; the supervised label needed to be
strong enough, and `count >= 2` / `force_close-aware` labels
aren't. Race-outcome labels (the predictors' supervision target)
are.

> "Now we have this great prediction model, we want to make
> sure we can train for all three cases — placers, value betting
> and arbing."

This is the framing this plan responds to. The three modes are
the deliverable. See [`strategy_modes.md`](strategy_modes.md).

## Verdict

**B+: Targeted integration, layered on the in-flight v2 stack.**
Predictor outputs become opt-in observation features behind
`observations.use_race_outcome_predictor: false`. A new
`training.strategy_mode` config gates which strategy the env and
trainer specialise for. The arb mode keeps current behaviour
when the flag is off (byte-identical regression test). The
each-way action surface is a small one-session addition that
reuses the already-complete `plans/ew-settlement/`
infrastructure.

If, after Session 07's three-way comparison, **no mode** beats
the v2 arb-only baseline, the diagnosis goes deeper than
"missing feature" and the v3 conversation re-opens with concrete
signal. Until then, the cheap test runs first.
