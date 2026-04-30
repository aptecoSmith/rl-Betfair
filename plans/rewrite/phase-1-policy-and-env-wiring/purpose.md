---
plan: rewrite/phase-1-policy-and-env-wiring
status: design-locked
opened: 2026-04-26
depends_on: phase-0-supervised-scorer (GREEN, 2026-04-26)
---

# Phase 1 — new policy classes + env wiring

## Purpose

Replace the v1 multi-head continuous policy
(`agents/policy_network.py`, 70-dim Box action space) with a
**discrete-action policy** that has a **per-runner value head** and
**reads the Phase 0 scorer's calibrated `P(mature | features)` as a
frozen feature**. Wire it to the existing env via a thin shim — the
env stays untouched (rewrite plan hard constraint #1).

**No training in this phase.** The deliverable is: random-weight
policy → shim → existing `BetfairEnv` → 1000 steps with no crash,
emitting one episode of `episodes.jsonl` rows that match the v1
schema closely enough that the existing UI / scoreboard reader
doesn't break.

Phase 2 implements the new PPO trainer. Phase 1's value head is
present but un-trained.

## Why this phase

`plans/greenfield_review.md` and the rewrite README pin the
recurring v1 problems (force-close stuck ~75 %, selectivity gap, H1
label conflation, H2 partial attenuation) on the 70-dim continuous-
multi-head action space. The prescription:

1. **Discrete head over "what to do this tick".** A categorical
   over a small set of actions (no-op, open back on runner_i,
   open lay on runner_i, close on runner_i) makes "pick the
   selective opportunity" a first-class output, not something the
   policy has to learn to express via fine-grained continuous
   choices on 14 independent heads.
2. **Per-runner value head.** A scalar value head on a per-runner
   action space loses credit-assignment between which runner caused
   which P&L delta. Per-runner V(s, runner) gives GAE per-runner.
3. **Scorer as frozen observation, not joint-trained head.**
   Phase 0's standalone scorer hit AUC 0.965 on the strict mature
   label. Feeding it as obs avoids the H1 failure mode where the
   joint-trained `fill_prob_head`'s label conflated force-closes
   with maturations and steered the actor toward bad opens.

## What's locked

### Action space (locked)

**One discrete categorical per tick** over the action set:

```
size = 1 (no-op) + 2 × max_runners (open_back_i, open_lay_i)
                 + max_runners       (close_i)
```

For the standard `max_runners=14` env that's `1 + 28 + 14 = 43`
choices per tick. Picking `open_back_i` triggers an aggressive back
on runner `i` paired with an equal-profit passive lay; the
arb_spread defaults to the env's `MAX_ARB_TICKS=25` (or the locked
`arb_ticks=20` from Phase 0's findings — implementer's call, see
hard constraint §6).

**Two small continuous heads** active conditional on the discrete
choice:

- `stake` ∈ `[MIN_BET_STAKE, max_stake_cap]` — Beta-distributed,
  re-scaled. Squashed to a Beta to keep the support strictly
  bounded (Normals get clipped and that's where v1's gradient
  pathologies started).
- `arb_spread_ticks` ∈ `[MIN_ARB_TICKS, MAX_ARB_TICKS]` — Beta or
  scaled-Beta, optional (gated off if Phase 0's `arb_ticks=20`
  empirically dominates). **Implementer should default to a
  fixed `arb_ticks=20` and ship without this head**; bringing it
  back is a Phase 2 follow-on if needed.

The discrete action's argmax-or-sample directly determines what
gets placed; continuous heads only set sizing/shape on the
already-chosen action. No 70-dim continuous output anywhere.

**Action masking (required).** The categorical sees a per-tick
mask zeroing illegal actions:

- No-op is always legal.
- `open_back_i` masked off if runner `i` is `INACTIVE` / has no
  LTP / has an open position already / hard cap exceeded /
  budget below `MIN_BET_STAKE`.
- `open_lay_i` masked symmetrically.
- `close_i` masked off if runner `i` has no open pair to close.

Masking is at the logits level (`logits[masked] = -inf`), not by
post-hoc rejection — the policy learns to put no probability mass
on illegal actions. Not optional; v1 paid a heavy entropy-control
tax for the inverse choice.

### Observation space (locked: extend, don't replace)

Keep the existing v1 obs vector as a base — re-implementing
feature extraction would duplicate the env's `_build_observation`
path and re-litigate decisions that aren't this phase's scope
(hard constraint #2). **Append** the per-runner scorer outputs as
new dimensions:

```
obs_v2 = obs_v1 ⊕ scorer_features

scorer_features = concat per runner i ∈ [0, max_runners) of:
    [calibrated_p_mature_back_i, calibrated_p_mature_lay_i]

→ extra dim = 2 × max_runners
```

The shim computes the 30 features the scorer expects from env
state per runner per side, runs the booster, applies the
calibrator, packs into the obs. Zero (with a `scorer_invalid_i`
flag bit) when the runner is inactive / the booster refuses
(NaN feature, etc.).

**Why obs and not action-head input.** The H1 failure mode
(`plans/per-runner-credit/findings.md`) was that joint training
let the policy steer the head toward a wrong-target label. Frozen
obs has no such pathway — the booster's weights never update from
RL.

### Per-runner value head (locked)

`V_head: state → R^max_runners` — outputs a scalar per runner.
GAE in Phase 2 will be computed per-runner using per-runner
realised P&L (already split per pair_id by `BetManager`).

The aggregate scalar value (for global-baseline ablation) can be
recovered by summing across runners, so v1-style bookkeeping is
still possible if Phase 2 needs it.

### Scorer integration (locked)

The shim layer:

1. Loads `models/scorer_v1/model.lgb` once at construction.
2. Loads `models/scorer_v1/calibrator.joblib` once at construction.
3. Loads `models/scorer_v1/feature_spec.json` to confirm the
   feature contract matches the on-disk model
   (`tests/test_scorer_v1_inference.py::test_feature_spec_matches_booster`
   is the regression guard).
4. On every env step, computes the 30 features per (active runner,
   side) using the **same `FeatureExtractor`** the dataset
   pipeline used (`training_v2.scorer.feature_extractor.FeatureExtractor`
   — re-import, do not re-implement; reproducing the rolling
   windows wrong is the failure mode).
5. Predicts → applies isotonic → packs into the obs vector.

**Scorer is frozen.** No optimiser touches its parameters in
Phase 1, 2, or 3. If the scorer needs to change, that's a
Phase 0 re-run — file as a follow-on. Hard constraint #3.

### Architecture variants

Three policy backbones, mirroring v1's options:

- `DiscreteLSTMPolicy` — single-layer LSTM, hidden=128. The
  baseline.
- `DiscreteTimeLSTMPolicy` — TimeLSTM variant (carries time-since
  state).
- `DiscreteTransformerPolicy` — small causal transformer,
  `d_model=128`, `ctx_ticks=128` (Phase 0 found the medium ctx
  was the sweet spot in v1's ablations).

Each shares a backbone → discrete logits + masked categorical +
continuous heads + per-runner value head. The shared base class
matches v1's `BasePolicy` contract enough that registry /
checkpoint wiring works (architecture-hash treats them as new
variants per the existing `model_store.py` shape check — no
schema migration).

## Success bar (Phase 1)

Phase 1 ships iff ALL of:

1. **Smoke test:** random-init policy → shim → env, runs 1000 steps
   on a real day's data without exception. `episodes.jsonl` has at
   least one row written. Discrete action distribution shows ALL
   action types fired at least once across the run (no-op + at
   least one open + at least one close).
2. **Action masking correctness:** with the mask applied, zero
   illegal actions land on the env. Test: instrument the shim to
   count refusals and assert it stays at zero across the smoke run.
3. **Per-runner value shape:** `value_head(obs).shape ==
   (batch, max_runners)`. Sum-over-runners can be computed; no
   crash if Phase 2 wants it.
4. **Scorer wiring correctness:** the shim's per-tick scorer
   prediction on a known feature vector matches the standalone
   booster's prediction (the
   `test_feature_spec_matches_booster` regression guard already
   protects feature-name drift; Phase 1 adds one for prediction
   equality).
5. **No env changes.** `git diff env/ data/` is empty at end of
   phase (or surfaces only env-bug fixes the implementer files as
   a SEPARATE commit per hard constraint #1).

If any of 1–4 fails: stop and discuss. If 5 fails: revert, file
the env issue as a Phase −1 follow-on, do not bundle.

## Deliverables (Phase 1 closeout)

A new directory `agents_v2/` with:

- `agents_v2/__init__.py` — exports.
- `agents_v2/action_space.py` — `DiscreteActionSpace` enum +
  per-runner index math + mask helpers.
- `agents_v2/env_shim.py` — `DiscreteActionShim`: translates the
  discrete + small-continuous policy outputs → the existing
  `BetfairEnv`'s 70-dim Box action vector. Loads + applies the
  Phase 0 scorer per step. Computes obs extension.
- `agents_v2/discrete_policy.py` — `BaseDiscretePolicy` +
  `DiscreteLSTMPolicy` (Transformer + TimeLSTM variants are
  optional in this phase — implementer's call; ship LSTM as the
  baseline).
- `agents_v2/smoke_test.py` — runs the smoke described in success
  bar #1. CLI: `python -m agents_v2.smoke_test`.

Tests under `tests/`:

- `tests/test_agents_v2_action_space.py` — masking correctness,
  index round-trip, illegal-action rejection.
- `tests/test_agents_v2_env_shim.py` — discrete → continuous
  translation produces a valid Box action; scorer output gets
  packed into obs at the right indices; obs dim arithmetic.
- `tests/test_agents_v2_discrete_policy.py` — forward pass shape
  guards (logits shape, value shape, continuous-head shapes),
  masked-categorical respects the mask (no probability on
  masked actions), backward pass produces gradients on every
  param.
- `tests/test_agents_v2_smoke.py` — slow-marked end-to-end smoke
  (1000 steps on a fixture day, asserts the success bar
  conditions).

A short writeup at
`plans/rewrite/phase-1-policy-and-env-wiring/findings.md`:
shape decisions, surprises (e.g. if action masking turns out
harder than expected — common pitfall in JAX/Torch
distributions), Phase 2 implications.

## Hard constraints

1. **Don't touch the env.** Even if you find a bug. File and
   ship the fix as a separate Phase −1 commit before resuming.
   (Rewrite-plan hard constraint #1.)
2. **Don't touch the data pipeline.** (Rewrite-plan hard
   constraint #2.)
3. **Scorer is frozen.** No `requires_grad=True` on any scorer
   weight; no optimiser sees scorer parameters. If you need a
   different scorer, that's a Phase 0 re-run.
4. **Parallel tree.** All new code under `agents_v2/`. Do not
   modify `agents/` (v1 stays the comparison baseline through
   Phase 3).
5. **No new shaped rewards in v2.** (Rewrite-plan hard constraint
   #5.) Phase 2 inherits this; Phase 1 doesn't add reward
   shaping anyway, but flag in case the env shim is tempted to
   "help" the policy with an extra reward term — it isn't.
6. **No hyperparameter search.** Pick defensible defaults and
   ship; the smoke bar doesn't measure performance.
7. **No "while we're at it" refactors of v1.** v1 stays running.

## Out of scope

- Training the new policy (Phase 2).
- New trainer architecture (Phase 2).
- Frontend wiring (Phase 3 — Phase 1's smoke run can write
  episodes.jsonl with whatever schema is convenient; the UI
  adapter is Phase 3).
- GA breeding / mutation (Phase 3).
- Multiple runs / cohort comparisons (Phase 3).
- Removing v1 code (after Phase 3 success per rewrite README).
- A second supervised scorer or scorer re-training.
- Ablation studies (force-close rate vs scorer input weight,
  etc.) — those are Phase 2's domain when training actually
  changes weights.

## Phase 0 findings that shape Phase 1

From `plans/rewrite/phase-0-supervised-scorer/findings.md`
(2026-04-26, GREEN):

1. **Scorer test AUC = 0.965** (per-side back 0.96, lay 0.91).
   Strong enough that feeding calibrated probabilities as obs
   should give the policy a real selectivity gradient.
2. **`side_back` is the dominant feature (3× the next).** Phase
   1's per-runner obs already carries side via the action layout
   (open_back vs open_lay are distinct discrete actions); the
   scorer's per-side prediction is computed twice per runner
   (once per side) so the policy sees the asymmetry directly
   without the scorer needing to "rediscover" side via the obs.
3. **F7 velocity features are dead** (`time_since_last_trade_seconds`,
   `traded_volume_last_30s` are NaN/0 in 100 % of training data).
   Phase 1's shim must compute them the same way the dataset
   builder did — the FeatureExtractor reuse handles this for
   free, but if the implementer is tempted to "fix" them online,
   the scorer will be systematically miscalibrated. **Don't.**
4. **Bar 3's P&L estimate was over-optimistic** (+£60k/day on
   test under deterministic per-row P&L). Phase 1's smoke test
   doesn't measure P&L, but Phase 2 should expect realised P&L
   to be much lower than the scorer's calibrated `P(mature)`
   suggests — execution slippage and settlement variance live
   in real env steps and weren't simulated.

## Sessions

1. `01_action_space_and_env_shim.md` — design + implement the
   discrete action space, the masking helpers, and the env shim
   (discrete → 70-dim Box translator, scorer wiring into obs).
   Tests for action-space correctness and shim translation
   round-trip. **No policy class in this session.**
2. `02_policy_class_and_smoke_test.md` — implement
   `DiscreteLSTMPolicy` (Transformer / TimeLSTM optional), the
   masked categorical, the per-runner value head. End-to-end
   smoke test runs the success-bar checks. Findings writeup.

Each session is independently re-runnable. Session 02 imports
Session 01's shim verbatim; if Session 01's design needs to
change after Session 02 surfaces a problem, that's a Session 01
revisit — file as a finding and stop, don't patch in Session 02.
