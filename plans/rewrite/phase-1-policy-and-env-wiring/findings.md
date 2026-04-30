---
plan: rewrite/phase-1-policy-and-env-wiring
session: 02
status: GREEN — Phase 1 complete, ready for Phase 2
opened: 2026-04-27
---

# Phase 1, Session 02 — discrete policy + smoke test findings

## TL;DR

**Verdict: GREEN.** All 5 success bars PASS on the 1000-step smoke
run against `2026-04-23` (real day, 11,872 ticks, 77 races) with a
random-init `DiscreteLSTMPolicy(hidden=128)`. Phase 1 is unblocked
for Phase 2.

| Bar | Threshold | Result | Pass |
|---|---|---|:---:|
| 1 — no exceptions raised | strict | clean 1000-step run | ✅ |
| 2 — episodes.jsonl row written | ≥ 1 row | 1 row at `logs/agents_v2_smoke/smoke.jsonl` | ✅ |
| 3 — action histogram covers NOOP, OPEN_BACK, OPEN_LAY, CLOSE | ≥ 1 each | 94 / 320 / 375 / 211 | ✅ |
| 4 — refusal counter | == 0 | **0** illegal actions sampled across all 1000 masked-categorical draws | ✅ |
| 5 — scorer matches standalone booster | strict equality | re-loaded booster + isotonic from disk; predictions match to 1e-9 | ✅ |

## What landed

### `agents_v2/discrete_policy.py`

```
DiscretePolicyOutput  — frozen dataclass with logits, masked_logits,
                        action_dist, stake_alpha, stake_beta,
                        value_per_runner, new_hidden_state.
BaseDiscretePolicy    — abstract: __init__(obs_dim, action_space,
                        hidden_size), init_hidden, forward(mask=…),
                        pack_hidden_states, slice_hidden_states.
DiscreteLSTMPolicy    — Linear(obs→hidden) → ReLU → LSTM(hidden,
                        hidden) → 4 heads.
```

LSTM hidden is `(h, c)` each `(num_layers=1, batch, hidden_size)`.
The pack/slice helpers concat / index_select along **dim=1**, mirroring
v1's contract (`CLAUDE.md` "Recurrent PPO: hidden-state protocol on
update"). The base class default (dim=0) is correct for the
transformer follow-on; the LSTM subclass overrides.

Stake heads emit `softplus(x) + 1.0`, so `α, β > 1` always — Beta is
unimodal, no near-0/1 hairpin pathologies. The policy returns the raw
parameters; the action caller builds the Beta and re-scales the
sample to `[MIN_BET_STAKE, max_stake_cap]` outside the policy.

The value head is plain `Linear(hidden → max_runners)` — no
activation, no clamp. **Per-runner is non-negotiable** (Session 02
prompt §"hard constraints"); a regression that returned `(batch, 1)`
would silently collapse Phase 2's per-runner GAE.

### `agents_v2/smoke_test.py`

CLI: `python -m agents_v2.smoke_test --day 2026-04-23`.
Loads the day, builds env + shim + random policy, runs up to
`n_steps` steps with masked categorical sampling, and writes one
JSONL row at `logs/agents_v2_smoke/smoke.jsonl`. The driver also
captures one (feature_vec, runner, side, calibrated_prob) tuple
during the run, then re-loads the booster + isotonic from disk and
asserts strict equality (success bar #5).

### Tests

| File | Tests | Status |
|---|---:|---|
| `tests/test_agents_v2_discrete_policy.py` | 14 | All passing — pure-torch, no scorer deps. |
| `tests/test_agents_v2_smoke.py` | 2 | `@pytest.mark.slow`, skip when scorer artefacts absent. |

The discrete-policy suite covers: forward shapes (batched + sequence),
explicit per-runner value-head shape guard, Beta `α, β > 1` invariant,
mask correctness (`-inf` at masked indices, no probability mass on
masked actions across 1000 samples, broadcast of 1-D mask), hidden
state init / pack / slice round-trip, end-to-end backward producing
gradients on every `requires_grad` parameter (catches accidental
`detach()`), and `value_head` gradient isolation from `logits_head`.

Combined regression count across Phase 1: **46/46 passing** on the
machine that ran Session 02 (action_space 17 + env_shim 15 +
discrete_policy 14 + smoke 2 — last 2 only execute under `-m slow`).

## Smoke run numbers

```
day: 2026-04-23 (production data, 77 races, 11,872 ticks)
n_steps run: 1000  (episode did not terminate within 1000 steps)
total_reward: -55.115
day_pnl:      -£41.37
refusal_count: 0
wall: 11.15 s   ⇒  11.15 ms / step (CPU, single-batch forward)

action histogram:
    NOOP        =  94
    OPEN_BACK   = 320
    OPEN_LAY    = 375
    CLOSE       = 211
```

The histogram is far from uniform across the 43 actions. That's
expected — the mask blocks `open_*_i` whenever runner `i` already has
an unsettled bet, and `close_i` is only legal when there's a
half-filled pair to close. Random sampling from the legal-only
distribution produces the observed asymmetric mix; the *uniformity*
that the prompt mentions is uniformity across **legal** actions, not
across the full 43-way space.

OPEN_BACK and OPEN_LAY together (695 / 1000 = 69.5 %) dominate the
mix because the random policy with 14 runners can typically open on
~10–14 of them per tick (most are legal once a position closes), and
each of those 28 OPEN_* actions independently competes against NOOP +
~4 CLOSE legal options. The marginal win-rate of OPEN over NOOP comes
out about right.

The negative reward and P&L are **expected and not a Phase 1 concern**
— a random-init policy executing 695 OPEN actions in 1000 ticks at
random sizing on real horse markets pays the spread on ~600+ pairs.
Phase 2 is when the policy starts learning.

## Forward-pass wall time

11.15 ms per step on CPU, batch=1. For Phase 2 reference: PPO update
wall is what matters; rollout-time forward is dwarfed by env-step
cost on real days (the loader's 9.92 s feature-engineering + 2.10 s
parquet load is one-shot per day, not per step).

## Findings worth pinning before Phase 2

1. **`obs_dim` for `max_runners=14` is `1792`** (env base 1764 + 28
   scorer features = 1764 + 2 × 14). The shim's `obs_dim` property is
   the source of truth; Phase 2's PPO trainer must construct the
   policy with `obs_dim=shim.obs_dim`, not by reading
   `env.observation_space.shape[0]` directly (that misses the scorer
   extension and the runtime would later fail with a `Linear(1764,
   128)` shape mismatch on the obs tensor of width 1792).

2. **The masked categorical interacts cleanly with PPO's
   importance-sampling ratio.** The `Categorical(logits=masked_logits)`
   distribution's `log_prob(action)` returns `-inf` for masked
   actions — but those actions can never be sampled, so the ratio
   `exp(new_lp − old_lp)` only ever evaluates at finite log-probs.
   PPO's clip-and-mean is well-defined. Phase 2 doesn't need a
   "compensate for mask" branch.

   The one caveat: if the env state changes the mask between rollout
   and update such that an action that was legal at rollout becomes
   masked at update time, `new_lp` would be `-inf` and the ratio
   would explode. **In our setting this can't happen** — the policy
   re-evaluates `forward(obs, hidden, mask)` at update time using the
   SAME mask captured during rollout (Phase 2's `Transition` should
   carry the mask alongside `obs` and `hidden_state_in`). Flag for
   Phase 2's design.

3. **NOOP-always-legal is load-bearing for masked softmax stability.**
   PyTorch's `Categorical(logits=…)` with all logits at `-inf`
   produces NaN probabilities (`softmax(-inf, …, -inf) = 0/0`).
   `compute_mask` guarantees `mask[0] = True` unconditionally, so
   that pathological case is unreachable. The smoke driver and
   `test_agents_v2_smoke` both assert `mask[0]` every step as a
   belt-and-braces guard.

4. **Stake-head squeezing.** `Linear(hidden → 1)` outputs
   `(batch, 1)`; the policy `.squeeze(-1)`'s before applying
   `softplus + 1` so the contract is `(batch,)`. Tests assert the
   shape explicitly. If Phase 2 wants per-runner stake (different
   stakes per runner), the head's output dim becomes `max_runners`
   and the squeeze is dropped — a natural extension, but **not
   needed in Phase 1** (the prompt locked the stake to a single
   scalar per tick).

5. **Action space size at `max_runners=14` is 43.** No-op + 14
   OPEN_BACK + 14 OPEN_LAY + 14 CLOSE. Symmetric in BACK / LAY by
   design — the per-side scorer prediction in obs is what gives the
   policy the asymmetry signal (Phase 0 finding: `side_back` is the
   dominant feature).

## Hand-off to Phase 2

Phase 2's PPO trainer construction surface (locked by Phase 1):

```python
shim = DiscreteActionShim(env)              # Session 01 deliverable
policy = DiscreteLSTMPolicy(
    obs_dim=shim.obs_dim,
    action_space=shim.action_space,
    hidden_size=128,
)
hidden = policy.init_hidden(batch=1)        # carries across episode
out = policy(obs, hidden_state=hidden, mask=mask)
action = out.action_dist.sample()           # (batch,) discrete index
log_prob = out.action_dist.log_prob(action) # for old/new ratio
stake_dist = Beta(out.stake_alpha, out.stake_beta)
stake_unit = stake_dist.sample()            # (batch,) ∈ (0, 1)
stake_log_prob = stake_dist.log_prob(stake_unit)
# value_per_runner: (batch, max_runners) — Phase 2's GAE input.
```

**Required `Transition` slots** (carry-overs from `CLAUDE.md`
"Recurrent PPO: hidden-state protocol on update" + Phase 1 mask
caveat above):

- `obs`, `action_idx`, `stake_unit_sample`
- `log_prob` (categorical) + `stake_log_prob` (Beta)
- `mask` — same mask the rollout categorical was conditioned on
- `hidden_state_in` — `(h, c)` numpy 2-tuple BEFORE the forward pass
- `value_per_runner` — for per-runner GAE

The pack/slice helpers on `DiscreteLSTMPolicy` already match v1's
batching contract (dim=1 for LSTM, default dim=0 for transformer);
Phase 2 can re-use them verbatim.

## Out of scope (deferred)

- **Transformer / TimeLSTM variants** — purpose.md flagged these as
  optional in this session. **Not shipped.** File as Phase 1
  follow-on; Session 02 stayed inside the LSTM-only scope to keep the
  smoke bar tight. The base class is already shaped to admit them
  (the abstract methods plus the dim=0 pack/slice defaults match the
  transformer's expected hidden-state shape).
- **`arb_spread` continuous head** — purpose.md §"Action space" said
  "default to a fixed `arb_ticks=20`; bringing it back is a Phase 2
  follow-on." Session 02 did not add an `arb_spread` head; the shim's
  `arb_ticks=20` is wired through.
- **Profiling forward-pass speed** — purpose.md §"Out of scope". The
  11.15 ms / step number is in this writeup for context, not as a
  benchmarked target.

## Hard constraints

| Constraint | Status |
|---|:---:|
| No training code (no optimiser, no loss, no GAE) | ✅ |
| No env edits (`git diff env/ data/` for THIS session is empty; pre-existing dirt unrelated) | ✅ |
| No re-import of v1 classes (parallel tree) | ✅ |
| Per-runner value head (not scalar) | ✅ — explicit shape guard |
| Mask at logits level, not post-hoc rejection | ✅ |
| No new shaped rewards | ✅ — smoke uses env's existing reward |
| No hyperparameter search | ✅ — `hidden=128`, no other tunables |

## Verdict

**Phase 1 GREEN, ready for Phase 2.**
