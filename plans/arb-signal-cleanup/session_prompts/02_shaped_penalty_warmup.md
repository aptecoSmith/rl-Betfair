# Session 02 prompt — Shaped-penalty warmup

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — especially the
  "(B) Shaped-penalty warmup" subsection of the design
  sketch and the "explore-cliff" failure-mode entry.
- [`../hard_constraints.md`](../hard_constraints.md) —
  §19–§23 (warmup mechanics), §28 (invariant
  parametrised), §29 (JSONL fields), §30–§32 (testing).
- [`../master_todo.md`](../master_todo.md) — Session 02
  deliverables.
- [`../progress.md`](../progress.md) — read the Session
  01 entry. If Session 01 discovered any API shift in
  the shaped-component accumulation, the warmup scaling
  hooks adjust accordingly.
- `env/betfair_env.py` — the file edited in Session 01.
  Locate `_settle_current_race` and the `shaped` sum
  (the assignment around line 2105–2115 in the
  pre-Session-01 code; may have moved). The warmup
  scale multiplies two specific terms:
  - `efficiency_cost` (or `efficiency_term`, check the
    name)
  - `precision_reward`
  Nothing else gets scaled.
- `agents/ppo_trainer.py` — the per-episode counter
  that distinguishes PPO training episodes from BC
  pretrain episodes. Warmup index counts PPO episodes
  only.
- `CLAUDE.md` "Reward function: raw vs shaped",
  especially "Symmetry around random betting" — the
  warmup MUST preserve zero-mean properties of the
  symmetrically-centred terms. Scaling a zero-mean
  term by a scalar preserves its zero-mean property,
  so this is safe, but note it in the CLAUDE.md entry.

## Why this is necessary

Per `purpose.md`, the 2026-04-21 `arb-curriculum-probe`
Validation found 7/66 agents with positive cumulative
cash P&L but only 1/66 with positive `total_reward`.
The gap came from `efficiency_cost × bet_count` plus
the centred `(precision − 0.5) × precision_bonus`
overwhelming positive raw_pnl. Both terms are
calibrated for late-training, disciplined policies. In
the first ~10 episodes post-BC the policy is confident
on oracle targets but hasn't calibrated precision yet
— so it explores at exploration-level bet counts and
mediocre precision, and the shaping terms treat that
exploration as a failure.

Warming up those two specific terms linearly across the
first N episodes lets the agent learn without being
punished for the exploration shape it NEEDS to reach
the stable region. Force-close (Session 01) bounds the
variance the agent navigates; warmup clears the
penalty floor for reaching the basin.

**Why only these two terms:** the other shaping
contributions either reward behaviour we want (MTM,
matured-arb, early_pick) or penalise behaviour we
definitely don't want at any episode (naked losses,
drawdowns, inactivity). Warming only the penalties
avoids rewarding "do nothing" — positive gradient for
good arbing from ep 1.

## What to do

### 1. Scope confirmation (5 min)

Read `env/betfair_env.py` around the shaped-sum site
Session 01 left us at. Confirm:

- `efficiency_cost` and `precision_reward` are
  computed separately and added into `shaped`.
- The `shaped` formula is additive (not multiplicative
  — scaling by a scalar works).
- `episode_idx` is accessible at settle time, OR you
  can plumb it in cleanly (from the trainer; see
  step 3).

If the terms have been refactored into a different
structure, PAUSE and update the prompt + hard
constraints before editing.

### 2. Env-side: add the config read

`BetfairEnv.__init__`:

```python
# Shaped-penalty warmup (plans/arb-signal-cleanup,
# Session 02, 2026-04-21). Linearly scales
# efficiency_cost and precision_reward from 0 to 1
# across the first N PPO episodes. Zero-mean terms
# (efficiency, precision centred at 0.5) scaled by a
# scalar preserve their zero-mean property — still
# safe per CLAUDE.md "Symmetry around random betting".
# Default 0 = disabled = byte-identical.
# See hard_constraints.md s19-s23.
self._shaped_penalty_warmup_eps: int = int(
    training_cfg.get("shaped_penalty_warmup_eps", 0)
)
```

(Adjust `training_cfg` path to match how config flows
— it's `config["training"][...]` accessible during
BetfairEnv init, or a plumbed parameter from the
caller. Check Session 01 for whether
`curriculum_day_order` reads the same way — mirror
that.)

### 3. Plumb `episode_idx` if not already present

If the env already receives the episode index at
settle time, skip. If not, add a parameter to the
settlement call or expose an episode-scoped counter on
BetfairEnv that the trainer updates before each
rollout:

```python
# In BetfairEnv or at the env's reset interface:
def set_episode_idx(self, episode_idx: int) -> None:
    """Called by PPOTrainer before each rollout.

    BC pretrain episodes do NOT increment this; only
    PPO rollout episodes do.
    """
    self._episode_idx = int(episode_idx)
```

In `agents/ppo_trainer.py`, find the per-episode
rollout loop. Before each rollout, call
`env.set_episode_idx(ppo_episode_idx)`. The counter
starts at 0 for ep1, 1 for ep2, etc.

Crucial: BC episodes MUST NOT increment this. Find
where BC runs (grep `bc_pretrain_steps`) and confirm
the PPO episode counter is separate from any BC step
counter.

### 4. Env-side: apply the scale

In `_settle_current_race` where `shaped` is summed:

```python
# Shaped-penalty warmup scale (plans/arb-signal-
# cleanup Session 02). Scales efficiency_cost and
# precision_reward only.
if self._shaped_penalty_warmup_eps > 0:
    warmup_scale = min(
        1.0,
        self._episode_idx / max(1, self._shaped_penalty_warmup_eps),
    )
else:
    warmup_scale = 1.0

# NOTE: efficiency_cost is SUBTRACTED in the shaped
# sum (it's a cost), precision_reward is ADDED (it's
# a reward). Both get scaled by warmup_scale because
# "warmup" means "less signal" from both.
scaled_efficiency_cost = efficiency_cost * warmup_scale
scaled_precision_reward = precision_reward * warmup_scale

shaped = (
    early_pick_bonus
    + scaled_precision_reward             # was precision_reward
    - scaled_efficiency_cost              # was efficiency_cost
    + drawdown_term
    + spread_cost_term
    + inactivity_term
    + naked_penalty_term
    + early_lock_term
    + matured_arb_term
    # + MTM per-step is elsewhere — not affected
)
```

Verify the sign conventions match the existing code
before editing. `precision_reward` might already be
signed (positive above 0.5, negative below) — if so,
the scalar multiplication still preserves the sign
and zero-mean property.

### 5. Telemetry

In `_get_info`:

```python
"shaped_penalty_warmup_scale": warmup_scale,
"shaped_penalty_warmup_eps":
    self._shaped_penalty_warmup_eps,
```

`_log_episode` mirrors both.

### 6. config.yaml

```yaml
training:
  ... existing ...
  # Shaped-penalty warmup (plans/arb-signal-cleanup
  # Session 02, 2026-04-21). Scales efficiency_cost
  # and precision_reward from 0 to 1 across the first
  # N PPO episodes. BC pretrain episodes don't count.
  # Default 0 = disabled. Typical active value: 10.
  # See plans/arb-signal-cleanup/purpose.md.
  shaped_penalty_warmup_eps: 0
```

### 7. Tests — `tests/arb_signal_cleanup/test_shaped_penalty_warmup.py`

Per `hard_constraints.md` §32. Six tests:

1. **Default (warmup=0) byte-identical.** Scripted
   rollout with `warmup_eps=0`; assert per-episode
   (raw, shaped, total) match pre-change to
   float-eps. (Use a reference rollout fixture or
   compare against a hand-computed expected value.)
2. **Linear ramp.** `warmup_eps=10`; call
   `env.set_episode_idx(idx)` with
   `idx ∈ {0, 5, 9, 10, 20}`. Assert
   `warmup_scale = 0.0, 0.5, 0.9, 1.0, 1.0` on the
   emitted `info["shaped_penalty_warmup_scale"]`.
3. **Only two terms affected.** Scripted race with
   known non-zero `efficiency_cost = 2.0`,
   `precision_reward = 3.0`, `early_pick_bonus =
   1.0`, all others zero. At `warmup_scale = 0.5`:
   assert `shaped == 1.0 + 0.5 * 3.0 − 0.5 * 2.0 =
   1.5`. At `warmup_scale = 1.0`: `shaped = 1.0 +
   3.0 − 2.0 = 2.0`.
4. **JSONL field present.** Post-episode row carries
   both `shaped_penalty_warmup_scale` and
   `shaped_penalty_warmup_eps`.
5. **No cliff at warmup+1.** Scripted 15-episode
   run with `warmup_eps=10` and identical race
   content per episode. Extract `shaped` per episode;
   the delta between ep10 and ep11 should match the
   delta between ep9 and ep10 (both should be a
   `0.1 × terms` step — linear ramp). No
   discontinuity.
6. **Invariant parametrised.** Extend the existing
   invariant test (or a sibling in
   `tests/arb_signal_cleanup/`) to cover
   `shaped_penalty_warmup_eps ∈ {0, 5}` at
   `episode_idx ∈ {0, 2, 4, 5, 10}`. Stacks with
   Session 01's parametrisation — include
   `force_close_before_off_seconds ∈ {0, 30}` so the
   combined matrix is exercised.

### 8. CLAUDE.md

Under "Reward function: raw vs shaped", new
subsection at the end (after the naked-loss annealing
entry):

```
### Shaped-penalty warmup (2026-04-21)

Plan-level `training.shaped_penalty_warmup_eps`
linearly scales `efficiency_cost` and
`precision_reward` from 0 → 1 across the first N PPO
rollout episodes. Default 0 = no-op (byte-identical).

    if episode_idx < warmup_eps:
        scale = episode_idx / warmup_eps
    else:
        scale = 1.0

    shaped = early_pick_bonus
           + scale * precision_reward
           - scale * efficiency_cost
           + other terms unchanged

BC pretrain episodes do NOT count toward the warmup
index. Only PPO rollout episodes.

Motivation: the 2026-04-21 `arb-curriculum-probe`
Validation observed 7/66 agents with positive
cumulative cash P&L but only 1/66 with positive
`total_reward` — the efficiency and precision terms
overwhelmed early cash P&L when the policy's post-BC
exploration shape had high bet counts and un-
calibrated precision. Warmup gives the agent a
penalty-lite window to learn before the full
shaping discipline kicks in.

Why only these two terms: other shaping contributions
either reward behaviour we want (MTM, matured-arb,
early_pick) or penalise behaviour we definitely
don't want at any episode (naked losses, drawdowns,
inactivity). Warming only the penalties avoids
rewarding "do nothing".

Zero-mean property preserved: `precision_reward` is
centred at 0.5 symmetrically; scaling by a scalar
keeps it zero-mean. `efficiency_cost` is symmetric
around the per-bet count expected under a random
policy; scaling preserves that too.

Reward-scale change: `shaped_penalty_warmup_eps > 0`
changes per-episode `shaped_bonus` magnitude during
the warmup window. Scoreboard rows from runs with
warmup active are NOT comparable to pre-plan rows on
`shaped_bonus` during ep1..warmup_eps; `raw_pnl_reward`
is unchanged. See `plans/arb-signal-cleanup/`.
```

### 9. Full-suite check

```
pytest tests/arb_signal_cleanup/ -x
```

Then — ONLY if no training is active —
`pytest tests/ -q --timeout=120`.

### 10. Commit

```
feat(env): shaped-penalty warmup across first N episodes (default disabled)

Linear ramp on efficiency_cost and precision_reward
from 0 to 1 across the first N PPO rollout episodes.
BC pretrain episodes don't count. Default 0 = no-op.

Why: 2026-04-21 arb-curriculum-probe Validation found
7/66 agents cash-positive but only 1/66 reward-
positive. Post-BC the policy's exploration shape
(high bet count, un-calibrated precision) is exactly
what efficiency_cost and precision_reward penalise at
full strength. Warmup clears the penalty floor for the
first ~10 episodes while preserving full strength on
all other terms (naked penalty, drawdown, inactivity,
MTM, matured-arb, early_pick).

Zero-mean property preserved: scaling a symmetric
zero-mean term by a scalar keeps it zero-mean.

Changes:
- Env reads training.shaped_penalty_warmup_eps
  (default 0).
- Env exposes set_episode_idx(idx) hook; trainer
  calls it before each PPO rollout. BC path does NOT.
- _settle_current_race applies warmup_scale to the
  two terms only.
- EpisodeStats + JSONL row gain
  shaped_penalty_warmup_scale,
  shaped_penalty_warmup_eps.

Tests: 6 in tests/arb_signal_cleanup/
test_shaped_penalty_warmup.py. Invariant test
parametrised over warmup_eps in {0, 5} x episode_idx
in {0, 2, 4, 5, 10}, stacking with Session 01's
force-close and alpha_lr parametrisation.

CLAUDE.md: new dated subsection under "Reward
function: raw vs shaped".

Not changed: other shaping terms, PPO stability
defences, controller, matcher, BC.

Per plans/arb-signal-cleanup/hard_constraints.md s19-s23.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT scale any term other than `efficiency_cost`
  and `precision_reward`. The other terms' full-
  strength behaviour is load-bearing.
- Do NOT count BC pretrain episodes toward the warmup
  index. The warmup is about PPO rollouts, not total
  model updates.
- Do NOT make the warmup non-linear without a test
  matrix. Linear is the simplest shape that avoids an
  explore-cliff; if the probe shows a cliff at
  warmup+1 in spite of linearity, that's a Session
  02b tuning question, not this commit.
- Do NOT set `shaped_penalty_warmup_eps` non-zero in
  `config.yaml`. The default is disabled; the probe
  plan sets it to 10 on the relevant cohorts.
- Do NOT run the full pytest suite during active
  training.

## After Session 02

1. Append a progress entry to
   [`../progress.md`](../progress.md).
2. Note any surprises in `lessons_learnt.md` (e.g.
   episode_idx plumbing required a bigger refactor
   than the prompt suggests).
3. Hand back for Session 03 (plan draft + validator +
   launch).
