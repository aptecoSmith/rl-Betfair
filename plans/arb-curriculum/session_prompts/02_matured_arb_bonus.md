# Session 02 prompt — Matured-arb bonus (knob at 0 default)

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "Matured-arb bonus"
  subsection of the design sketch; failure-mode entry on
  bonus dominating P&L.
- [`../hard_constraints.md`](../hard_constraints.md). §10
  (zero-mean correction), §11 (default 0), §12 (cap),
  §24 (invariant parametrised),
  §25 (JSONL field), §28 (tests).
- [`../master_todo.md`](../master_todo.md) — Session 02
  deliverables.
- `env/betfair_env.py` — the file being edited. Locate
  `_settle_current_race` and the per-race shaping
  accumulator.
- `CLAUDE.md` "Reward function: raw vs shaped" — especially
  "Symmetry around random betting". The bonus MUST be
  zero-mean corrected to preserve the invariant from that
  section.
- `plans/reward-densification/session_prompts/01_mark_to_market_scaffolding.md`
  — structural template for adding a new shaped-reward knob.

## Why this is necessary

Per `purpose.md`, per-pair feedback addresses credit-
assignment blur (Force 2). The race-settle signal can't
distinguish which pair was good; a per-pair bonus on pair
MATURATION (second leg filled, regardless of P&L sign) does.

Zero-mean correction is critical: an uncorrected per-pair
bonus trains the agent to maximise pairs at any cost, which
is exactly the "active-bleeding" failure mode we've been
fighting.

## Locate the code

```
grep -n "arbs_completed\|arbs_closed\|arbs_naked" env/betfair_env.py
grep -n "scalping_locked_pnl\|shaped\s*=\|shaped\s*\+=" env/betfair_env.py
grep -n "_REWARD_OVERRIDE_KEYS" env/betfair_env.py
grep -n "_REWARD_GENE_MAP" agents/ppo_trainer.py
```

Confirm before editing:

1. `_settle_current_race` computes `scalping_arbs_completed`,
   `scalping_arbs_closed`, `scalping_arbs_naked` before
   building `shaped`. The matured count is
   `completed + closed` (both had their second leg fill).
2. The `shaped` accumulator is built additively in the same
   block; that's where the matured-arb term goes.
3. `_REWARD_OVERRIDE_KEYS` is the set we whitelist gene
   names against. `_REWARD_GENE_MAP` is the trainer-side
   map from gene name → reward-config key.

## What to do

### 1. Env-side: add the config read

In `BetfairEnv.__init__`, near where other reward-config
knobs land:

```python
# Matured-arb bonus (plans/arb-curriculum, Session 02,
# 2026-04-19). Shaped contribution per pair that matured
# (second leg filled - naturally or via close_signal).
# Zero-mean corrected so random policies don't harvest
# free reward. See hard_constraints.md s10-s12.
self._matured_arb_bonus_weight: float = float(
    reward_cfg.get("matured_arb_bonus_weight", 0.0)
)
self._matured_arb_bonus_cap: float = float(
    reward_cfg.get("matured_arb_bonus_cap", 10.0)
)
self._matured_arb_expected_random: float = float(
    reward_cfg.get("matured_arb_expected_random", 2.0)
)
```

### 2. Env-side: add the shaping term

In `_settle_current_race`, after `scalping_arbs_completed`
and `scalping_arbs_closed` are finalised and before
`shaped` is summed:

```python
matured_arb_term = 0.0
if self.scalping_mode and self._matured_arb_bonus_weight > 0.0:
    n_matured = scalping_arbs_completed + scalping_arbs_closed
    # Zero-mean correction: subtract the expected random-
    # policy pair count so free reward can't accrue.
    raw_matured_contribution = (
        self._matured_arb_bonus_weight
        * (n_matured - self._matured_arb_expected_random)
    )
    # Cap in both directions so one runaway race can't
    # dominate total shaped signal.
    matured_arb_term = float(np.clip(
        raw_matured_contribution,
        -self._matured_arb_bonus_cap,
        +self._matured_arb_bonus_cap,
    ))

shaped = (
    early_pick_bonus
    + precision_reward
    - efficiency_cost
    + drawdown_term
    + spread_cost_term
    + inactivity_term
    + naked_penalty_term
    + early_lock_term
    + matured_arb_term            # NEW
)
```

Record the active weight for telemetry:

```python
# In _get_info(), add:
"matured_arb_bonus_active": self._matured_arb_bonus_weight,
```

### 3. Whitelist the gene

`env/betfair_env.py`:

```python
_REWARD_OVERRIDE_KEYS: frozenset[str] = frozenset({
    ...existing keys...
    "mark_to_market_weight",
    # Arb-curriculum Session 02 (2026-04-19): per-pair
    # shaped bonus on pair maturation. Whitelisted so a
    # per-agent gene override flows through.
    "matured_arb_bonus_weight",
})
```

### 4. Trainer-side: add to gene map

`agents/ppo_trainer.py`:

```python
_REWARD_GENE_MAP: dict[str, tuple[str, ...]] = {
    ...existing entries...
    "mark_to_market_weight": ("mark_to_market_weight",),
    # Arb-curriculum Session 02 (2026-04-19).
    "matured_arb_bonus_weight": ("matured_arb_bonus_weight",),
}
```

### 5. EpisodeStats + _log_episode

`EpisodeStats` gains:

```python
matured_arb_bonus_active: float = 0.0
```

`_build_episode_stats` reads
`info.get("matured_arb_bonus_active", 0.0)`.

`_log_episode` writes it into the JSONL row as an optional
field.

### 6. config.yaml

```yaml
reward:
  ... existing ...
  # Arb-curriculum Session 02 (2026-04-19). Per-pair
  # shaped bonus on pair maturation (completed or
  # agent-closed). Zero-mean corrected against
  # matured_arb_expected_random (default 2 pairs/race).
  # Bounded by matured_arb_bonus_cap. Default 0.0 is a
  # no-op. See plans/arb-curriculum/purpose.md.
  matured_arb_bonus_weight: 0.0
  matured_arb_bonus_cap: 10.0
  matured_arb_expected_random: 2.0
```

### 7. Tests — `tests/arb_curriculum/test_matured_arb_bonus.py`

Per §28:

1. **Weight=0 byte-identical.** Scripted rollout with
   `weight=0` matches pre-change per-episode (raw, shaped,
   total) to float-eps.
2. **Bonus emitted only on maturation.** A race where all
   pairs are naked → shaped term should be
   `weight * (0 - expected_random)` = negative-then-clipped.
3. **Zero-mean at the expected count.** Scripted race with
   exactly `expected_random` matured pairs → shaped term
   is 0.
4. **Cap enforced.** Force a race with 100 matured pairs;
   shaped term equals `+cap`.
5. **Symmetry.** Force a race with -100 offset (0 pairs,
   weight high); shaped term equals `-cap`.
6. **JSONL field present.** Post-episode row carries
   `matured_arb_bonus_active`.
7. **Gene passthrough.** Running env with
   `reward_overrides={"matured_arb_bonus_weight": 0.5}`
   sets `_matured_arb_bonus_weight` to 0.5.
8. **Invariant parametrised.** Extend
   `test_invariant_raw_plus_shaped_equals_total_reward` to
   cover `matured_arb_bonus_weight ∈ {0.0, 1.0}`.

### 8. CLAUDE.md

Under "Reward function: raw vs shaped", new subsection:

```
### Matured-arb bonus (2026-04-19)

A small shaped reward per pair that matured (second leg
filled, naturally or via close_signal), zero-mean
corrected against an expected random-policy pair count.
Shaped contribution per race:

    raw_bonus = weight * (n_matured - expected_random)
    matured_arb_term = clip(raw_bonus, -cap, +cap)

Default weight 0.0 = no-op. When > 0, the bonus rewards
the SKILL of closing pair lifecycles (independent of
P&L sign), not the outcome. Cap prevents a runaway race
from dominating shaped reward. See
plans/arb-curriculum/purpose.md for the credit-
assignment motivation.
```

### 9. Full-suite check

```
pytest tests/arb_curriculum/ -x
```

Then — ONLY if no training is active —
`pytest tests/ -q --timeout=120`. Per operator directive.

### 10. Commit

```
feat(env): per-pair matured-arb shaped bonus (weight=0 default)

Add a shaped reward per pair whose second leg fills -
naturally (scalping_arbs_completed) or via close_signal
(scalping_arbs_closed). Zero-mean corrected against an
expected random-policy pair count so a random policy's
expected bonus is zero. Cap prevents one outlier race
from dominating the shaped signal.

Formula (hard_constraints s10-s12):

  raw_bonus = weight * (n_matured - expected_random)
  matured_arb_term = clip(raw_bonus, -cap, +cap)

Default weight 0.0 -> byte-identical to pre-change.

Why: 2026-04-19 reward-densification diagnosis: per-race
settlement P&L can't distinguish good pairs from bad
within the same episode. A per-pair bonus on pair
MATURATION (not P&L) gives per-pair feedback that isn't
blurred by GAE, and rewards the skill of closing
lifecycles. Zero-mean correction prevents random-policy
drift toward "complete tiny unprofitable arbs for
reward".

Changes:
- Env reads matured_arb_bonus_weight / _cap /
  _expected_random from reward config.
- Shaping term added to _settle_current_race's shaped sum.
- Whitelisted in _REWARD_OVERRIDE_KEYS and
  _REWARD_GENE_MAP so per-agent genes flow through.
- EpisodeStats + JSONL row gain matured_arb_bonus_active.

Tests: 8 in tests/arb_curriculum/test_matured_arb_bonus.py.
Invariant test parametrised over weight in {0.0, 1.0}.

Not changed: matcher, controller, PPO stability defences,
raw P&L accounting, action/obs schemas, other reward knobs.

Per plans/arb-curriculum/hard_constraints.md s10-s12.

pytest tests/arb_curriculum/ -x: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT emit the bonus in raw. It's training-signal,
  not cash — the whole point is it's zero-mean.
- Do NOT skip zero-mean correction. A fixed bonus per
  matured pair is exactly the shaping bug CLAUDE.md
  "Symmetry around random betting" warns against.
- Do NOT run the full pytest suite during active training.

## After Session 02

1. Append a progress entry.
2. Hand back for Session 03 (naked-loss annealing).
