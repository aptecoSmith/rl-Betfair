# Session 02 prompt — Plan-level default weight

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "Picking the default
  weight" subsection; 0.05 first-cut rationale.
- [`../hard_constraints.md`](../hard_constraints.md). §10
  (knob / default), §11 (not a gene in this plan),
  §17 (pin test for the chosen value), §22 (Session 02
  blocks if Session 01 didn't land).
- [`../master_todo.md`](../master_todo.md) — Session 02
  deliverables.
- `config.yaml` — the file being edited. Locate the
  `reward:` block; the new key slots in alphabetically
  next to other reward-shaping weights.

## Locate the code

```
grep -n "^reward:" config.yaml
grep -n "early_pick_bonus\|efficiency_penalty\|fill_prob_loss_weight" config.yaml
grep -n "mark_to_market" config.yaml env/betfair_env.py
```

Confirm before editing:

1. Session 01 has landed (Session 01 commit present on
   `master`, hard_constraints §22). If not, STOP and
   complete Session 01 first.
2. `mark_to_market_weight` appears in `env/betfair_env.py`
   reading from `config.reward.mark_to_market_weight`.
3. The `reward` block in `config.yaml` has the expected
   shape with sibling weights like
   `early_pick_bonus_min`, `efficiency_penalty`, etc.

## What to do

### 1. Add the default to `config.yaml`

In `config.yaml` under the `reward:` block, add:

```yaml
reward:
  # ... existing keys ...
  # Per-step mark-to-market shaping weight
  # (plans/reward-densification, Session 02, 2026-04-19).
  # 0.0 = no-op (rollouts byte-identical to pre-change);
  # 0.05 is the first-cut default calibrated so the
  # cumulative shaped MTM across a race is order-of-
  # magnitude-comparable with the race's raw P&L without
  # dominating it. See purpose.md "Picking the default
  # weight" for the calibration argument; see §11 of
  # hard_constraints.md for why this is NOT a GA gene
  # in this plan.
  mark_to_market_weight: 0.05
```

Preserve the alphabetical or existing ordering convention
of the file — don't reorder other keys.

### 2. Pin the default in tests

Add to `tests/test_config_defaults.py` (or create it if it
doesn't exist):

```python
def test_mark_to_market_weight_default_matches_session_02():
    """Default ``reward.mark_to_market_weight`` is 0.05
    (plans/reward-densification, Session 02, 2026-04-19).
    The original 0.0 default from Session 01 was a no-op
    migration; 0.05 engages the mechanism for any run
    that doesn't override via hp. Any change to this
    default needs to co-land with a plan update."""
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("config.yaml").read_text())
    assert config["reward"]["mark_to_market_weight"] == 0.05
```

If a `test_config_defaults.py` already exists, add the test
there following the file's conventions for similar pinned
values.

### 3. CLAUDE.md

Add a dated paragraph under the existing "Per-step mark-to-
market shaping (2026-04-19)" subsection from Session 01:

```
**Default weight 0.05 (2026-04-19, Session 02).** MTM deltas
are O(pennies-to-pounds) per tick on typical stakes; 0.05 ×
cumulative |MTM delta| across a race scales the shaped
contribution to order-of-magnitude-comparable with per-race
raw P&L (typical £-5 to £+30 range per race). Too small and
the signal is lost in advantage-normalisation noise; too
large and the policy optimises per-tick flicker at the
expense of settle P&L. The knob is a plan-level default
only — not a GA gene in this plan. See
``plans/reward-densification/purpose.md`` "Picking the
default weight" for the calibration argument.
```

### 4. Full suite

```
pytest tests/ -q
```

Must be green. Regression guards from Session 01 — MTM
formula tests, telescope test, invariant tests — all still
green because the default change is byte-equivalent to
running Session 01's code with `mark_to_market_weight=0.05`,
which Session 01's tests already exercise.

### 5. Commit

```
feat(config): default mark_to_market_weight to 0.05

Session 01 landed the mark-to-market mechanism with
``mark_to_market_weight=0.0`` default (byte-identical
migration). Session 02 engages the mechanism project-wide
by setting the default to 0.05.

Rationale: MTM deltas are O(pennies-to-pounds) per tick on
typical stakes; 0.05 x cumulative ``|MTM delta|`` across a
race scales the shaped contribution to order-of-magnitude-
comparable with per-race raw P&L (typical -£5 to +£30 per
race) without dominating it. Calibration argument in
plans/reward-densification/purpose.md "Picking the default
weight".

Changes:
- config.yaml: reward.mark_to_market_weight: 0.05 (new key,
  preserves existing ordering).
- tests/test_config_defaults.py:
  test_mark_to_market_weight_default_matches_session_02
  pins the value so a future refactor can't silently
  revert.
- CLAUDE.md: dated paragraph documenting the non-zero
  default under the Session-01 "Per-step mark-to-market
  shaping" subsection.

Not changed: mechanism (landed in Session 01), semantics,
formula, clamp bounds, invariant tests, target-entropy
controller, PPO stability defences.

Reward-scale change: runs started after this commit are
NOT byte-identical to pre-change. Per-episode reward
magnitudes differ (raw P&L preserved; shaped_bonus now
includes MTM shaping). Scoreboard rows from pre-change
runs remain comparable on raw_pnl_reward but not on
total_reward. Per hard_constraints.md §19.

pytest tests/ -q: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- No mechanism changes (that's Session 01 territory).
- No GA gene-range edits — per hard_constraints §11 the
  knob stays plan-level in this plan.
- No training-plan redraft — that's Session 03.

## After Session 02

1. Append a `progress.md` entry: commit hash, the 0.05
   default landing, test pin.
2. Hand back for Session 03 (registry reset + probe plan
   redraft). Session 03 is operator-gated.
