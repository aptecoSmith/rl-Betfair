# Session 03 prompt — Naked-loss annealing knob + generation schedule

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "Naked-loss annealing"
  subsection of the design sketch; failure-mode entry on
  residual reward-shape dependency after anneal completes.
- [`../hard_constraints.md`](../hard_constraints.md). §13
  (env-side scaler semantics), §14 (plan-level
  `naked_loss_anneal`), §15 (scoreboard comparability),
  §24 (invariant parametrised), §29 (tests).
- [`../master_todo.md`](../master_todo.md) — Session 03
  deliverables.
- `env/betfair_env.py`'s `_settle_current_race` — the
  scalping raw reward split. Locate where per-pair naked
  P&L is summed into `race_pnl`.
- `CLAUDE.md` "Reward function: raw vs shaped" — scalping
  raw reward semantics.
- `training/population_manager.py` (or whichever module
  owns generation progression) — where the per-generation
  anneal factor gets applied to each agent's genes.

## Why this is necessary

Per `purpose.md`, random arbing is expected-negative
largely because naked losses land at full cash value.
Annealing the loss side downward EARLY in training
reduces the sign-inversion at init (Force 1) while
preserving the sign of the signal: nakeds still cost
money, just less early so the policy can survive long
enough to learn entry selection.

## Design recap (from purpose.md / hard_constraints)

Env-side per-race raw reward, scalping mode, is:

```
race_pnl = scalping_locked_pnl
         + scalping_closed_pnl
         + sum(per_pair_naked_pnl)
```

After this session:

```
scaled_naked_sum = sum(
    min(0, per_pair_naked_pnl) * naked_loss_scale
    + max(0, per_pair_naked_pnl)
)
race_pnl = scalping_locked_pnl
         + scalping_closed_pnl
         + scaled_naked_sum
```

**Asymmetry is deliberate** — loss side anneals, win side
stays at full. Reason: we want to bootstrap the policy
past the naked valley without teaching it that naked
winners are free reward. Winners are still clipped 95% in
shaped (from `naked-clip-and-stability`).

Generation-level annealing (`naked_loss_anneal`) linearly
interpolates the per-agent `naked_loss_scale` gene toward
1.0 over the configured generation window.

## Locate the code

```
grep -n "_compute_scalping_reward_terms\|get_naked_per_pair_pnls" env/betfair_env.py env/bet_manager.py
grep -n "scalping_locked_pnl\|scalping_closed_pnl\|naked_per_pair\|race_pnl" env/betfair_env.py
grep -n "_REWARD_OVERRIDE_KEYS" env/betfair_env.py
grep -n "_REWARD_GENE_MAP" agents/ppo_trainer.py
grep -rn "def apply_genes\|class PopulationManager\|def advance_generation" training/
```

Confirm before editing:

1. `_compute_scalping_reward_terms(race_pnl, naked_per_pair,
   n_close_signal_successes)` is the helper that splits
   race reward between raw and shaped. Its signature needs
   to accept `naked_loss_scale`.
2. `bm.get_naked_per_pair_pnls(market_id)` returns the
   per-pair naked cash flows — this is the list we
   scale.
3. PopulationManager (or equivalent) has a hook for
   per-generation gene transformations. If it does,
   annealing lives there; if not, we wire it in
   `training/worker.py` at the per-agent spawn path.

## What to do

### 1. Env-side: gene read + scaler

In `BetfairEnv.__init__`:

```python
# Naked-loss annealing scale (plans/arb-curriculum,
# Session 03, 2026-04-19). Per-pair scalar on the LOSS
# side of naked cash flows; winners unchanged. 1.0 =
# byte-identical to pre-change. < 1.0 = bootstrap the
# policy past the naked valley. Generation-level
# annealing applies this gene on top of the schedule.
# See hard_constraints.md s13-s15.
self._naked_loss_scale: float = float(
    reward_cfg.get("naked_loss_scale", 1.0)
)
# Clamp defensively so a bad gene can't zero out
# unrelated pair P&L.
if not (0.0 <= self._naked_loss_scale <= 1.0):
    logger.warning(
        "naked_loss_scale=%s out of [0,1]; clamping",
        self._naked_loss_scale,
    )
    self._naked_loss_scale = float(
        np.clip(self._naked_loss_scale, 0.0, 1.0)
    )
```

### 2. Env-side: modify `_compute_scalping_reward_terms`

Add `naked_loss_scale` parameter. Inside the function,
when building `race_pnl`:

```python
def _compute_scalping_reward_terms(
    *, race_pnl: float,
    naked_per_pair: list[float],
    n_close_signal_successes: int,
    naked_loss_scale: float = 1.0,  # NEW
) -> tuple[float, float]:
    # Apply scale to losses only. Winners untouched.
    # scaled_naked = sum(
    #     min(0, x) * scale + max(0, x) for x in naked_per_pair
    # )
    # Adjusted race_pnl (raw) = original - (1-scale) * sum(losses)
    loss_sum = sum(min(0.0, x) for x in naked_per_pair)
    # Original formula kept losses at full; new formula:
    race_pnl_adjusted = race_pnl - (1.0 - naked_loss_scale) * loss_sum
    # (note: loss_sum <= 0, so subtracting (1-scale)*loss_sum
    # REDUCES the magnitude of the loss, as intended.)
    ...
```

Update the caller in `_settle_current_race` to pass
`naked_loss_scale=self._naked_loss_scale`.

Record `naked_loss_scale_active` in `_get_info()`.

### 3. Whitelist + gene map

Same pattern as Session 02. Add to
`_REWARD_OVERRIDE_KEYS` (env) and `_REWARD_GENE_MAP`
(trainer).

### 4. Plan-level annealing schedule

Extend the plan JSON schema with an optional
`naked_loss_anneal` field:

```json
"naked_loss_anneal": {
  "start_gen": 0,
  "end_gen": 2
}
```

In the population-manager's per-generation agent-prep
hook (or the trainer's gene-materialisation step),
compute:

```python
def anneal_factor(current_gen: int,
                   start: int, end: int) -> float:
    """Interpolation progress in [0, 1]."""
    if end <= start:
        return 1.0  # degenerate -> no annealing
    if current_gen <= start:
        return 0.0
    if current_gen >= end:
        return 1.0
    return (current_gen - start) / (end - start)

def effective_naked_loss_scale(
    gene_value: float,
    current_gen: int,
    schedule: dict | None,
) -> float:
    if schedule is None:
        return gene_value
    p = anneal_factor(
        current_gen, schedule["start_gen"], schedule["end_gen"],
    )
    return gene_value + (1.0 - gene_value) * p
```

Apply `effective_naked_loss_scale` to each agent's hp
dict before passing to `PPOTrainer` / the env. This
transforms the gene value BEFORE the trainer sees it so
the downstream plumbing doesn't need to know about
annealing.

### 5. config.yaml + schema

```yaml
reward:
  ...
  # Arb-curriculum Session 03 (2026-04-19). Per-pair
  # loss-side scalar on naked cash flows (winners
  # untouched). 1.0 = byte-identical. Per-agent gene;
  # plan-level anneal schedule may interpolate toward
  # 1.0 across generations.
  naked_loss_scale: 1.0
```

Gene schema (wherever HP ranges are declared for default
GA runs): add `naked_loss_scale` with range
`[0.05, 1.0]` and a sensible default mutation rate.

### 6. EpisodeStats + _log_episode

Add `naked_loss_scale_active: float = 1.0` to
`EpisodeStats`. Populate from
`info.get("naked_loss_scale_active", 1.0)`. Write to
JSONL.

### 7. Tests — `tests/arb_curriculum/test_naked_loss_annealing.py`

Per §29:

1. **Scale 1.0 byte-identical.** Scripted rollout with
   `scale=1.0` matches pre-change.
2. **Scale 0.5 halves loss magnitude.** Scripted race
   with a known naked-loss sum; adjusted `race_pnl`
   reflects exactly the expected reduction.
3. **Scale 0.0 zeros out losses but preserves wins.**
   Winners unchanged; losses zero.
4. **Winners side untouched across all scales.** Race
   with only naked winners; `race_pnl` independent of
   `naked_loss_scale`.
5. **Invariant preserved at scale<1.** Extend
   `test_invariant_raw_plus_shaped_equals_total_reward`
   to cover `naked_loss_scale ∈ {0.5, 1.0}`.
6. **Annealing interpolation.** Unit test on
   `effective_naked_loss_scale` across 5 generations of
   a `{start:0, end:4}` schedule.
7. **Annealing degenerate.** `{start:0, end:0}` →
   always 1.0 (no-annealing).
8. **JSONL field present.**

### 8. CLAUDE.md

```
### Naked-loss annealing (2026-04-19)

scaled_naked_sum = sum(
    min(0, p) * naked_loss_scale   # loss side scaled
    + max(0, p)                    # win side untouched
    for p in per_pair_naked_pnl
)
race_pnl = scalping_locked_pnl
         + scalping_closed_pnl
         + scaled_naked_sum

Default scale 1.0 = byte-identical. Plan-level
``naked_loss_anneal: {start_gen, end_gen}`` linearly
interpolates each agent's effective scale toward 1.0 over
the window. Used to bootstrap the policy past the naked
valley early in training.

Scoreboard comparability: scale<1 runs are NOT
comparable to scale=1 runs on raw_pnl_reward. The loss
side is intentionally undercounted during annealing; the
agent pays full price once end_gen is reached.
```

### 9. Full-suite check (NO active training)

```
pytest tests/arb_curriculum/ -x
pytest tests/ -q --timeout=120
```

### 10. Commit

```
feat(env): per-pair naked-loss scale gene + generation-level annealing

Add a loss-side scalar (naked_loss_scale, [0, 1]) on per-
pair naked cash flows inside scalping _race_pnl. Winners
side untouched. Plan-level ``naked_loss_anneal`` schedule
interpolates each agent's effective scale toward 1.0
across generations; default disabled (no-anneal; gene at
full effect).

Why: 2026-04-19 reward-densification diagnosis. Random
arbing is expected-negative at init primarily because
naked losses land at full cash value while wins are
commission-capped. Annealing the LOSS side early bootstraps
the policy past the naked valley without rewarding lucky
directional wins. Asymmetric by design -- wins stay at
full value (still 95% clipped in shaped).

Formula:

  scaled_naked_sum = sum(
      min(0,p) * naked_loss_scale + max(0,p)
      for p in per_pair_naked_pnl
  )
  race_pnl = scalping_locked_pnl + scalping_closed_pnl
           + scaled_naked_sum

Annealing:

  effective_scale(gen) = gene_value
      + (1 - gene_value) * progress(gen, start, end)

Default {start:0, end:0} -> no anneal; effective_scale
equals gene_value.

Changes:
- Env reads naked_loss_scale from reward config;
  _compute_scalping_reward_terms accepts it.
- Whitelisted in _REWARD_OVERRIDE_KEYS and
  _REWARD_GENE_MAP.
- Plan JSON schema gains optional naked_loss_anneal.
- Trainer / population manager apply effective_scale
  per-generation before the env sees the gene.
- EpisodeStats + JSONL row gain naked_loss_scale_active.

Tests: 8 in tests/arb_curriculum/test_naked_loss_annealing.py.
Invariant parametrised over naked_loss_scale in
{0.5, 1.0}.

Reward-scale change for scale<1: raw_pnl_reward on those
runs is NOT comparable to scale=1 rows. Documented in
CLAUDE.md.

Per plans/arb-curriculum/hard_constraints.md s13-s15.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT scale the winner side. Asymmetric anneal is the
  design; symmetric undercounting of both sides would let
  the agent escape the loss-signal entirely.
- Do NOT apply the anneal inside the env. The env takes a
  single `naked_loss_scale` value; the generation-level
  interpolation is the trainer/population manager's job
  so downstream code doesn't need to track gen index.
- Do NOT omit the clamp to [0, 1]. A gene mutation could
  produce values outside this range; defensive clamp is
  cheap correctness.

## After Session 03

1. Append a progress entry.
2. Hand back for Session 04 (BC pretrainer — the big one).
