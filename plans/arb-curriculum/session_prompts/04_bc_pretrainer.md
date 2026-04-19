# Session 04 prompt — BC pretrainer + trainer integration + entropy-controller handshake

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "BC pretrainer"
  subsection; failure-mode "BC overfits to oracle".
- [`../hard_constraints.md`](../hard_constraints.md).
  §16 (per-agent, never shared), §17 (only signal +
  arb_spread heads), §18 (target-entropy warmup),
  §19 (separate optimiser), §20 (empty-cache skip),
  §30 (tests).
- [`../master_todo.md`](../master_todo.md) — Session 04
  deliverables.
- [`../lessons_learnt.md`](../lessons_learnt.md) — both
  inherited entries.
- `plans/arb-improvements/session_7_bc_pretrainer.md` —
  the 2026-04-14 BC scoping. Reuse the structure; deltas
  captured below.
- `plans/arb-improvements/lessons_learnt.md` — the
  per-agent BC invariant origin.
- `training/arb_oracle.py` (Session 01) — the loader.
- `agents/ppo_trainer.py` — the target-entropy controller
  logic (SGD on `log_alpha`, target 150). Needs to learn
  about `bc_target_entropy_warmup_eps`.
- `CLAUDE.md` "Entropy control — target-entropy
  controller" — the handshake amendment lives adjacent.

## Why this is critical

Every prior attempt at escaping the local minimum has
failed because the gradient at init points away from
arbing. BC starts the policy in a region where the
gradient already points the right way. If this lands and
Sessions 02/03/05 provide the supporting shaping, the
whole plan's premise holds or falls here.

**Two subtle risks to pre-empt:**

1. **BC ↔ controller interaction.** Post-BC entropy is
   low (confident policy); controller target is 150
   (high); controller will aggressively boost `alpha` on
   the first PPO update, undoing BC. Handshake: anneal
   `target_entropy` from post-BC measured up to 150 over
   `bc_target_entropy_warmup_eps` episodes.
2. **Population diversity collapse.** Sharing BC weights
   across agents = same local region for all = GA can't
   recover. **Per-agent BC is a correctness invariant,
   not a style choice.** Test #1 below enforces it.

## Amendments to the 2026-04-14 BC scoping

| Item | 2026-04-14 | 2026-04-19 amendment |
|---|---|---|
| Heads trained | `signal`, `arb_spread` | Same, but action vector is now 7 dims (scalping mode) — must target dims 0 and 4, not 0 and 1. |
| Oracle loader | new module | Reuse `training/arb_oracle.load_samples` from Session 01. |
| Controller interaction | not considered | NEW: `bc_target_entropy_warmup_eps` gene + controller anneal (§18). |
| Integration test | loss decreases | ALSO spy on `_update_reward_baseline` per 2026-04-18 units-mismatch lesson. |
| Schema versions | not mentioned | BC must load oracle cache with matching `obs_schema_version` / `action_schema_version`; hard error on mismatch. |

## Locate the code

```
grep -n "def _update_log_alpha\|_log_alpha\|target_entropy" agents/ppo_trainer.py
grep -n "class PPOTrainer\|def __init__" agents/ppo_trainer.py
grep -n "class PolicyNetwork\|def forward\|action_head\|signal_head" agents/*.py
grep -n "load_samples" training/arb_oracle.py
```

Confirm before editing:

1. `_log_alpha` is a `torch.nn.Parameter`; the controller
   runs once per `_ppo_update`.
2. Policy architectures (`ppo_lstm_v1`,
   `ppo_time_lstm_v1`, `ppo_transformer_v1`) share a
   common action-head structure that BC can target — find
   the named module(s) corresponding to "signal" and
   "arb_spread" outputs.
3. `training/worker.py` (or wherever agents spawn) has a
   clear "before first PPO rollout" point where BC slots
   in.

## What to do

### 1. New module `agents/bc_pretrainer.py`

```python
"""Per-agent behavioural cloning from arb oracle samples.

Hard contracts (plans/arb-curriculum/hard_constraints.md
s16-s20):
- Per-agent; never share weights across the population.
- Only signal and arb_spread heads train; stake,
  aggression, cancel, requote_signal, close_signal heads
  have their parameters frozen (bit-identical before/after).
- Separate optimiser from PPO's Adam.
- Empty oracle cache -> skip cleanly.
- Schema version must match the running env; hard fail
  on mismatch.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from training.arb_oracle import OracleSample, load_samples


@dataclass
class BCLossHistory:
    signal_losses: list[float] = field(default_factory=list)
    arb_spread_losses: list[float] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)
    final_signal_loss: float = 0.0
    final_arb_spread_loss: float = 0.0


class BCPretrainer:
    def __init__(self, lr: float = 3e-4,
                 batch_size: int = 64,
                 signal_weight: float = 1.0,
                 arb_spread_weight: float = 0.1) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.signal_weight = signal_weight
        self.arb_spread_weight = arb_spread_weight

    def pretrain(self, policy, samples: list[OracleSample],
                 n_steps: int) -> BCLossHistory:
        """Pretrain signal + arb_spread heads. Returns loss
        history."""
        if not samples or n_steps <= 0:
            return BCLossHistory()
        # Snapshot non-target parameters for invariant test.
        frozen = [p for name, p in policy.named_parameters()
                  if not _is_bc_target_head(name)]
        frozen_snapshots = [p.detach().clone() for p in frozen]
        for p in frozen:
            p.requires_grad_(False)
        target_params = [p for name, p in policy.named_parameters()
                         if _is_bc_target_head(name)]
        opt = torch.optim.Adam(target_params, lr=self.lr)
        history = BCLossHistory()
        for step in range(n_steps):
            batch = _sample_batch(samples, self.batch_size)
            # build obs, signal targets, arb_spread targets
            ...
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.signal_losses.append(signal_loss.item())
            history.arb_spread_losses.append(arb_spread_loss.item())
            history.total_losses.append(loss.item())
        # Restore requires_grad on frozen params so PPO's
        # optimiser can pick them up later.
        for p in frozen:
            p.requires_grad_(True)
        history.final_signal_loss = (
            history.signal_losses[-1] if history.signal_losses else 0.0
        )
        history.final_arb_spread_loss = (
            history.arb_spread_losses[-1]
            if history.arb_spread_losses else 0.0
        )
        return history


def _is_bc_target_head(name: str) -> bool:
    """Match parameter names for signal and arb_spread
    heads ONLY. Adjust to the actual policy module
    naming — tested via unit test #3."""
    return any(key in name for key in
               ("signal_head", "arb_spread_head"))
```

### 2. Trainer integration in `training/worker.py`

Find the per-agent entrypoint (probably
`run_agent(model_id, plan, ...)` or similar). After the
policy is instantiated and BEFORE the first
`trainer.train()` / `trainer._ppo_update()` call:

```python
from agents.bc_pretrainer import BCPretrainer
from training.arb_oracle import load_samples

bc_steps = int(agent_hp.get("bc_pretrain_steps", 0) or 0)
if scalping_mode and bc_steps > 0:
    # Union of oracle samples across the agent's
    # training dates.
    all_samples = []
    for d in training_dates:
        try:
            all_samples.extend(
                load_samples(d, Path("data/oracle_cache"),
                             strict=True)
            )
        except FileNotFoundError:
            logger.warning(
                "oracle cache missing for %s; skipping", d,
            )
    if not all_samples:
        logger.warning(
            "BC requested (steps=%d) but no oracle "
            "samples for any training date; skipping BC",
            bc_steps,
        )
    else:
        emit_progress({"phase": "bc_warmup",
                       "item": "loading"})
        lr = float(agent_hp.get("bc_learning_rate", 3e-4))
        bc = BCPretrainer(lr=lr)
        history = bc.pretrain(
            trainer.policy, all_samples, n_steps=bc_steps,
        )
        # Stash for the first JSONL row.
        trainer._bc_loss_history = history
        # Measure post-BC entropy so the controller can
        # anneal target_entropy from here up to config.
        trainer._post_bc_entropy = _measure_entropy(
            trainer.policy, all_samples[:256],
        )
        trainer._bc_target_entropy_warmup_eps = int(
            agent_hp.get("bc_target_entropy_warmup_eps", 5)
            or 5
        )
        emit_progress({"phase": "bc_warmup",
                       "item": "complete",
                       "final_signal_loss":
                           history.final_signal_loss,
                       "final_arb_spread_loss":
                           history.final_arb_spread_loss})
```

### 3. Controller handshake in `agents/ppo_trainer.py`

Add fields:

```python
self._bc_target_entropy_warmup_eps: int = 0
self._post_bc_entropy: float | None = None
self._eps_since_bc: int = 0
```

Amend the controller's `target_entropy` access to:

```python
def _effective_target_entropy(self) -> float:
    """Post-BC warmup: anneal from _post_bc_entropy up to
    the configured target over
    _bc_target_entropy_warmup_eps episodes."""
    if (self._post_bc_entropy is None
            or self._eps_since_bc >= self._bc_target_entropy_warmup_eps
            or self._bc_target_entropy_warmup_eps <= 0):
        return float(self._target_entropy)
    p = self._eps_since_bc / self._bc_target_entropy_warmup_eps
    return self._post_bc_entropy + p * (
        float(self._target_entropy) - self._post_bc_entropy
    )
```

Everywhere the controller reads `self._target_entropy`,
route through `self._effective_target_entropy()`. Bump
`self._eps_since_bc` once per episode in the
episode-completion handler.

### 4. EpisodeStats + _log_episode

Add to `EpisodeStats`:

```python
bc_pretrain_steps: int = 0
bc_final_signal_loss: float = 0.0
bc_final_arb_spread_loss: float = 0.0
```

`_log_episode` writes them as optional JSONL fields on
the FIRST post-BC episode (leave blank on subsequent
eps).

### 5. New genes

In the HP schema module:

```python
{"name": "bc_pretrain_steps", "type": "int",
 "min": 0, "max": 2000, "default": 0},
{"name": "bc_learning_rate", "type": "float",
 "min": 1e-5, "max": 1e-3, "default": 3e-4},
{"name": "bc_target_entropy_warmup_eps", "type": "int",
 "min": 0, "max": 20, "default": 5},
```

Whitelist in `_REWARD_OVERRIDE_KEYS`? No — these aren't
reward-shape knobs. Make sure they flow via the agent-hp
dict without being dropped. Confirm by tracing the
`agent_hp` dict construction in `training/worker.py`.

### 6. Tests — `tests/arb_curriculum/test_bc_pretrainer.py`

Per §30:

1. **Per-agent independence.** Two agents, same genes,
   different seeds → after BC, parameter tensors differ.
   (The hard-constraint §16 regression guard.)
2. **Only signal + arb_spread heads change.** Snapshot
   other heads; assert bit-identical post-BC.
3. **Loss decreases on synthetic samples.** 100
   consistent synthetic samples → 20 BC steps → loss
   drops by a large margin.
4. **Empty oracle → skip cleanly.** `bc_pretrain(policy,
   [], 100)` returns empty history, no parameter
   changes, no errors.
5. **Gene-zero skip.** `bc_pretrain_steps=0` in the
   worker path → `BCPretrainer.pretrain` is never
   called.
6. **All three architectures.** Parameterised test that
   BC runs on `ppo_lstm_v1`, `ppo_time_lstm_v1`,
   `ppo_transformer_v1` without crash.
7. **Schema mismatch hard-fails.** Oracle cache with
   wrong `obs_schema_version` header → `load_samples`
   raises; BC path catches and skips with warning.
8. **Integration test: post-BC update path.** Run BC on
   a tiny synthetic oracle; call `trainer._ppo_update`
   once; spy on `_update_reward_baseline` and assert the
   argument is per-step mean, not episode sum (per the
   2026-04-18 lesson). **This is the load-bearing
   regression guard.**
9. **Controller handshake.** Set post-BC entropy to 40
   and target to 150 with warmup 5 eps; trace
   `_effective_target_entropy` across eps 0..6 and
   assert the interpolation (40, 62, 84, 106, 128, 150,
   150).

### 7. CLAUDE.md

New paragraph under "Entropy control — target-entropy
controller":

```
### BC-pretrain warmup handshake (2026-04-19)

When an agent's ``bc_pretrain_steps > 0``, behavioural
cloning runs before the first PPO rollout. Post-BC, the
policy's forward-pass entropy is typically LOW (confident
on oracle targets) while the controller's standing target
is 150. Without intervention, the controller would boost
``alpha`` aggressively on the first PPO update and undo
BC.

The handshake: after BC completes, the agent's effective
target entropy anneals linearly from the post-BC measured
entropy up to 150 over
``bc_target_entropy_warmup_eps`` episodes (gene, default
5). The stored ``self._target_entropy`` is unchanged; only
the value read BY the controller step is transformed.
Once the warmup episodes are done, the effective target
equals the configured target and normal controller
behaviour resumes.

Default ``bc_target_entropy_warmup_eps = 5`` is a first
cut; tune via the gene. ``0`` disables the warmup and
restores pre-BC controller behaviour for that agent —
useful for ablation.
```

Also a new paragraph under "Reward function: raw vs
shaped":

```
### BC pretrain (2026-04-19)

Per-agent behavioural cloning on arb oracle samples
(plans/arb-curriculum/session_prompts/01_oracle_scan.md)
runs before PPO when ``bc_pretrain_steps > 0``. Only
``signal`` and ``arb_spread`` heads are trained — stake,
aggression, cancel, requote_signal, close_signal heads
have their parameters frozen and restored to
require_grad=True only after BC completes. BC uses its own
Adam optimiser; PPO's optimiser state is untouched so LR
warmup and reward-centering still apply as designed.

Per-agent, never shared. Sharing BC-pretrained weights
across a population collapses GA diversity irreparably
(inherited lesson from plans/arb-improvements/
lessons_learnt.md).
```

### 8. Full-suite check (NO active training)

```
pytest tests/arb_curriculum/ -x
pytest tests/ -q --timeout=120
```

### 9. Commit

```
feat(training): per-agent BC pretrain on arb oracle + target-entropy warmup handshake

Add behavioural cloning as a pre-PPO warmstart phase for
agents with bc_pretrain_steps > 0. Per-agent (never
shared across the population), trains only the signal and
arb_spread heads, uses a separate Adam optimiser so PPO's
optimiser state stays fresh. Gracefully skips on empty
oracle caches with a warning.

New genes:
- bc_pretrain_steps (int, [0, 2000], default 0)
- bc_learning_rate (float, [1e-5, 1e-3], default 3e-4)
- bc_target_entropy_warmup_eps (int, [0, 20], default 5)

Why: 2026-04-19 reward-densification diagnosis. Random
arbing is expected-negative, so policy gradient descends
to "arb less" before "arb better". BC starts the policy
in a region where the gradient points the right way.

Controller handshake: post-BC, policy entropy is LOW
(confident on oracle). Target-entropy controller at target
150 would boost alpha aggressively on first update and
undo BC. Handshake: effective target entropy anneals
linearly from post-BC measured up to 150 over
bc_target_entropy_warmup_eps episodes.

Per-agent invariant: test_bc_per_agent_independence.py
asserts two same-gene seeds diverge after BC. Inherited
hard-constraint from the never-shipped
plans/arb-improvements/ BC design (their lessons_learnt.md
entry on shared-weights footgun).

Integration regression guard: test_bc_post_update_units
spies on _update_reward_baseline during the first PPO
update post-BC, asserting per-step mean argument per the
2026-04-18 units-mismatch lesson. Unit tests alone don't
cover the optimiser-handoff path.

Changes:
- New agents/bc_pretrainer.py.
- training/worker.py integrates BC before first PPO
  rollout; progress event "phase: bc_warmup".
- agents/ppo_trainer.py gains post-BC state + effective
  target entropy computation routing the controller
  through the warmup anneal.
- EpisodeStats + JSONL row gain bc_pretrain_steps,
  bc_final_signal_loss, bc_final_arb_spread_loss.

Tests: 9 in tests/arb_curriculum/test_bc_pretrainer.py.

Not changed: matcher, obs/action schemas, PPO stability
defences, matured-arb bonus, naked-loss annealing (all
orthogonal).

Per plans/arb-curriculum/hard_constraints.md s16-s20,
s30.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT share BC weights across agents. Test #1 enforces.
- Do NOT let BC touch heads other than signal and
  arb_spread. Test #2 enforces.
- Do NOT bleed BC's Adam state into PPO's Adam state.
  Separate optimiser instance; test #8 guards via the
  units-mismatch regression.
- Do NOT load an oracle cache whose schema versions
  don't match. Silent coercion here corrupts BC targets
  invisibly. Hard-fail.
- Do NOT run the full pytest suite during active training.

## After Session 04

1. Append a progress entry with BC loss stats on a
   representative run.
2. Hand back for Session 05 (curriculum day ordering).
