---
plan: rewrite/phase-5-restore-genes
opened: 2026-05-03
status: GREEN
session_03_completed: 2026-05-03
---

# Phase 5 — restore-genes: cumulative findings

## What this phase delivered

`CohortGenes` grew **7 → 18 fields**. Eleven new per-agent genes
were promoted from cohort-wide `reward_overrides` flags, with
documented ranges and an `--enable-gene NAME` CLI switch so the
operator decides per cohort which genes evolve. Disabled genes
stay frozen at their pre-Phase-5 cohort-wide default (preserving
byte-identity for legacy launches at the same seed).

## Schema-extension table

| Gene | Range | Distribution | Default-when-disabled | Routes via | Plan that called it a gene |
|---|---|---|---|---|---|
| `open_cost` | `[0.0, 2.0]` | uniform | `0.0` | reward_overrides | `selective-open-shaping` |
| `matured_arb_bonus_weight` | `[0.0, 5.0]` | uniform | `0.0` | reward_overrides | `arb-curriculum` (designed-as-gene) |
| `mark_to_market_weight` | `[0.0, 0.10]` | uniform | `0.05` | reward_overrides | `reward-densification` (`-gene` follow-on) |
| `naked_loss_scale` | `[0.0, 1.0]` | uniform | `1.0` | reward_overrides | `arb-curriculum` (designed-as-gene) |
| `stop_loss_pnl_threshold` | `[0.0, 0.30]` | uniform | `0.0` | reward_overrides | `force-close-architecture` Session 02 |
| `arb_spread_scale` | `[0.5, 2.0]` | uniform | `1.0` | scalping_overrides | scalping-mechanics gene |
| `fill_prob_loss_weight` | `[0.0, 0.30]` | uniform | `0.0` | reward_overrides | `scalping-active-management` Session 02 |
| `mature_prob_loss_weight` | `[0.0, 0.30]` | uniform | `0.0` | reward_overrides | `per-runner-credit` follow-on |
| `risk_loss_weight` | `[0.0, 0.30]` | uniform | `0.0` | reward_overrides | `scalping-active-management` Session 03 |
| `alpha_lr` | `[1e-2, 1e-1]` | log-uniform | `1e-2` | (gene-only, see below) | `arb-signal-cleanup` |
| `reward_clip` | `[1.0, 10.0]` | uniform | `10.0` | reward_overrides | (designed-as-gene) |

The 7 legacy genes (`learning_rate`, `entropy_coeff`, `clip_range`,
`gae_lambda`, `value_coeff`, `mini_batch_size`, `hidden_size`) keep
their existing ranges and ALWAYS evolve — they don't take an
`--enable-gene` flag because they're unconditionally on (legacy
contract).

## Trainer-hyperparameter gap (alpha_lr, reward_clip)

Two of the 11 promoted genes were originally designed to override
PPO trainer hyperparameters at construction time (`alpha_lr` →
SGD learning rate on `log_alpha`; `reward_clip` → per-step reward
clip applied at the training-signal layer). The v2 trainer
(`training_v2.discrete_ppo.trainer.DiscretePPOTrainer`) does NOT
yet accept those two knobs in its constructor — they're hardcoded
inside the trainer (and `reward_clip` is whitelisted in
`BetfairEnv._REWARD_OVERRIDE_KEYS` purely so the gene
passthrough path doesn't trip the unknown-key debug log).

Phase 5 ships the gene **schema** for both, the **CLI switch**, and
the **per-agent draws** appearing in `CohortGenes.to_dict()` /
scoreboard `hyperparameters`. What it does NOT ship is the trainer
constructor wiring — that's a small follow-on plan
(`v2-trainer-genes-pickup`) gated on validating the v2 trainer's
alpha-controller and reward-clip code paths against v1's
implementation. When it lands, the worker's `train_one_agent` /
`train_cluster_batched` need a one-line trainer-construction
override:

```python
trainer = DiscretePPOTrainer(
    ...,
    **({"alpha_lr": float(genes.alpha_lr)}
       if "alpha_lr" in enabled_set else {}),
    **({"reward_clip": float(genes.reward_clip)}
       if "reward_clip" in enabled_set else {}),
)
```

The Phase 5 scope is scoped to the schema + plumbing — not the
trainer-side support. Per Session 02's stop conditions, this is
documented and explicitly out of scope for this plan.

## Validation cohort

A small validation cohort
(`registry/v2_phase5_validation_1777803407`) launched with
**all 11 Phase 5 genes enabled** (`--enable-gene` × 11),
3 agents × 1 generation, training day `2026-04-26`, eval day
`2026-04-28`, `--data-dir data/processed_amber_v2_window
--device cpu --seed 42`. Wall: **430.7 s** (~7 min).

| Agent | open_cost | mtm | stop_loss | arb_scale | alpha_lr | pairs_opened | arbs_completed | arbs_stop_closed | bets | eval pnl |
|---|---|---|---|---|---|---|---|---|---|---|
| b27a9711 | 1.784 | 0.0422 | 0.066 | 1.258 | 0.0351 | 228 | 37 | 31 | 306 | -278.3 |
| dd1998e7 | 0.311 | 0.0337 | 0.029 | 1.771 | 0.0344 | 151 | 29 | 36 | 221 | -17.0 |
| 0758b211 | 1.409 | 0.0228 | 0.024 | 0.849 | 0.0232 | 140 | 31 | 39 | 219 | -365.5 |

Gene values vary sharply across agents (uniform / log-uniform
sampling produces a meaningful spread), and per-agent env
behaviour follows the gene draws (`pairs_opened` 140 → 228,
`arbs_stop_closed` 31 → 39). All 3 agents completed without
plumbing crashes; per-agent `reward_overrides` /
`scalping_overrides` reach the env correctly. The eval P&Ls are
weak (none positive) — expected for a wiring-validation run on
1 training day with random seeds and no breeding pass.

The unit-test layer (40 tests across `test_v2_cohort_genes.py`,
`test_v2_cohort_runner.py`, `test_v2_cohort_worker.py`) is the
load-bearing correctness layer for this plan:

- `TestPhase5Genes` (11 tests): per-gene range validation,
  disabled-gene defaults, `enabled_set` respected by sample /
  mutate / crossover, byte-identity for legacy launches, all 18
  fields serialised in `to_dict()`.
- `TestPhase5EnableGeneCli` (6 tests): CLI parser dedupes /
  validates / errors-on-unknown-name, mutual-exclusion guard
  triggers for `--reward-overrides` + `--enable-gene` collision,
  legacy launch passes no Phase 5 keys, `enabled_set` reaches the
  worker function, enabled gene values vary across the cohort.
- `TestPerAgentOverrideHelpers` (6 tests):
  `_build_per_agent_reward_overrides` /
  `_build_per_agent_scalping_overrides` translate enabled_set +
  gene draws into the dicts the env consumes.

See `cohort.log` in the output dir for the full operator log.

## CLI usage

### Legacy (no Phase 5 genes evolve — byte-identical to pre-plan)

```
python -m training_v2.cohort.runner --n-agents 12 ...
```

### Incremental (one new gene per cohort)

```
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 4 ... \
    --enable-gene open_cost
```

### Everything on (full 18-gene exploration)

```
python -m training_v2.cohort.runner \
    --n-agents 20 --generations 6 ... \
    --enable-gene open_cost \
    --enable-gene matured_arb_bonus_weight \
    --enable-gene mark_to_market_weight \
    --enable-gene naked_loss_scale \
    --enable-gene stop_loss_pnl_threshold \
    --enable-gene arb_spread_scale \
    --enable-gene fill_prob_loss_weight \
    --enable-gene mature_prob_loss_weight \
    --enable-gene risk_loss_weight \
    --enable-gene alpha_lr \
    --enable-gene reward_clip
```

### Mutual exclusion

```
python -m training_v2.cohort.runner ... \
    --enable-gene open_cost \
    --reward-overrides open_cost=1.0
```

errors at startup with the message *"Cannot combine --enable-gene
with --reward-overrides for the same gene name(s): ['open_cost'].
Operator must pick one source of truth per knob per run."*

## Notes on dimension explosion

The GA's search space grew **7 → 18 dimensions**. Existing 12-agent
cohort runs are likely insufficient to explore the full enabled
space:

- 12 agents per generation × 18 dims = ~0.7 agents/dim per
  generation.
- Recommended for full 11-gene exploration: 20+ agents per
  generation, 6+ generations.
- For incremental (1–2 new genes enabled): 12 agents × 4
  generations is fine.

Per the plan's hard constraints, `--enable-gene` is opt-in. The
operator can stage the dimension growth: enable `open_cost` for
one cohort, add `matured_arb_bonus_weight` for the next, and so
on, attributing eval-PnL deltas to the newly-evolved gene rather
than diluting the search across the full 11-dim Phase 5 space at
once.

## What's now possible (next plans)

The three follow-up directions in
`force-close-architecture/findings.md` are now testable as
gene-evolved cohorts:

1. **`open_cost` evolution** — multi-gen cohort with
   `--enable-gene open_cost` to find the working point that the
   uniform cohort-wide `1.0` couldn't.
2. **`matured_arb_bonus_weight` as positive shaping
   complement** — enables the matured-bonus signal per-agent so
   the GA can find the win-with-the-grain mix.
3. **Multi-gen + everything-on** — full 11-gene exploration for
   the v1-catch-up retest.

A small follow-on plan (`v2-trainer-genes-pickup`) covers the
two trainer-only genes (`alpha_lr`, `reward_clip`).

## Verdict

**GREEN** — all 11 genes wired into the schema, `--enable-gene`
CLI switch live, mutual-exclusion guard active, byte-identity
preserved for legacy launches at the same seed. Trainer-only
override path for `alpha_lr` and `reward_clip` deferred to a
small follow-on plan; documented in
`training_v2/cohort/worker.py::_PHASE5_GENES_VIA_REWARD_OVERRIDES`.
The rewrite's per-agent search space is now feature-complete vs
the design intent for the env-consumed and trainer-aux genes.
