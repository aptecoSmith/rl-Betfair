# Session prompt — phase-5-restore-genes Session 02: CLI + worker plumbing

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked from Session 01, and the constraints. Do not require
any context from prior sessions.

---

## The task

Session 01 extended `CohortGenes` with 11 new fields and made
`sample_genes` / `mutate` / `crossover` respect an
`enabled_set: frozenset[str]` parameter. **Session 02 wires
that parameter to the CLI and plumbs each agent's gene values
into the env / trainer at training time.**

End-of-session bar:

1. **`--enable-gene NAME` repeatable CLI flag** in
   `training_v2/cohort/runner.py`. Operator passes one flag
   per gene to evolve. Unknown gene names error at startup.
2. **Mutual exclusion guard.** Setting both `--reward-overrides
   X=Y` AND `--enable-gene X` errors at startup with a clear
   message. Operator picks one source of truth per knob per
   run.
3. **`enabled_set` flows through `run_cohort`** to
   `sample_genes` / `mutate` / `crossover` calls. Each agent's
   `CohortGenes` reflects the cohort-level `enabled_set`.
4. **Per-agent gene values reach the env** via a per-agent
   `reward_overrides` dict assembled by the worker. For each
   gene name in `PHASE5_GENE_NAMES` that's in `enabled_set`,
   the agent's `getattr(genes, gene_name)` becomes
   `reward_overrides[gene_name]`. The cohort-level
   `--reward-overrides` flags STILL apply to non-gene knobs
   (e.g. `force_close_before_off_seconds`, `target_pnl_pair_
   sizing_enabled`).
5. **Per-agent gene values reach the trainer** for genes that
   target trainer hyperparameters. Specifically:
   - `alpha_lr` → `PPOTrainer`'s alpha-controller learning
     rate.
   - `reward_clip` → `PPOTrainer`'s per-step reward-clip
     hyperparameter.
6. **Byte-identity for legacy launches.** A cohort launched
   without ANY `--enable-gene` flags produces results
   identical to a pre-plan run at the same seed.
7. **Tests:** CLI parsing, mutual-exclusion guard,
   gene-to-env-and-trainer flow, byte-identity.

## What you need to read first

1. `plans/rewrite/phase-5-restore-genes/purpose.md`.
2. `plans/rewrite/phase-5-restore-genes/session_prompts/
   01_gene_schema_and_breeding.md` for the gene schema +
   `enabled_set` contract.
3. `training_v2/cohort/runner.py` — CLI parser is around line
   570 onwards; `_parse_reward_overrides` is the reference
   shape for the `--enable-gene` parser.
4. `training_v2/cohort/worker.py::train_one_agent` — where the
   per-agent env is built and the trainer is constructed.
   Look for the `reward_overrides=...` argument flow; that's
   where Phase 5 gene values plug in.
5. `training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer.
   __init__` — find where `alpha_lr` and `reward_clip` are
   read (they're already plumbed via the trainer's
   hyperparameter dict; just need the worker to populate
   them from gene values).
6. `tests/test_v2_cohort_runner.py` — convention for CLI tests.

## What to do

### 1. Pre-flight (~15 min)

- Confirm Session 01's `enabled_set` parameter is present on
  `sample_genes` / `mutate` / `crossover` and Session 01's
  tests pass.
- Locate the trainer's `alpha_lr` and `reward_clip` reads.
  Confirm they're already gene-pluggable (the trainer should
  already accept these in its hyperparameter dict — they were
  designed as genes per the original plans).
- Read `_parse_reward_overrides` to understand the existing
  flag-parsing convention; mirror it for `_parse_enabled_genes`.

### 2. CLI parser + mutual-exclusion guard (~30 min)

In `training_v2/cohort/runner.py`, add a CLI flag and a parser:

```python
p.add_argument(
    "--enable-gene", action="append", default=[],
    help=(
        "Enable a Phase 5 gene to evolve per-agent (repeatable). "
        "Disabled genes use cohort-wide defaults. Cannot be "
        "combined with --reward-overrides for the same gene name. "
        "Valid names: open_cost, matured_arb_bonus_weight, "
        "mark_to_market_weight, naked_loss_scale, "
        "stop_loss_pnl_threshold, arb_spread_scale, "
        "fill_prob_loss_weight, mature_prob_loss_weight, "
        "risk_loss_weight, alpha_lr, reward_clip"
    ),
)
```

Add a parser helper:

```python
def _parse_enabled_genes(items: list[str]) -> frozenset[str]:
    """Validate and dedupe --enable-gene values."""
    from training_v2.cohort.genes import PHASE5_GENE_NAMES
    enabled = set()
    for name in items or []:
        if name not in PHASE5_GENE_NAMES:
            raise ValueError(
                f"--enable-gene: unknown gene name {name!r}. "
                f"Valid: {sorted(PHASE5_GENE_NAMES)}"
            )
        enabled.add(name)
    return frozenset(enabled)
```

Add the mutual-exclusion guard after parsing both flags:

```python
enabled_set = _parse_enabled_genes(args.enable_gene)
reward_overrides = _parse_reward_overrides(args.reward_overrides)
collision = enabled_set & set(reward_overrides)
if collision:
    raise ValueError(
        f"Cannot combine --enable-gene with --reward-overrides "
        f"for the same gene name(s): {sorted(collision)}. "
        f"Operator must pick one source of truth per knob per "
        f"run. Either evolve the gene per-agent (--enable-gene) "
        f"or fix it cohort-wide (--reward-overrides), not both."
    )
```

### 3. Plumbing through run_cohort (~20 min)

`run_cohort` currently doesn't know about `enabled_set`. Add it
as a parameter:

```python
def run_cohort(
    *,
    n_agents: int,
    n_generations: int,
    days: int,
    data_dir: Path,
    device: str,
    seed: int,
    output_dir: Path,
    mutation_rate: float = 0.1,
    train_one_agent_fn: Callable[..., AgentResult] = train_one_agent,
    event_emitter: Callable[[dict], None] | None = None,
    reward_overrides: dict | None = None,
    enabled_set: frozenset[str] = frozenset(),  # NEW
    batched: bool = False,
) -> list[AgentResult]:
    ...
```

Pass `enabled_set` to `sample_genes`, `mutate`, `crossover`
calls. The `_breed_next_generation` helper (~line 280) takes
the same parameter; thread it through.

The CLI's `main()` builds `enabled_set` from `--enable-gene`
flags and passes it.

### 4. Worker plumbing — gene → reward_overrides (~30 min)

In `training_v2/cohort/worker.py::train_one_agent`, expand the
per-agent `reward_overrides` dict to include enabled-gene values.

The current pattern (paraphrased):

```python
def train_one_agent(
    *, agent_id, genes, days_to_train, eval_day, data_dir,
    device, seed, model_store, generation,
    parent_a_id=None, parent_b_id=None,
    event_emitter=None, agent_idx, n_agents,
    reward_overrides: dict | None = None,
    ...
) -> AgentResult:
    ...
    env, shim = _build_env_for_day(
        ..., reward_overrides=reward_overrides,
    )
```

Extend with a Phase-5 gene contribution. The worker computes a
per-agent `reward_overrides` dict that's the union of:

- The cohort-level `reward_overrides` passed by the runner.
- A per-agent dict mapping each enabled-gene NAME to the
  agent's gene VALUE.

```python
def _build_per_agent_reward_overrides(
    *,
    cohort_overrides: dict | None,
    genes: CohortGenes,
    enabled_set: frozenset[str],
) -> dict | None:
    """Combine cohort-level overrides with this agent's enabled-
    gene values. Cohort-level overrides apply to ALL agents
    (e.g. force_close_before_off_seconds=60). Enabled-gene
    values are per-agent (e.g. open_cost = this agent's draw).

    Phase 5 invariant: enabled-gene names cannot collide with
    cohort-level override keys (CLI guard at runner enforces).
    """
    from training_v2.cohort.genes import PHASE5_GENE_NAMES
    out: dict = dict(cohort_overrides or {})
    for name in PHASE5_GENE_NAMES:
        if name in enabled_set:
            out[name] = float(getattr(genes, name))
    return out or None
```

Then:

```python
per_agent_overrides = _build_per_agent_reward_overrides(
    cohort_overrides=reward_overrides,
    genes=genes,
    enabled_set=enabled_set,
)
env, shim = _build_env_for_day(
    ..., reward_overrides=per_agent_overrides,
)
```

`train_one_agent` and `train_cluster_batched` both need an
`enabled_set` parameter; the runner passes it in.

### 5. Trainer hyperparameter plumbing (~20 min)

Two of the 11 new genes target the PPO trainer, not the env:
`alpha_lr` and `reward_clip`. The trainer's `__init__` already
accepts both via its hyperparameter dict (per the original
plans that designed them as genes). The worker just needs to
override them when these genes are in `enabled_set`.

In `train_one_agent`, after building the trainer, override the
two trainer hyperparameters from gene values when enabled:

```python
trainer = DiscretePPOTrainer(
    ...,
    hyperparameters={
        "learning_rate": float(genes.learning_rate),
        "entropy_coeff": float(genes.entropy_coeff),
        ...
        # Phase 5: alpha_lr and reward_clip are gene-pluggable.
        # When enabled they take the agent's gene value;
        # otherwise the trainer's default applies.
        **({"alpha_lr": float(genes.alpha_lr)}
           if "alpha_lr" in enabled_set else {}),
        **({"reward_clip": float(genes.reward_clip)}
           if "reward_clip" in enabled_set else {}),
    },
)
```

(Mirror the actual trainer constructor's argument shape.)

### 6. Tests (~30 min)

In `tests/test_v2_cohort_runner.py`, add:

```python
def test_enable_gene_flag_parses(monkeypatch, tmp_path):
    """--enable-gene open_cost --enable-gene mark_to_market_weight
    builds enabled_set = {open_cost, mark_to_market_weight}."""
    ...

def test_enable_gene_unknown_name_errors():
    """--enable-gene fake_gene errors at parse time."""
    ...

def test_reward_overrides_and_enable_gene_collision_errors():
    """--reward-overrides open_cost=1.0 --enable-gene open_cost
    errors at startup."""
    ...

def test_legacy_launch_byte_identical(tmp_path):
    """A cohort launched WITHOUT --enable-gene flags produces
    results identical at fixed seed to a pre-plan baseline.
    Stub train_one_agent_fn that records the per-agent
    reward_overrides dict; assert no Phase 5 keys appear in
    any agent's dict."""
    ...

def test_enabled_gene_value_reaches_env_via_reward_overrides(tmp_path):
    """A cohort with --enable-gene open_cost has each agent's
    per-agent reward_overrides dict containing open_cost set
    to its sampled gene value (not the default)."""
    ...
```

Add to `tests/test_v2_cohort_worker.py`:

```python
def test_per_agent_reward_overrides_combines_cohort_and_genes():
    """_build_per_agent_reward_overrides merges cohort-level
    overrides with enabled-gene values. Cohort key + gene value
    in same dict; gene values come from this agent's
    CohortGenes."""
    ...
```

### 7. Regression sweep + commit

```
pytest tests/test_v2_cohort_genes.py
        tests/test_v2_cohort_runner.py
        tests/test_v2_cohort_worker.py -v
pytest tests/ --ignore=tests/test_orchestrator.py -q
        --timeout=120
```

Commit as a single logical unit with a clear message.

## Stop conditions

- **Existing cohort tests fail with empty `--enable-gene`** →
  byte-identity broken. Roll back to Session 01-only state and
  re-derive.
- **`alpha_lr` or `reward_clip` not pluggable in the trainer**
  → those plans never landed; document and SKIP those two
  genes for this session (still wire 9 of 11). Update
  Session 03 to flag the gap.
- **Past 4 h excluding tests** → the plumbing is wrong. The
  cleanest path is: gene values flow through the existing
  `reward_overrides` passthrough; no new env code is added.

## Hard constraints

Inherited from `purpose.md` plus:

1. **No env edits.** All work in `training_v2/cohort/`.
2. **CLI is forward-only.** `--enable-gene` is a NEW flag;
   doesn't replace or rename anything existing.
3. **Mutual-exclusion guard is mandatory.** If both flags set
   the same name, error at startup. No silent precedence.
4. **Backwards compatibility for legacy launches** at fixed
   seed: byte-identical training metrics.
5. **CUDA self-parity holds** at fixed seed and fixed
   `--enable-gene` flag set.

## Out of scope

- The validation cohort (Session 03 owns it).
- New scoreboard fields. Per-agent gene values already flow
  into `hyperparameters` via the existing
  `genes.to_dict()` path. No schema migration needed.
- Frontend / UI changes.
