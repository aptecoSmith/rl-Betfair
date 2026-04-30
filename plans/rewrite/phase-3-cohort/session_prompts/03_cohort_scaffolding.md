# Session prompt — Phase 3, Session 03: GA cohort scaffolding

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Stand up the v2 cohort runner — worker pool, gene schema, breeding,
mutation, registry integration. Replaces v1's `training/run_training.py`
+ `training/worker.py` with a parallel-tree v2 implementation.

**No frontend events yet** (Session 04). **No comparison run yet**
(Session 04). This session ships a 4-agent dry-run cohort that
trains, breeds, writes scoreboard rows in the registry's existing
shape, and stops.

End-of-session bar:

1. `python -m training_v2.cohort.runner --n-agents 4 --days 7 --device cuda --generations 2` runs end-to-end.
2. The registry receives 4 weight files + 4 (or 8 with breeding) scoreboard rows in the existing schema.
3. Genes are sampled from the locked Phase 3 schema (~6–8 genes); breeding produces children that mix parent genes deterministically.

## What you need to read first

1. `plans/rewrite/phase-3-cohort/purpose.md` — cohort scope, gene
   schema rules, success bar 4.
2. `plans/rewrite/phase-3-cohort/session_prompts/01_gpu_saturation.md`
   + `02_multi_day_training.md` — the building blocks Session 03
   composes.
3. `training/run_training.py` — v1 cohort runner. **Read for the
   pattern, do not import.** Specifically:
   - `__main__` flow → arg parsing → `_run_cohort` (~ line 200–500).
   - `ThreadPoolExecutor` worker pool + `_evaluate_agent` calls (~ line 800–900).
   - Registry write pattern (`registry/model_store.py` calls).
   - GA breeding inside `_breed_next_generation` / `_mutate_genes`.
4. `training/worker.py` — v1 single-agent worker. Same "read,
   don't import" rule.
5. `registry/model_store.py` — registry shape that v2 must match
   on writes. Schema is shared between v1 and v2 during the
   comparison window.
6. `agents/genes.py` — v1 gene schema. **Phase 3 ships a much
   smaller gene set; do not copy v1 wholesale.**
7. `plans/rewrite/README.md` §"What survives" GA row — the
   architectural argument for keeping GA + 6–8 genes.

## What to do

### 1. `training_v2/cohort/genes.py` — Phase 3 gene schema (~30 min)

**Locked Phase 3 gene schema**:

| Gene | Type | Range | What it does |
|---|---|---|---|
| `learning_rate` | float (log) | `[1e-5, 1e-3]` | Adam LR. |
| `entropy_coeff` | float (log) | `[1e-4, 1e-1]` | Fixed coefficient (no controller). |
| `clip_range` | float | `[0.1, 0.3]` | PPO clip. |
| `gae_lambda` | float | `[0.9, 0.99]` | GAE λ. |
| `value_coeff` | float | `[0.25, 1.0]` | Value-loss weight. |
| `mini_batch_size` | int (categorical) | `{32, 64, 128}` | PPO mini-batch. |
| `hidden_size` | int (categorical) | `{64, 128, 256}` | LSTM hidden dim. |

**That's it.** No reward-shaping genes (rewrite hard constraint
§5). No entropy-controller genes. No BC-related genes. No
curriculum genes. No force-close genes (force_close stays at the
plan-level config; Phase 3 doesn't sweep it).

```python
@dataclass(frozen=True)
class CohortGenes:
    learning_rate: float
    entropy_coeff: float
    clip_range: float
    gae_lambda: float
    value_coeff: float
    mini_batch_size: int
    hidden_size: int

def sample_genes(rng: random.Random) -> CohortGenes: ...
def crossover(parent_a, parent_b, rng) -> CohortGenes: ...  # uniform
def mutate(genes, rng, mutation_rate=0.1) -> CohortGenes: ...
```

Crossover: uniform per-gene with 50/50 parent pick. Mutation:
log-uniform sample within range for floats, uniform-categorical
for ints, with `mutation_rate` probability per gene.

### 2. `training_v2/cohort/worker.py` — single-agent worker (~60 min)

```python
def train_one_agent(
    *,
    agent_id: str,
    genes: CohortGenes,
    days_to_train: list[str],
    eval_day: str,
    data_dir: Path,
    device: str,
    seed: int,
    output_dir: Path,
) -> AgentResult:
    """Train one agent across N days, evaluate on the held-out day,
    write the scoreboard row + weights to the registry, return the
    AgentResult."""
```

Reuses Session 02's multi-day loop verbatim (import `train.main`
or refactor the loop body into a function the worker calls). The
worker:

1. Builds the policy from `genes.hidden_size`.
2. Builds the trainer from the rest of the genes (LR, clip,
   gae_lambda, etc.).
3. Iterates over `days_to_train` (Session 02's loop).
4. Evaluates on `eval_day`: one rollout, no PPO update.
5. Writes weights to the registry via `registry/model_store.py`.
6. Returns `AgentResult` (genes, train summary, eval summary).

**Eval phase = rollout-only.** Re-use Session 01's
`RolloutCollector` with `policy.eval()` and no PPO update; capture
`day_pnl`, `total_reward`, action histogram, force-close rate.

### 3. `training_v2/cohort/runner.py` — orchestrator (~60 min)

```python
def run_cohort(
    *,
    n_agents: int,
    n_generations: int,
    days_to_train: int,
    data_dir: Path,
    device: str,
    seed: int,
    output_dir: Path,
) -> None: ...
```

Loop:

```
1. Sample `n_agents` initial genes (gen 0).
2. For gen in 0..n_generations:
   a. Train each agent in the cohort (sequential or
      ThreadPoolExecutor — Session 03 starts sequential, see
      §"Concurrency" below).
   b. Sort by eval-day reward.
   c. If gen < n_generations - 1:
      - Keep top half (elites).
      - Breed bottom half from random pairs of elites.
      - Mutate children.
3. Write final scoreboard.
```

**Concurrency for Session 03 = sequential.** Multi-agent on a
single GPU is a follow-on plan; v1's ThreadPoolExecutor lets
agents share the GPU but is fragile (OOM at high N) and not the
hill we want to die on for Phase 3 first run. Sequential 4 agents
× 7 days × 2 generations = 56 episodes on GPU = ~15-20 minutes,
acceptable for a dry-run.

If Session 03's sequential 4-agent run is < 30 min, leave
sequential in place. If it's > 1 hour and Session 04 is gated on
faster cohort runs, follow-on plan adds the worker pool.

### 4. Registry integration (~30 min)

v1's `registry/model_store.py` exposes:

- `add_model(arch_name, state_dict_path, scoreboard_row, gene_dict)`
- The `arch_name` discriminates architectures; weight-shape
  hashing further protects against load-mismatch.

v2 writes with `arch_name="v2_discrete_ppo_lstm_h{hidden_size}"`.
Different `hidden_size` → different `arch_name` → registry won't
try to cross-load. Scoreboard row schema **matches v1 exactly**
(read v1's row construction; copy field-by-field).

### 5. Dry-run cohort (~30 min)

```
python -m training_v2.cohort.runner \
    --n-agents 4 \
    --generations 2 \
    --days 7 \
    --device cuda \
    --seed 42 \
    --output-dir registry/v2_dryrun_$(date +%s)
```

Validate:

- 4 weight files in `registry/v2_dryrun_*/weights/`.
- 8 scoreboard rows in `registry/v2_dryrun_*/scoreboard.jsonl`
  (4 per generation × 2 generations).
- Genes column sane: each row's gene dict has all 7 keys.
- No exceptions, no orphan threads, no GPU memory leaks across
  generations (a `torch.cuda.empty_cache()` between generations
  is fine — match v1's `training/run_training.py:799` pattern).

### 6. Tests (~30 min)

`tests/test_v2_cohort_genes.py`:

- Sampling produces in-range genes.
- Crossover produces a child whose every gene is from one of the
  two parents.
- Mutation with rate=0 is identity; rate=1 always produces a new
  gene per slot.

`tests/test_v2_cohort_worker.py`:

- A worker runs end-to-end on a synthetic 1-day dataset with
  `genes.hidden_size=64` (cheap to instantiate).
- The returned AgentResult contains the eval-day P&L.

`tests/test_v2_cohort_runner.py` (lightweight integration):

- `run_cohort(n_agents=2, generations=1, days=1)` writes 2
  scoreboard rows, no exceptions.

## Stop conditions

- Bar 4 fails (cohort doesn't run end-to-end) → **stop**. Triage
  before Session 04. Most likely cause: a registry-shape
  mismatch (v2 writes a field v1 doesn't expect, or vice versa).
- Generation 1 → Generation 2 breeding produces invalid genes →
  **stop**. The crossover or mutation has a sign error.
- GPU OOM during gen 2 → **stop**. v1's
  `training/run_training.py:799` clears GPU cache between agents;
  copy the pattern.
- `arch_name` collision with v1 weights in the same registry →
  **stop and rename**. v1 and v2 weights coexist during the
  comparison window; the discriminator is `arch_name`, so it must
  not collide.

## Hard constraints

- **No env edits.** Same as all phases.
- **No new shaped rewards.** Even if the cohort run shows poor
  P&L, do NOT add reward shaping to "fix" it. That's the rewrite's
  bet — finding it doesn't pay is a Phase 3 finding, not a Phase 3
  patch.
- **Locked gene schema.** No additions, no removals, no range
  changes. Mid-cohort schema changes invalidate breeding.
- **No frontend events.** Session 04 wires the websocket adapter.
  Session 03 writes plain JSONL + registry only.
- **No v1 trainer / worker / runner imports.** Read for the
  pattern; re-implement in `training_v2/cohort/`.
- **Registry shape matches v1 exactly.** Field-for-field. The
  scoreboard UI consumes both v1 and v2 rows during the
  comparison window.

## Out of scope

- Frontend events (Session 04).
- 12-agent cohort (Session 04 is where 12 agents land; Session 03
  caps at 4 to keep the dry-run fast).
- Multi-GPU (follow-on).
- Concurrent worker pool (follow-on; Session 03 is sequential).
- Curriculum (follow-on).
- BC pretrain (rewrite removes it; Phase 0's scorer replaces the
  discriminative half).

## Useful pointers

- v1 cohort runner: `training/run_training.py` (read, don't
  import).
- v1 worker: `training/worker.py`.
- v1 GA breeding: `agents/genes.py` + the `_breed_next_generation`
  helper in v1's runner.
- v1 GPU cache clear: `training/run_training.py:799`.
- Registry shape: `registry/model_store.py` + a recent
  `registry/scoreboard.jsonl` for the row format.
- v1 ThreadPoolExecutor pattern: `training/run_training.py:870`
  (do NOT use in Session 03; document for Session 04 / follow-on).

## Estimate

3.5 hours.

- 30 min: gene schema.
- 60 min: worker.
- 60 min: runner.
- 30 min: registry integration.
- 30 min: dry-run cohort.
- 30 min: tests.

If past 5 hours, stop and check scope. The most likely overrun is
the registry-shape match — v1's row schema has accreted fields
over time; nail down the smallest superset v2 needs and write a
note for Session 04 listing fields v2 leaves null.
