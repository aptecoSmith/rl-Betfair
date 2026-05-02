# Session prompt — phase-5-restore-genes Session 01: schema + breeding

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

The v2 cohort GA evolves only 7 genes. Across the rewrite's
plans, **11 additional knobs were explicitly designed as
per-agent genes** but never landed in `CohortGenes`. This
session promotes all 11 to fields on `CohortGenes`, with
documented ranges and an `enabled_set: frozenset[str]`
parameter on the breeding helpers so the operator can decide
per-cohort which genes evolve and which stay frozen at default.

The 11 new genes:

| Gene | Range | Distribution | Default-when-disabled |
|---|---|---|---|
| `open_cost` | `[0.0, 2.0]` | uniform | `0.0` |
| `matured_arb_bonus_weight` | `[0.0, 5.0]` | uniform | `0.0` |
| `mark_to_market_weight` | `[0.0, 0.10]` | uniform | `0.05` |
| `naked_loss_scale` | `[0.0, 1.0]` | uniform | `1.0` |
| `stop_loss_pnl_threshold` | `[0.0, 0.30]` | uniform | `0.0` |
| `arb_spread_scale` | `[0.5, 2.0]` | uniform | `1.0` |
| `fill_prob_loss_weight` | `[0.0, 0.30]` | uniform | `0.0` |
| `mature_prob_loss_weight` | `[0.0, 0.30]` | uniform | `0.0` |
| `risk_loss_weight` | `[0.0, 0.30]` | uniform | `0.0` |
| `alpha_lr` | `[1e-2, 1e-1]` | log-uniform | `1e-2` |
| `reward_clip` | `[1.0, 10.0]` | uniform | `10.0` |

Distributions are uniform unless the gene's value spans
multiple orders of magnitude (`alpha_lr`), in which case
log-uniform.

Default-when-disabled values match each plan's pre-plan
cohort-wide value so a launch without any `--enable-gene` flags
is byte-identical to a pre-Phase-5 run at the same seed.

The 7 existing genes (`learning_rate`, `entropy_coeff`,
`clip_range`, `gae_lambda`, `value_coeff`, `mini_batch_size`,
`hidden_size`) keep their existing ranges and ALWAYS evolve —
they don't take an `--enable-gene` flag because they're
unconditionally on (legacy contract).

End-of-session bar:

1. **`CohortGenes` has 18 fields.** All 11 new gene fields are
   present, with type annotations matching the existing pattern
   (`float` for continuous, `int` for discrete-categoricals; no
   discrete-categorical genes in this batch but mirror the
   convention).
2. **Range constants defined** for each new gene, named
   `<UPPER_NAME>_RANGE` (e.g. `OPEN_COST_RANGE = (0.0, 2.0)`).
3. **`_sample_field` extended** to dispatch on the new gene
   names. Uniform genes use `_sample_uniform`; log-uniform
   genes use `_sample_log_uniform`.
4. **`sample_genes(rng, enabled_set)` extended** to take an
   `enabled_set: frozenset[str]` parameter. Disabled genes get
   their default value; enabled genes are sampled as usual.
   The 7 legacy genes are always sampled (they don't appear in
   `enabled_set` semantics — they're implicit).
5. **`mutate(genes, rng, mutation_rate, enabled_set)` extended**
   to skip disabled genes. A disabled gene's value never
   changes during mutation.
6. **`crossover(parent_a, parent_b, rng, enabled_set)`
   extended** to skip disabled genes. A disabled gene in the
   child takes the default value (NOT the parent's value),
   keeping the cohort-wide default invariant.
7. **`assert_in_range(genes)` extended** to validate the new
   ranges.
8. **`to_dict(genes)` extended** to serialise all 18 fields.
9. **All existing `tests/test_v2_cohort_genes.py` tests pass**
   without modification (default `enabled_set =
   frozenset()` means new genes stay at default → existing
   behaviour byte-identical).
10. **New tests pass** covering: per-gene range validation,
    disabled-gene defaults, `enabled_set` respected by sample /
    mutate / crossover.

## What you need to read first

1. `plans/rewrite/phase-5-restore-genes/purpose.md` — this
   plan's scope and constraints.
2. `training_v2/cohort/genes.py` — the file you're modifying.
   Read all of it (it's small).
3. `tests/test_v2_cohort_genes.py` — the existing test file.
   Read all tests so the new ones match the convention.
4. The 11 plans that introduced these knobs — at least the
   relevant gene-promotion lines:
   - `plans/selective-open-shaping/hard_constraints.md` line ~110
   - `plans/selective-open-shaping/master_todo.md` line ~185
   - `plans/scalping-active-management/` (any session prompt
     for the aux-head loss weights — they're documented as genes)
   - `plans/per-runner-credit/findings.md` (mature_prob mention)
   - `plans/arb-signal-cleanup/` (alpha_lr promotion mention)

## What to do

### 1. Pre-flight (~15 min)

- Confirm the gene/range table above against
  `_REWARD_OVERRIDE_KEYS` in `env/betfair_env.py`. Each new
  gene name MUST already be a whitelisted reward_overrides key
  (or `arb_spread_scale` which lives in `scalping_overrides`,
  see Session 02 for that plumbing).
- Run `pytest tests/test_v2_cohort_genes.py
  tests/test_v2_cohort_runner.py
  tests/test_v2_cohort_worker.py -v` and confirm all pass on
  the current code. This is your "known-good" reference.
- Read CLAUDE.md for any §"... as per-agent gene" notes — the
  rewrite has accumulated a few of these. Make sure your gene
  ranges match the documented intent.

### 2. Range constants + sampling (~20 min)

In `training_v2/cohort/genes.py`, add range constants near the
existing ones:

```python
# Phase 5 — restore-genes. New per-agent genes promoted from
# cohort-wide reward_overrides flags. Each plan that introduced
# the knob documented "should be a gene"; this phase delivers
# on those. See plans/rewrite/phase-5-restore-genes/purpose.md.

OPEN_COST_RANGE = (0.0, 2.0)
MATURED_ARB_BONUS_WEIGHT_RANGE = (0.0, 5.0)
MARK_TO_MARKET_WEIGHT_RANGE = (0.0, 0.10)
NAKED_LOSS_SCALE_RANGE = (0.0, 1.0)
STOP_LOSS_PNL_THRESHOLD_RANGE = (0.0, 0.30)
ARB_SPREAD_SCALE_RANGE = (0.5, 2.0)
FILL_PROB_LOSS_WEIGHT_RANGE = (0.0, 0.30)
MATURE_PROB_LOSS_WEIGHT_RANGE = (0.0, 0.30)
RISK_LOSS_WEIGHT_RANGE = (0.0, 0.30)
ALPHA_LR_RANGE = (1e-2, 1e-1)
REWARD_CLIP_RANGE = (1.0, 10.0)

# Default-when-disabled values. Match each plan's pre-plan
# cohort-wide default so a launch without any --enable-gene
# flags is byte-identical to a pre-Phase-5 run.
PHASE5_GENE_DEFAULTS: dict[str, float] = {
    "open_cost": 0.0,
    "matured_arb_bonus_weight": 0.0,
    "mark_to_market_weight": 0.05,
    "naked_loss_scale": 1.0,
    "stop_loss_pnl_threshold": 0.0,
    "arb_spread_scale": 1.0,
    "fill_prob_loss_weight": 0.0,
    "mature_prob_loss_weight": 0.0,
    "risk_loss_weight": 0.0,
    "alpha_lr": 1e-2,
    "reward_clip": 10.0,
}

PHASE5_GENE_NAMES: frozenset[str] = frozenset(PHASE5_GENE_DEFAULTS)
```

Extend `_sample_field` with the new gene names, dispatching to
`_sample_log_uniform` for `alpha_lr` and `_sample_uniform` for
the rest.

### 3. CohortGenes extension (~20 min)

Add the 11 fields to the dataclass, with default values
matching `PHASE5_GENE_DEFAULTS` (so a `CohortGenes()` call with
only the 7 legacy fields specified still works for any code
constructing one directly):

```python
@dataclass(frozen=True)
class CohortGenes:
    # Legacy 7 (Phase 3).
    learning_rate: float
    entropy_coeff: float
    clip_range: float
    gae_lambda: float
    value_coeff: float
    mini_batch_size: int
    hidden_size: int

    # Phase 5 — restore-genes (2026-05-03). Default values match
    # cohort-wide pre-plan defaults so unused genes stay neutral.
    open_cost: float = 0.0
    matured_arb_bonus_weight: float = 0.0
    mark_to_market_weight: float = 0.05
    naked_loss_scale: float = 1.0
    stop_loss_pnl_threshold: float = 0.0
    arb_spread_scale: float = 1.0
    fill_prob_loss_weight: float = 0.0
    mature_prob_loss_weight: float = 0.0
    risk_loss_weight: float = 0.0
    alpha_lr: float = 1e-2
    reward_clip: float = 10.0
```

Extend `to_dict` to serialise all 18 fields.

Extend `assert_in_range` with range checks for each new gene.

### 4. Sample / mutate / crossover with enabled_set (~30 min)

Update `sample_genes`, `mutate`, `crossover` signatures to
accept `enabled_set: frozenset[str] = frozenset()`. The empty
default is the legacy behaviour: no Phase 5 genes evolve;
they all stay at default.

```python
def sample_genes(
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Sample one fresh agent's genes from the locked schema.

    The 7 legacy genes (PPO + architecture) ALWAYS evolve.
    Phase 5 genes evolve only when their name is in
    ``enabled_set``; otherwise they take the cohort-wide
    default from ``PHASE5_GENE_DEFAULTS``.
    """
    kwargs: dict = {}
    for f in fields(CohortGenes):
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            kwargs[f.name] = PHASE5_GENE_DEFAULTS[f.name]
        else:
            kwargs[f.name] = _sample_field(rng, f.name)
    return CohortGenes(**kwargs)


def mutate(
    genes: CohortGenes,
    rng: random.Random,
    mutation_rate: float = 0.1,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Per-gene mutation respecting ``enabled_set``."""
    kwargs: dict = {}
    for f in fields(CohortGenes):
        current = getattr(genes, f.name)
        # Phase 5 genes that are disabled stay at default —
        # mutation can't move them.
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            kwargs[f.name] = PHASE5_GENE_DEFAULTS[f.name]
            continue
        if rng.random() < mutation_rate:
            kwargs[f.name] = _sample_field(rng, f.name)
        else:
            kwargs[f.name] = current
    return CohortGenes(**kwargs)


def crossover(
    parent_a: CohortGenes,
    parent_b: CohortGenes,
    rng: random.Random,
    enabled_set: frozenset[str] = frozenset(),
) -> CohortGenes:
    """Uniform per-gene crossover, respecting ``enabled_set``."""
    kwargs: dict = {}
    for f in fields(CohortGenes):
        # Disabled Phase 5 genes always take the default —
        # never inherit a parent's value.
        if f.name in PHASE5_GENE_NAMES and f.name not in enabled_set:
            kwargs[f.name] = PHASE5_GENE_DEFAULTS[f.name]
            continue
        if rng.random() < 0.5:
            kwargs[f.name] = getattr(parent_a, f.name)
        else:
            kwargs[f.name] = getattr(parent_b, f.name)
    return CohortGenes(**kwargs)
```

### 5. Tests (~30 min)

In `tests/test_v2_cohort_genes.py`, add a `TestPhase5Genes`
class:

```python
class TestPhase5Genes:
    """Phase 5 — restore-genes. Per-agent gene promotion with
    operator-controlled enable/disable per cohort."""

    def test_legacy_sampling_byte_identical(self):
        """sample_genes() with empty enabled_set produces the
        same 7-gene-evolved + Phase-5-defaults shape every time."""
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        genes_a = sample_genes(rng_a)
        genes_b = sample_genes(rng_b)
        assert genes_a == genes_b
        # Disabled phase-5 genes are at default.
        for name, default in PHASE5_GENE_DEFAULTS.items():
            assert getattr(genes_a, name) == default

    def test_enabled_gene_is_sampled(self):
        """A gene in enabled_set produces varied values
        across seeds."""
        values = set()
        for seed in range(10):
            rng = random.Random(seed)
            genes = sample_genes(rng, enabled_set=frozenset({"open_cost"}))
            values.add(genes.open_cost)
        # Not all the same value (uniform sampling on [0, 2]).
        assert len(values) > 5

    def test_disabled_gene_stays_at_default_during_mutation(self):
        """mutate() with mutation_rate=1.0 still leaves disabled
        genes at default."""
        rng = random.Random(42)
        genes = sample_genes(rng)  # all disabled → all at default
        rng2 = random.Random(99)
        # Mutate with rate 1.0 (re-sample everything we're allowed to).
        mutated = mutate(genes, rng2, mutation_rate=1.0)
        # Phase 5 genes still at default.
        for name, default in PHASE5_GENE_DEFAULTS.items():
            assert getattr(mutated, name) == default

    def test_enabled_gene_mutates(self):
        """mutate() with mutation_rate=1.0 re-samples enabled
        Phase 5 genes."""
        rng = random.Random(42)
        genes = sample_genes(rng, enabled_set=frozenset({"open_cost"}))
        rng2 = random.Random(99)
        mutated = mutate(
            genes, rng2, mutation_rate=1.0,
            enabled_set=frozenset({"open_cost"}),
        )
        # open_cost should differ (high-probability test).
        assert mutated.open_cost != genes.open_cost

    def test_disabled_gene_in_crossover_takes_default(self):
        """crossover() with empty enabled_set: child gets defaults
        for Phase 5 genes regardless of parents."""
        rng_a = random.Random(1)
        parent_a = CohortGenes(
            learning_rate=1e-3, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
            open_cost=1.5, mark_to_market_weight=0.08,  # non-default
        )
        parent_b = CohortGenes(
            learning_rate=2e-4, entropy_coeff=1e-2, clip_range=0.1,
            gae_lambda=0.98, value_coeff=0.7, mini_batch_size=128,
            hidden_size=256,
            open_cost=0.5, mark_to_market_weight=0.02,  # non-default
        )
        rng = random.Random(42)
        child = crossover(parent_a, parent_b, rng)  # empty enabled
        assert child.open_cost == 0.0  # default, not parent value
        assert child.mark_to_market_weight == 0.05  # default

    def test_each_gene_range_respected(self):
        """Every Phase 5 gene's sampled value lies in its
        documented range."""
        for name in PHASE5_GENE_NAMES:
            for seed in range(20):
                rng = random.Random(seed)
                genes = sample_genes(rng, enabled_set=frozenset({name}))
                value = getattr(genes, name)
                # Look up range constant by gene-name convention.
                # ... (range check per gene)
                pass

    def test_assert_in_range_validates_phase5_genes(self):
        """assert_in_range raises when a Phase 5 gene is out
        of bounds."""
        bad_genes = CohortGenes(
            learning_rate=1e-3, entropy_coeff=1e-3, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
            hidden_size=128,
            open_cost=3.0,  # > 2.0 upper bound
        )
        with pytest.raises(ValueError):
            assert_in_range(bad_genes)
```

### 6. Run regression sweep + commit

```
pytest tests/test_v2_cohort_genes.py -v
pytest tests/test_v2_cohort_runner.py
        tests/test_v2_cohort_worker.py -v
```

Both should pass. Commit as a single logical unit with a
clear message.

## Stop conditions

- **Existing test_v2_cohort_genes.py tests fail** → backwards-
  compat broken. The legacy default `enabled_set = frozenset()`
  must reproduce pre-plan behaviour byte-identically. Roll
  back and re-derive.
- **CohortGenes constructor breaks for code that passes only
  the 7 legacy fields** → defaults missing on the new fields.
  Add defaults.
- **`fields(CohortGenes)` order changes** → existing readers
  that depend on field order may break. The new fields go AT
  THE END of the dataclass; order matters.
- **Past 3 h excluding tests** → the abstractions are wrong.
  Most of this work is repetitive (each gene adds ~5 lines of
  similar code in 4-5 places). A long session means the
  enabled_set parameter is being threaded the wrong way.

## Hard constraints

Inherited from `purpose.md` §"Hard constraints" plus:

1. **No env edits.** All work in `training_v2/cohort/genes.py`
   and the test file.
2. **No removal of existing genes.** The 7 PPO/arch genes
   stay; their ranges and types unchanged.
3. **Backwards compatibility for legacy launches.**
   `sample_genes(rng)` (no `enabled_set`) MUST return a
   `CohortGenes` whose Phase 5 fields are all at default.
4. **Schema is forward-only.** `CohortGenes` adds 11 fields
   at the end; never removes or renames.

## Out of scope

- CLI plumbing (Session 02 owns this).
- Worker plumbing — converting gene values to reward_overrides
  (Session 02 owns this).
- Validation cohort (Session 03 owns this).
- Mutual-exclusion guard between `--reward-overrides` and
  `--enable-gene` (Session 02 owns this).
