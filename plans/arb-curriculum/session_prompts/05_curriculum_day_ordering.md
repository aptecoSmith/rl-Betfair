# Session 05 prompt — Curriculum day ordering driven by oracle density

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — "Curriculum day
  ordering" subsection of the design sketch, and the
  operator's day-4 observation (3 arbs across the whole
  day — structurally curriculum-hostile).
- [`../hard_constraints.md`](../hard_constraints.md). §21
  (opt-in modes), §22 (every day seen once per epoch),
  §23 (missing-cache fallback), §31 (tests).
- [`../master_todo.md`](../master_todo.md) — Session 05
  deliverables.
- `training/arb_oracle.py` (Session 01) — density header
  format.
- `training/worker.py` — the per-agent day-iteration path.

## Why this is necessary

The operator observed one training day had only 3 arbs.
Under random order, an agent might hit this day early
with a fresh BC-pretrained policy — and learn "the
oracle was lying, there are no arbs, stop arbing." By
density-ordering the curriculum, arb-rich days come first
(let BC's confidence match the data), then arb-sparse
days land after the policy is robust.

Critically: **every day must still be seen exactly once
per epoch**. Curriculum changes order, not membership.

## Locate the code

```
grep -n "training_dates\|for.*date.*races\|def run_agent\|iter_days" training/worker.py
grep -n "def list_days\|def all_days" data/episode_builder.py
grep -n "def scan_day" training/arb_oracle.py
grep -n "curriculum" training/worker.py agents/ppo_trainer.py config.yaml
```

Confirm before editing:

1. The per-agent loop has a clear `for day in
   training_dates:` that we can transform.
2. Oracle density lives in the `.npz` header (Session 01
   writes it) — access without loading the full sample
   list.
3. `training_dates` is constructed before the inner loop
   — we reorder it there.

## What to do

### 1. Density loader

Add to `training/arb_oracle.py`:

```python
def density_for_date(date: str, data_dir: Path) -> float:
    """Return cached arb density for a date.

    Reads only the header.json -- does not load the full
    .npz. Missing cache -> returns 0.0 (the Session 05
    fallback convention from hard_constraints.md s23).
    """
    p = data_dir / date / "header.json"
    if not p.exists():
        return 0.0
    try:
        data = json.loads(p.read_text())
        return float(data.get("density", 0.0))
    except Exception:
        return 0.0
```

### 2. Ordering helper

New module or in `training/worker.py`:

```python
def order_days_by_density(
    dates: list[str],
    mode: str,                    # "random" | "density_desc" | "density_asc"
    data_dir: Path,
    rng: random.Random,
) -> list[str]:
    """Return dates reordered per mode. Membership
    preserved exactly (s22).

    - random:         rng.sample
    - density_desc:   densest first; missing-cache = 0 -> end
    - density_asc:    sparsest first; missing-cache = 0 -> start
                      (with a warning logged)
    """
    if mode == "random":
        return rng.sample(dates, len(dates))
    densities = {d: density_for_date(d, data_dir) for d in dates}
    missing = [d for d, v in densities.items() if v == 0.0]
    if missing:
        logger.warning(
            "Curriculum mode=%s: %d dates have 0 density "
            "(cache missing or empty). Will be placed at "
            "the %s.", mode, len(missing),
            "end" if mode == "density_desc" else "start",
        )
    reverse = (mode == "density_desc")
    return sorted(dates, key=lambda d: densities[d], reverse=reverse)
```

### 3. Wire into the worker

```python
curriculum_mode = config.get("training", {}).get(
    "curriculum_day_order", "random",
)
training_dates = order_days_by_density(
    raw_training_dates, curriculum_mode,
    Path("data/oracle_cache"),
    rng=random.Random(agent_seed),
)
# Log the final ordering for reproducibility.
logger.info("Curriculum mode=%s day order: %s",
            curriculum_mode, training_dates[:5])
```

Record the active mode on every JSONL row:

```python
# EpisodeStats:
curriculum_day_order: str = "random"
# Populate:
curriculum_day_order=config["training"].get(
    "curriculum_day_order", "random"),
```

### 4. config.yaml + plan JSON

```yaml
training:
  ...
  # Arb-curriculum Session 05 (2026-04-19). Per-agent
  # day ordering derived from oracle density. Every day
  # is still seen exactly once per epoch; only the order
  # changes. Default random = pre-change behaviour.
  curriculum_day_order: random
```

Plan JSON schema: allow `training.curriculum_day_order`
override per-plan.

### 5. Tests — `tests/arb_curriculum/test_curriculum_ordering.py`

Per §31:

1. **Random mode reproduces pre-change.** Given three
   dates, the randomly-seeded ordering matches the
   pre-existing `rng.sample` behaviour.
2. **Density_desc sorts descending.** Three dates with
   known densities 0.01 / 0.005 / 0.001 → sorted
   0.01, 0.005, 0.001.
3. **Density_asc sorts ascending.** Same data → reversed.
4. **Missing cache → density 0, placed at end for desc.**
   Two dates, one missing. Desc mode places the known-
   density date first, missing at end. Warning captured
   via `caplog`.
5. **Membership preserved.** Any mode → the returned
   list is a permutation of the input.
6. **Config round-trip.** `training.curriculum_day_order
   = "density_desc"` flows through config load → worker
   init → episode info → JSONL.
7. **Invalid mode defaults to random.** Misconfigured
   mode (e.g. `"best"`) logs error and falls back to
   random rather than crashing the worker.

### 6. CLAUDE.md

Under the training / runtime section:

```
### Curriculum day ordering (2026-04-19)

Per-agent training-day order is driven by arb-oracle
density when ``training.curriculum_day_order`` is set to
``density_desc`` or ``density_asc``. Default ``random``
preserves pre-change behaviour (per-seed shuffle).

``density_desc``: arb-rich days first. Pairs naturally
with BC warm-start -- the post-BC policy sees days where
oracle targets match the data before encountering
curriculum-hostile sparse days.

``density_asc``: reverse. Provided for ablation only.

Every day is still seen exactly once per epoch regardless
of mode (hard_constraints.md s22). Curriculum changes
order, not membership.

Missing oracle cache for a date is treated as density
zero (placed at the end/start per mode). Worker logs a
warning so the operator knows to re-run the oracle scan.
```

### 7. Full-suite check (NO active training)

```
pytest tests/arb_curriculum/ -x
pytest tests/ -q --timeout=120
```

### 8. Commit

```
feat(training): opt-in curriculum day ordering driven by oracle density

Add training.curriculum_day_order config key with three
modes: random (default, current behaviour),
density_desc (arb-rich days first), density_asc
(arb-sparse first, debugging only). Per-day density
comes from the oracle-scan header.json written in
Session 01.

Why: 2026-04-19 operator observation -- one training day
had only 3 arbs across the whole day, and under random
order a freshly BC-pretrained agent could hit that day
early and learn "the oracle was lying". Density ordering
front-loads arb-rich days so post-BC confidence is
rewarded before curriculum-hostile days arrive.

Invariants (hard_constraints.md s21-s23):
- Every day seen exactly once per epoch (order not
  membership).
- Missing cache -> density 0, placed at end/start;
  warning logged.
- Invalid mode -> fall back to random with error log.

Changes:
- training/arb_oracle.density_for_date reads header only
  (cheap).
- training/worker.py applies order_days_by_density before
  the inner loop.
- EpisodeStats + JSONL row gain curriculum_day_order.
- config.yaml documents the new key.

Tests: 7 in tests/arb_curriculum/test_curriculum_ordering.py.

Not changed: oracle scan semantics, BC pretrainer,
reward path, matcher, controller.

Per plans/arb-curriculum/hard_constraints.md s21-s23, s31.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT drop days from the curriculum. Membership must
  be preserved (§22).
- Do NOT load the full `.npz` when building density —
  use the header only. Oracle caches can be large.
- Do NOT hard-fail on missing cache; log a warning and
  fall back to density 0. The operator may run the
  gene-sweep before re-scanning the oracle.

## After Session 05

1. Append a progress entry.
2. Hand back for Session 06 (registry reset + plan redraft).
