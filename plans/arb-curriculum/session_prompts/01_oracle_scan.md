# Session 01 prompt — Offline arb oracle scan + per-day density metric

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — the diagnosis; why an
  oracle is the foundation, not an optional extra.
- [`../hard_constraints.md`](../hard_constraints.md). §6–§9
  (oracle offline-only, profitability uses
  `scalping_math.locked_pnl_per_unit_stake`, reachability
  uses matcher predicates, cache tagged with schema
  versions), §24 (invariant must stay green — not a direct
  concern for this session, but regressions land here),
  §27 (8 tests).
- [`../master_todo.md`](../master_todo.md) — Session 01
  deliverables.
- `plans/arb-improvements/session_6_oracle_scan.md` — the
  2026-04-14 scoping. Reuse heavily; amendments below.
- `plans/arb-improvements/hard_constraints.md` — pre-existing
  oracle constraints (determinism, env-reachability,
  offline-only) all carry forward.
- `env/exchange_matcher.py` — filter predicates. If they
  aren't exported as standalone functions, expose them.
- `env/scalping_math.py` — `locked_pnl_per_unit_stake`,
  `min_arb_ticks_for_profit`. Profitability check calls
  these.
- `data/episode_builder.py` — existing per-day loader.
  Reuse, don't reinvent.

## Amendments to the 2026-04-14 scoping

| Item | 2026-04-14 scope | 2026-04-19 amendment |
|---|---|---|
| Profitability check | "crossed post-commission book" | Use `locked_pnl_per_unit_stake(P_back, P_lay, commission) > 0` directly — matches env. |
| Reachability | "pass the matcher filters" | Include freed-budget reservation check from `scalping-asymmetric-hedging` when evaluating paired-leg fit. |
| Cache header | Not specified | Tag `obs_schema_version=6`, `action_schema_version=4`, `scalping_mode=True`, `created_at`. Hard error on load mismatch. |
| Obs vector | "reuse env `_build_observation`" | Obs vector at oracle time must be the SCALPING obs (version 6), not legacy. Document the obs dim matches `env.observation_space.shape[0]` for a scalping-mode env. |
| Density metric | "`samples / total_ticks`" | Also emit `unique_arb_ticks / total_ticks` — a single tick may produce multiple samples (different runners). Both numbers are useful for curriculum day ordering (Session 05). |

## Locate the code

```
grep -n "locked_pnl_per_unit_stake\|min_arb_ticks_for_profit" env/scalping_math.py
grep -n "max_price_deviation_pct\|junk_filter\|max_back_price\|max_lay_price" env/exchange_matcher.py
grep -n "_build_observation\|_get_obs" env/betfair_env.py
grep -n "def load_day" data/episode_builder.py
grep -n "OBS_SCHEMA_VERSION\|ACTION_SCHEMA_VERSION" env/betfair_env.py
```

Confirm before editing:

1. `locked_pnl_per_unit_stake` accepts
   `(P_back, P_lay, commission)` and returns pounds per
   unit stake.
2. The matcher's "junk filter" (LTP-aware) is callable
   without instantiating a full `ExchangeMatcher`. If not,
   refactor to expose it as a pure function.
3. An env instantiated with `scalping_mode=True` on a day
   produces obs of shape matching
   `OBS_SCHEMA_VERSION=6` (runners × 93 + position × 4).

## What to do

### 1. New module `training/arb_oracle.py`

```python
"""Offline arb oracle scan.

For each tick of each race on a given date, detect moments
where a paired back+lay arb is profitable post-commission
AND reachable through the env's matcher. Emit a cache of
samples for downstream BC pretraining (Session 04) and
curriculum day ordering (Session 05).

Contract (hard_constraints.md §6-§9):
- Offline only. Never invoked inside the training loop.
- Deterministic. Same input -> same bytes.
- Reachability matches env. A sample is emitted only if
  the env would actually place the pair.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from env.scalping_math import locked_pnl_per_unit_stake
from env.exchange_matcher import passes_junk_filter, \
    passes_price_cap  # export these if needed
from env.betfair_env import (
    OBS_SCHEMA_VERSION, ACTION_SCHEMA_VERSION, BetfairEnv,
)


@dataclass(slots=True)
class OracleSample:
    tick_index: int
    runner_idx: int
    obs: np.ndarray  # float32
    arb_spread_ticks: int
    expected_locked_pnl: float


def scan_day(date: str, data_dir: Path,
             config: dict) -> list[OracleSample]:
    """Scan one day; return samples for every profitable
    reachable arb moment."""
    ...


def load_samples(date: str, data_dir: Path,
                 *, strict: bool = True) -> list[OracleSample]:
    """Load cached samples. strict=True asserts header's
    schema versions match current env."""
    ...


def _save_samples_atomic(samples: list[OracleSample],
                         path: Path, header: dict) -> None:
    """Write .npz to .tmp then rename for atomicity."""
    ...
```

### 2. CLI entrypoint

```python
# training/arb_oracle.py bottom

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    scan_p = sub.add_parser("scan")
    scan_p.add_argument("--date", required=True)
    scan_p.add_argument("--dates", default=None)
    args = ap.parse_args()
    dates = [args.date] if not args.dates else \
            args.dates.split(",")
    for d in dates:
        samples = scan_day(d, Path("data/processed"),
                           _load_config())
        print(f"{d}: samples={len(samples)} "
              f"ticks={_count_ticks(d)} "
              f"density={len(samples)/max(_count_ticks(d),1):.4f}")

if __name__ == "__main__":
    _cli()
```

### 3. Cache layout

```
data/oracle_cache/
  {date}/
    oracle_samples.npz    # tick_index, runner_idx,
                          # obs[float32 (N,obs_dim)],
                          # arb_spread_ticks[int8],
                          # expected_locked_pnl[float32]
    header.json           # obs_schema_version,
                          # action_schema_version,
                          # scalping_mode,
                          # created_at, commit_sha,
                          # samples, ticks, density
```

Add `data/oracle_cache/` to `.gitignore` if not already
covered by the general `data/` pattern.

### 4. Export matcher filters as pure functions

If `ExchangeMatcher` doesn't already expose
`passes_junk_filter(price, ltp, max_dev_pct) -> bool` and
`passes_price_cap(price, side, max_back, max_lay) -> bool`,
refactor the class to use those pure functions internally
and export them. The existing tests in
`tests/test_exchange_matcher.py` should still pass.

### 5. Tests — `tests/arb_curriculum/test_arb_oracle.py`

Create directory `tests/arb_curriculum/` with `__init__.py`.
Eight tests per §27:

1. **Synthetic day with one injected arb → one sample.**
   Hand-build a race where exactly one tick has a crossed,
   profitable, reachable book. Assert one sample with
   correct `runner_idx`, `tick_index`.
2. **Price-cap filter compliance.** Build a crossed book
   at a price above `max_back_price`. Assert 0 samples.
3. **Junk filter compliance.** Build a crossed book far
   from LTP. Assert 0 samples.
4. **Empty day.** No arb moments. Zero samples, no crash;
   `.npz` writes anyway (empty).
5. **Determinism.** Scan twice; byte-identical `.npz`.
6. **Round-trip.** Save, `load_samples`, assert equality.
7. **Density metric.** CLI prints
   `samples=X ticks=Y density=X/Y`. Capture stdout.
8. **Obs dim matches env.** On a real-data tick, assert
   `sample.obs.shape[0] == BetfairEnv(..., scalping_mode=True).observation_space.shape[0]`.

### 6. Full-suite check

```
pytest tests/arb_curriculum/ -x
pytest tests/ -q --timeout=120   # DO NOT run during live training
```

The second run is full-suite regression. Per user directive
2026-04-19, **only run the full suite when no training is
active**.

### 7. Progress entry

Append a dated entry to `progress.md`:
- Commit hash.
- Per-day densities on the current training-date window
  (runnable after Session 01 lands; cite at least 3 days).
- Highlight any day whose density is < 0.001 — those are
  curriculum-hostile and Session 05 will reorder them.

### 8. Commit

```
feat(training): offline arb oracle scan with per-day density cache

Produces per-date .npz caches of every (obs, runner_idx,
arb_spread_ticks, expected_locked_pnl) tick that is both
profitable post-commission and reachable through the env's
matcher. Consumed downstream by BC pretrain (Session 04)
and curriculum day ordering (Session 05).

Contract: offline-only (never runs inside training loop);
deterministic (same input, same bytes); env-reachable
(filter mirrors env exactly via the pure predicates
exported from ExchangeMatcher). Cache tagged with
obs_schema_version, action_schema_version, scalping_mode
so a silent schema bump hard-fails on load rather than
corrupting BC targets.

Why: 2026-04-19 reward-densification-probe / gene-sweep
converged on the diagnosis that random arbing is expected-
negative, so policy gradient finds "arb less" before "arb
better". The oracle gives us a rule-based source of
profitable-arb samples that BC can warm-start the policy
from.

Tests: 8 in tests/arb_curriculum/test_arb_oracle.py.

Not changed: matcher behaviour (predicates exposed, logic
untouched), env schemas, reward path, PPO, controller.

Per plans/arb-curriculum/hard_constraints.md s1-s9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT run the oracle inside the training loop. Ever.
- Do NOT emit samples the env would reject — the oracle is
  the source of BC truth; leakage here poisons BC.
- Do NOT write non-deterministic output. Sort samples
  before writing if iteration order matters.
- Do NOT run `pytest tests/ -q` during active training.
  (Operator directive 2026-04-19 after a pytest run
  collided with a live worker.)

## After Session 01

1. Append progress entry with per-day densities.
2. Hand back for Session 02 (matured-arb bonus).
