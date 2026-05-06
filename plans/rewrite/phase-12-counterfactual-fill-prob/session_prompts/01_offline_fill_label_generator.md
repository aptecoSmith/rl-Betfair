---
session: phase-12-counterfactual-fill-prob / S01
phase: rewrite/phase-12-counterfactual-fill-prob
parent_purpose: ../purpose.md
---

# S01 — offline fill-label generator + cache CLI

## Context

Read `purpose.md` and `hard_constraints.md` first. This session
builds the V1 offline label generator. No trainer changes — that's
S02. No validation cohort — that's S03.

The shape mirrors the v2 oracle scan
([training_v2/arb_oracle.py](../../../../training_v2/arb_oracle.py))
intentionally. Same data dependencies (load_day from
`data/episode_builder`), same tick walk, same env-matcher rule
checks. The DIFFERENCE: instead of emitting samples for "this is
a profitable arb moment", we emit dense labels for "would the
passive at this (tick, runner, side) fill before force-close".

## Pre-reqs

Read these before touching code:

- [training_v2/arb_oracle.py](../../../../training_v2/arb_oracle.py)
  end-to-end. The label generator borrows its structure (env
  build, per-race tick walk, env-matcher rule application). The
  output schema is different but the shape is closely related.

- [env/exchange_matcher.py](../../../../env/exchange_matcher.py)
  — confirm the junk filter, price cap, and best-price selection
  rules. The label generator must apply identical filters.

- [env/tick_ladder.py::tick_offset](../../../../env/tick_ladder.py)
  for the price-shifting helper.

- [env/scalping_math.py::min_arb_ticks_for_profit](../../../../env/scalping_math.py)
  — though Phase 12 doesn't filter on profitability (it labels
  fill probability regardless of profit), we use the same tick
  ladder math.

- CLAUDE.md §"Order matching: single-price, no walking" — the
  matcher's contract. The label uses the same junk filter and
  cap rules.

## V1 label semantics (LOCKED)

For each pre-race tick T in race R, for each active + priceable
runner k whose top-of-book passes the env matcher checks:

**back-first scalp label:**

```python
P_back   = best_atb_after_junk_filter(runner_k, tick=T)
            (i.e. max-price level inside ±max_dev_pct of LTP,
             passing max_back_price cap)
P_lay    = tick_offset(P_back, arb_spread_ticks, direction=-1)
T_close  = T + min(force_close_before_off_seconds_in_ticks,
                    in_play_tick - T)

label_back_first(T, k) = 1.0 if EXISTS t in (T, T_close]:
                          best_back(runner_k, tick=t) ≤ P_lay
                          AND that t is still pre-race (not in_play)
                          0.0 otherwise
```

`best_back(runner_k, tick=t) ≤ P_lay` means: at some tick t after
T, a participant placing an aggressive back at price P_lay would
match against the lay queue at price P_lay (which contains our
hypothetical passive). It's a NECESSARY condition for a fill,
not a SUFFICIENT one (queue position not modelled here — that's
V2).

**lay-first scalp label:**

Symmetric, swapping sides. Detail in `purpose.md`.

For V1 we generate BOTH labels per (tick, runner). Current
training uses only back-first scalps but lay-first is in scope
for future work.

## Deliverables

### 1. `training_v2/fill_label_scan.py`

Pure offline label generator. Public API:

```python
@dataclass(slots=True)
class FillLabel:
    tick_index: int      # global pre-race tick index across day
    runner_idx: int      # env's slot
    label_back: float    # 0.0 or 1.0
    label_lay: float     # 0.0 or 1.0
    P_back: float        # the price assumed
    P_lay: float
    P_lay_fill_first_tick: int   # earliest t at which best_back ≤ P_lay
                                  # (-1 if never reached). For diagnostic
                                  # use; not consumed by training.

def scan_day(date, data_dir, config, *,
             arb_spread_ticks: int) -> list[FillLabel]:
    """Walk every pre-race tick × active runner, label both sides."""

def save_labels(labels, date, data_dir, config,
                arb_spread_ticks, total_ticks) -> Path:
    """Write data/fill_labels/{date}/{arb_spread_ticks}_ticks.npz"""

def load_labels(date, data_dir, *, strict=True,
                expected_arb_spread_ticks: int | None = None,
                expected_obs_dim: int | None = None) -> list[FillLabel]:
    """Hard-error on schema/spread mismatch when strict."""
```

The cache directory pattern is `data/fill_labels/{date}/`,
distinct from `data/oracle_cache_v2/`. Within a date, the
`{arb_spread_ticks}_ticks.npz` filename allows multiple
spread variants to coexist (we'll need this when the operator
wants to test spread=10 vs spread=20).

### 2. `training_v2/fill_label_cli.py`

Thin CLI mirroring `oracle_cli.py`:

```
python -m training_v2.fill_label_cli scan --date 2026-05-03 \
                                          [--arb-spread-ticks 20]
python -m training_v2.fill_label_cli scan --dates 2026-04-29,...
```

Per-day stdout line:
```
{date}: labels=N (back+lay) ticks=T positive_back={X:.4f}
        positive_lay={Y:.4f} arb_spread_ticks={S}
```

`positive_back` / `positive_lay` are the densities of label=1.0
in each side. Sanity expectations:
- Conservative (price-reachable) label is an UPPER bound on real
  fill rate.
- Real fill rate observed in cohorts: 0.17 – 0.21.
- Expect this label density: 0.30 – 0.50. If much higher, the
  label is too loose (V2 queue-aware would tighten it). If much
  lower, the matcher rules diverge from the live env.

### 3. Tests `tests/test_v2_fill_labels.py`

Eight tests:

1. `test_scan_day_emits_one_label_per_priceable_runner_tick`
   — synthetic 1-runner 2-tick day; assert `len(labels) == 2`
   (one per pre-race tick). Each label has both back and lay
   fields populated.

2. `test_label_positive_when_price_walks_to_lay`
   — synthetic day where tick 0 has back=5.0, tick 1 has
   best_back=4.95 (P_lay would be at 5.95 for spread=20 so
   tick 1's best_back=4.95 doesn't cross). Construct a case
   where the price DOES walk through P_lay — assert label=1.0.

3. `test_label_zero_when_price_never_reaches_lay`
   — synthetic day where best_back stays above P_lay for all
   subsequent ticks. Assert label_back=0.0.

4. `test_junk_filter_applied_at_open_tick`
   — same as oracle: ATB outside ±50 % of LTP → no label
   emitted (priceable check fails).

5. `test_price_cap_applied_at_open_tick`
   — ATB > max_back_price → no label emitted for back-first.

6. `test_force_close_horizon_respected`
   — set force_close_before_off_seconds=60. Verify the search
   for fill stops at T_close, not at the end of the race.

7. `test_determinism`
   — scan twice, byte-identical labels.

8. `test_round_trip_save_load`
   — `save_labels` then `load_labels` returns equal data;
   strict mode rejects mismatched arb_spread_ticks header.

### 4. lessons_learnt.md entry

After running on a real training day, record:
- Observed positive density (back / lay) compared to the agent's
  natural-fill rate from the overnight cohorts.
- Wall time per scan-day vs the §9 5-minute budget.
- Any rule that diverged from the env matcher and had to be
  fixed.

## Stop conditions

- **Stop and ask** if scan-day exceeds 10 minutes on a typical
  training day. The §9 budget is 5 minutes; 10 minutes means we
  need to vectorise before continuing. Stop and propose the
  vectorisation rather than letting S02 wait.

- **Stop and ask** if positive density is < 0.10 or > 0.70 on a
  real day. < 0.10 means the matcher rules are wrong (too
  restrictive); > 0.70 means the label is far too loose
  (compromise V2 immediately). Either bounds violation suggests
  the V1 label is the wrong shape.

- **Stop and ask** if ANY tick × runner produces both
  `label_back=1.0` and `label_lay=1.0` simultaneously at the
  same arb_spread_ticks. That means the same runner had a
  fillable back-first AND fillable lay-first opportunity at
  the same tick — possible in principle (it's a "true two-sided
  arb") but rare. If it's frequent, suspect a bug in the
  symmetric label.

## Done when

- `python -m training_v2.fill_label_cli scan --date 2026-05-03`
  prints a sensible density line.
- All 8 tests in `tests/test_v2_fill_labels.py` pass.
- Existing tests unchanged (`pytest tests/test_v2_*.py -q` green).
- `lessons_learnt.md` has a S01 entry with observed densities,
  wall time, and any matcher-rule deltas.
- Commit: `feat(rewrite): phase-12 S01 - offline fill-label
  generator + cache CLI`.
