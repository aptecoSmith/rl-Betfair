---
session: phase-8-oracle-bc-pretrain / S01
phase: rewrite/phase-8-oracle-bc-pretrain
parent_purpose: ../purpose.md
---

# S01 — port oracle scan to v2; cache + density CLI

## Context

Read `plans/rewrite/phase-8-oracle-bc-pretrain/purpose.md` and
`hard_constraints.md` first.

The v1 oracle (`training/arb_oracle.py`) already works end-to-end.
It uses only shared env code (`env/`, `data/`), so the core scan
logic doesn't need rewriting — the question is whether v2's obs
format matches v1's, and whether to reuse or copy.

This session answers that question, produces a working cache CLI,
and writes the regression tests. No BC pretrain yet (that's S02).

## Pre-reqs

Read these before touching any code:

- [`training/arb_oracle.py`](../../../../training/arb_oracle.py)
  — full file. Understand `scan_day`, `save_samples`, `load_samples`,
  and the `OracleSample` dataclass. Note the obs construction:
  `env._static_obs[race_idx][tick_idx]` + `zero_agent_state` +
  `zero_position`.
- [`agents_v2/discrete_policy.py:132-170`](../../../../agents_v2/discrete_policy.py)
  — `BaseDiscretePolicy.__init__`; confirm `obs_dim` is the
  parameter it reads and what value the cohort runner passes for it.
- [`training_v2/cohort/worker.py`](../../../../training_v2/cohort/worker.py)
  — find where `BetfairEnv` is constructed for a training agent
  and how `obs_dim` is derived (likely from
  `env.observation_space.shape[0]`). This is the obs_dim the
  oracle must match.
- [`env/betfair_env.py`](../../../../env/betfair_env.py) — look up
  `OBS_SCHEMA_VERSION`, `AGENT_STATE_DIM`, `SCALPING_AGENT_STATE_DIM`,
  `POSITION_DIM`, `SCALPING_POSITION_DIM`. Confirm the oracle's
  zero-state construction covers all dimensions expected by v2.

## Decision: reuse vs copy

After reading the pre-reqs, answer:

> Does `training/arb_oracle.py::scan_day` produce obs of shape
> `(obs_dim,)` where `obs_dim == env.observation_space.shape[0]`
> for a `scalping_mode=True` v2 env?

- **If yes:** import `training.arb_oracle` directly from v2 code.
  No copy. Add `from training.arb_oracle import scan_day,
  save_samples, load_samples, OracleSample` at the top of the new
  v2 CLI. Document the confirmed parity in `lessons_learnt.md`.
- **If no (obs_dim mismatch):** copy the file to
  `training_v2/arb_oracle.py` and patch only the
  zero-agent-state construction to cover v2's dimensions. Do NOT
  change the scan logic or cache format header (keep
  `obs_schema_version` sourced from `env.betfair_env.OBS_SCHEMA_VERSION`).
  Document the divergence in `lessons_learnt.md`.

Stop and ask before implementing if the mismatch is larger than the
agent-state vector (e.g. if static_obs shape also differs).

## Deliverables

### 1. Oracle CLI entrypoint (`training_v2/oracle_cli.py`)

A thin CLI that calls the scan and prints density:

```python
"""CLI: python -m training_v2.oracle_cli scan --date 2026-05-01
                                          scan --dates 2026-04-29,...
"""
```

Subcommands:
- `scan --date DATE` — scan one day; print
  `{date}: samples={N} ticks={T} density={N/T:.4f}
   unique_arb_ticks={U} unique_arb_ticks_density={U/T:.4f}`.
- `scan --dates DATE,DATE,...` — iterate; same per-line output.

Reads config from `config.yaml` (same path as the cohort runner
uses). Uses whichever of reuse/copy was chosen above.

Cache lands in `data/oracle_cache/{date}/` (same as v1).

### 2. Tests (`tests/test_v2_oracle.py`)

Seven tests. Mirror v1's `tests/arb_curriculum/test_arb_oracle.py`
structure but confirm v2 obs_dim compatibility:

1. `test_scan_day_produces_obs_matching_v2_env_obs_dim` — instantiate
   a real `BetfairEnv(day, config, scalping_mode=True)` on one
   training day; call `scan_day`; assert every sample's
   `obs.shape[0] == env.observation_space.shape[0]`.
   **This is the primary v2 compatibility gate.** Fail here =
   copy path needed.
2. `test_scan_day_synthetic_one_arb_one_sample` — hand-build a
   race with exactly one profitable reachable crossed book; assert
   exactly one sample returned with correct `runner_idx` and
   `tick_index`.
3. `test_price_cap_filter` — crossed book above `max_back_price`;
   assert 0 samples.
4. `test_junk_filter` — crossed book far from LTP; assert 0
   samples.
5. `test_determinism` — scan same day twice; byte-identical `.npz`.
6. `test_round_trip` — `save_samples` then `load_samples`; samples
   equal field-by-field.
7. `test_schema_version_mismatch_raises` — write a cache with a
   wrong `obs_schema_version` in `header.json`; assert
   `load_samples(..., strict=True)` raises `ValueError`.

### 3. `lessons_learnt.md` entry

After the obs_dim check, record:
- Whether reuse or copy was chosen and why.
- The actual obs_dim on a real training day.
- Any dimension fields that required adjustment.

## Stop conditions

- **Stop and ask** if `env._static_obs` is not accessible from
  outside the env (e.g. private and unset before a rollout has
  run). The oracle relies on this for feature engineering without
  stepping through the env manually.
- **Stop and ask** if the obs_dim mismatch is in `_static_obs`
  itself (not just the agent-state suffix). A change there would
  affect the scan logic, not just the zero-state construction.
- **Stop and ask** if `config.yaml` doesn't expose
  `training.betting_constraints` or `training.starting_budget`
  (the oracle's budget check uses these).

## Done when

- `python -m training_v2.oracle_cli scan --date 2026-04-29`
  completes and prints a density line.
- All 7 tests in `tests/test_v2_oracle.py` pass.
- Existing tests unchanged (`pytest tests/ -q` green, or at least
  no new failures).
- `lessons_learnt.md` has a reuse/copy decision entry.
- Commit: `feat(rewrite): phase-8 S01 - oracle scan CLI + v2
  obs-parity test`.
