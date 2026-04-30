# Session 03 — dual-mode passive-fill env (volume / pragmatic)

Implementation of the design in
`session_prompts/03_dual_mode_fill_env.md`. Adds a per-day passive-
fill mode flag (`Day.fill_mode`) that is auto-detected at load time
and threaded through the env so historical days (where F7 means
`RunnerSnap.total_matched == 0` everywhere) can still produce passive
fills via market-level prorated volume, while post-F7-fix days remain
on the spec-faithful per-runner volume path.

## Implementation summary

| File | Change |
|---|---|
| [data/episode_builder.py](../../../../data/episode_builder.py) | `Day.fill_mode: Literal["volume", "pragmatic"] = "volume"` field; `_build_day` auto-detects per the design spec (volume iff any active runner on any tick of any race has `total_matched > 0`, else pragmatic). |
| [env/bet_manager.py](../../../../env/bet_manager.py) | `PassiveOrderBook.__init__` gains `fill_mode` arg (default `"volume"`); `on_tick` now dispatches to `_volume_phase_1` or `_pragmatic_phase_1`; existing volume code refactored verbatim into `_volume_phase_1`; new `_pragmatic_phase_1` prorates `tick.traded_volume` delta across runners by visible book size with the same crossability gate. `BetManager` gains `fill_mode` field, forwards to its `PassiveOrderBook`. |
| [env/betfair_env.py](../../../../env/betfair_env.py) | Both `BetManager()` constructions in `reset()` and the inter-race transition pass `fill_mode=self.day.fill_mode`; `RaceRecord.fill_mode: str = "volume"`; settle path stamps `self.day.fill_mode` onto each `RaceRecord`; `_get_info` returns `info["fill_mode_active"]`. |
| [agents/ppo_trainer.py](../../../../agents/ppo_trainer.py) | `EpisodeStats.fill_mode: str = "volume"`; rollout-end stats reads `info["fill_mode_active"]`; `_log_episode` writes `record["fill_mode"]`; per-episode operator log line and the WS `progress["detail"]` mirror gain `mode=...`. |
| [tests/test_passive_order_book_dual_mode.py](../../../../tests/test_passive_order_book_dual_mode.py) | New file. 8 regression guards: volume-mode-byte-identical, three pragmatic-mode behaviours (attribution, crossability gate, zero-visible edge case), two `Day.fill_mode` auto-detect cases, two telemetry surfaces. |
| [tests/test_episode_builder.py](../../../../tests/test_episode_builder.py) | Three new `_build_day` tests: synthetic `total_matched > 0` → volume; all-zero → pragmatic; empty-day → pragmatic. |

Phase 2 (the fill check + junk-band filter + threshold) is unchanged
— both modes feed the same downstream mechanic per the hard
constraint.

## Test results

```
python -m pytest tests/ -q
2694 passed, 7 skipped, 6 failed, 1 xfailed in 475.04s (0:07:55)
```

Failures break down as:

- **2** F7 regression guards in `tests/test_per_runner_total_matched_data.py`
  — `test_at_least_one_active_runner_has_nonzero_total_matched` and
  `test_market_with_high_traded_volume_has_per_runner_signal`. These
  test the data, not the env, and continue to fail for the same
  reason they failed before this plan: real historical parquets
  carry `total_matched == 0`. **Expected and intentional** per the
  session prompt's hard constraint "Don't change the F7 regression
  test."
- **3** pre-existing failures unrelated to this plan, confirmed by
  running them on `git stash` (master HEAD without my changes):
  - `test_orchestrator.py::test_skip_training_jumps_to_eval` —
    `UnboundLocalError` on `torch` in `training/run_training.py:1200`.
  - `test_ppo_advantage_normalisation.py::test_real_update_policy_loss_bounded`
    — pre-existing.
  - `test_ppo_stability.py::test_default_threshold_is_literature_standard`
    — asserts `kl_early_stop_threshold == 0.03` but the default is
    now `0.15` (CLAUDE.md "Per-mini-batch KL check (Session 02,
    2026-04-25)"). Test is stale.
- **1** `test_kl_early_stop_is_per_mini_batch_not_per_epoch` flaked
  in the full suite run; passes in isolation. Order-dependent /
  shared-state, not caused by this plan.

New tests:

- `tests/test_passive_order_book_dual_mode.py` — 8/8 pass.
- `tests/test_episode_builder.py` extension — 3/3 new pass; full
  file 69/69 pass.

## Verification on a real historical day

Per the session prompt's "Verification" step:

```
$ python -c "from data.episode_builder import load_day; ..."
date=2026-04-11  races=86  fill_mode='pragmatic'
sample race 1.256488956: ticks=222 final_market_tv=6,543,021
first-tick first-runner total_matched=0.0
```

`data/processed/2026-04-11.parquet` correctly auto-detects to
pragmatic mode. The recommended sample race
(`market 1.256488956`, 222 ticks, £4.9M+ market traded volume —
final tick £6.5M, even heavier than the prompt suggested) confirms
F7-shape data: every active runner's `total_matched == 0`.

A stub policy (zero action, no aggressive bets) stepped 50 ticks of
the first race in pragmatic mode without crashing, with
`info["fill_mode_active"] == "pragmatic"` surfaced both at reset and
mid-race.

## Hand-off note for Session 04 (UI surfacing)

The telemetry field names this session locked in:

- Per-tick `info` dict: `info["fill_mode_active"]` (string,
  `"volume"` | `"pragmatic"`).
- Per-race `RaceRecord.fill_mode` (string, same values).
- Per-episode `EpisodeStats.fill_mode` and the same key on each
  `episodes.jsonl` row.
- Per-episode operator log line and WS `progress["detail"]` block:
  `mode=<value>` on the second indented row alongside
  `pnl=£... bets=N loss=...`.

**Do not rename or remove these fields without coordinating with
Session 04.** They are the load-bearing surface that prevents
cohort metrics from blending modes silently.

Pre-plan rows in older `episodes.jsonl` lack `fill_mode` — downstream
readers (the learning-curves panel, validation scripts) must default
to `"volume"` on absence, matching the dataclass default.

## Notes on scoreboard comparability

Per the hard constraint "Reward magnitudes change":

- Pragmatic-mode passive fill timing differs from volume-mode by
  construction. Cohort scoreboards comparing pre-plan rows to
  post-plan rows are valid only **within-mode**.
- Within volume mode, behaviour is byte-identical to pre-plan
  (verified by the
  `test_volume_mode_unchanged_on_synthetic_tick` regression guard).
- Within pragmatic mode, no historical baseline exists — this is
  the new fall-back path for F7-shape data.

## Out of scope (explicitly deferred)

- UI surfacing → Session 04
  (`session_prompts/04_dual_mode_fill_ui.md`).
- Curriculum filtering by mode in the trainer.
- StreamRecorder1 per-runner volume fix → Session 02
  (`session_prompts/02_streamrecorder_per_runner_volume.md`).
- PRO Historical backfill — audit Option B; not this session.
- Spec rewrites in `docs/betfair_market_model.md` — pragmatic mode
  is documented in code + this writeup as an approximation; the
  spec stays correct as the reference implementation.
