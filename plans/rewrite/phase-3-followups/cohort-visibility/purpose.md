---
plan: rewrite/phase-3-followups/cohort-visibility
status: design-locked
opened: 2026-05-02
depends_on: rewrite/phase-3-followups/force-close-architecture (Session 02
            running 2026-05-02 surfaced the visibility gap mid-cohort)
---

# Cohort-visibility follow-on — see results as agents land

## Purpose

The Phase-3 cohort runner buffers its per-agent results to
end-of-generation. With `n_generations=1` (the standard
mechanics-iteration shape used by `no-betting-collapse`,
`force-close-architecture`, and any future single-gen probe),
this means **no per-agent data lands on disk for the entire
~3.5-hour run**. The operator can't tell whether the cohort is
producing the verdict signal until the very end — by which point
the only recourse for a divergent run is to throw away the wall
time and restart.

Concretely, while running force-close-architecture Session 02 on
2026-05-02, the operator asked "how's it going?" mid-cohort and
the only LIVE per-agent data available was:

1. Free-form `cohort.log` lines (`Agent X eval [Y] reward=... pnl=...
   bets=... arbs=N/M locked=... naked=... wall=...`) — useful for
   sanity-checking that training hadn't crashed, but the `arbs=N/M`
   format pre-dates the Session 01 / 02 counter additions, so it
   doesn't surface `arbs_closed` (agent close_signal),
   `arbs_force_closed` (T−N flat), or `arbs_stop_closed` (mid-race
   MTM trigger). The numbers visible are only matured + naked.
2. `models.db` `evaluation_days` table — written per-agent, but
   carries the **pre-Session-01 schema only**. New counters from
   `arb-signal-cleanup` (force_closed), `force-close-architecture
   S01` (target_pnl_refused, pairs_opened, closed_pnl,
   force_closed_pnl), and `S02` (stop_closed, stop_closed_pnl) are
   all missing.
3. `scoreboard.jsonl` — has the full breakdown but is
   **buffered to end-of-generation** in
   [`training_v2/cohort/runner.py:262-272`](../../../../training_v2/cohort/runner.py).
   For `n_generations=1` cohorts that means flushing only at
   end-of-cohort.

A side-script peek built mid-run (`C:/tmp/peek_session02.py`) hit
all three limits: it could compute an old-proxy `fc_rate` from
`(arbs_completed, arbs_naked)` only, with the explicit caveat that
"true Session 02 fc_rate is ≤ this value because stop-closed
pairs are folded into neither bucket." The verdict signal that
matters — does the new mechanism actually fire, and how does it
shift the macro fc rate — is not visible until cohort end.

This plan closes that gap.

## Why now

The plan-after-plan iteration cadence is constrained by visibility.
A 3.5-hour cohort that produces a verdict signal only at the end
forces the operator to commit to "either wait, or kill and lose
the run." That tradeoff dominates the iteration loop for every
mechanics plan downstream of Phase 3:

- `force-close-architecture` Sessions 01 (FAIL) and 02 (running)
  each spent ~3.5 h producing a single verdict.
- `no-betting-collapse` Session 02 (planned re-runs of AMBER v2
  baseline + ablations) will repeat the same shape.
- Any future stacked or threshold-sweep cohort eats this cost
  per iteration.

Visibility shrinks the feedback loop without changing the
mechanics — operator can detect "agent 1 looks fine, agents 2-3
diverged hard" 30 min in instead of 3.5 h in, and can decide to
kill+restart with confidence.

## What's locked

### Mechanics changes are NOT touched

This plan is purely about telemetry. The pair-placement path,
close-leg path, force-close logic, equal-profit math, reward
shaping, and the cohort GA loop are byte-identical pre/post each
session of this plan. The trainer's `_ppo_update`, the
`RolloutCollector`, and the env are off-limits in the same way
they're off-limits in `throughput-fix`.

Cross-cohort comparison against AMBER v2
(`registry/v2_amber_v2_baseline_1777577990/`) and the existing
`force-close-architecture` baselines must remain valid; mechanics
changes are not allowed to silently piggyback on a visibility
plan.

### Same `--seed 42` pre/post

Every test cohort in this plan re-runs at `--seed 42` against the
same data window. Differing seeds invalidate the comparison.

### Schema changes are forward-only with default-tolerant readers

`models.db` evaluation_days widening adds new columns; existing
queries against the old columns must keep working. v1 UI and any
existing scoreboard / replay tooling are spot-checked before
shipping. Migration is `ALTER TABLE ... ADD COLUMN ... DEFAULT NULL`
on existing dbs, performed idempotently in `ModelStore.__init__`.

### No env edits

The env (`env/betfair_env.py`, `env/bet_manager.py`,
`env/exchange_matcher.py`) is off-limits. The data this plan
surfaces ALREADY exists in the env's per-episode info dict and
in the cohort `EvalSummary` / scoreboard row paths. The plumbing
gap is between the producer and the on-disk consumer; the env
is not the producer that needs changing.

## Success bar

The plan ships GREEN iff at least one session produces, on a
fresh 12-agent / 1-gen cohort run:

1. **Per-agent scoreboard rows land in `scoreboard.jsonl` as
   each agent completes its eval** (sequential branch). At the
   ~18-min/agent cadence of AMBER v2 single-agent walls, that
   means new rows visible roughly every 18 min from agent 1
   onwards.
2. **`models.db` evaluation_days carries the new counters**
   (`arbs_closed`, `arbs_force_closed`, `arbs_stop_closed`,
   `pairs_opened`, `closed_pnl`, `force_closed_pnl`,
   `stop_closed_pnl`, `arbs_target_pnl_refused`) and the pre-
   existing v1 UI / scoreboard readers continue to work
   unchanged.
3. **A peek script (`C:/tmp/peek_cohort.py` or similar) reads
   the live db / jsonl and surfaces the macro verdict bar
   metrics** — mean fc_rate (true denominator), positive eval
   P&L count, mean policy-close fraction, mean stop-close
   fraction. Same metrics force-close-architecture's `findings.md`
   table demands.
4. **CUDA↔CUDA self-parity holds.** Two CUDA runs at the same
   seed produce bit-identical `total_reward` and `value_loss_mean`
   per agent. This plan changes ONLY where data is written, not
   what is computed; parity must hold by construction. Same
   load-bearing guard `phase-3-cohort` Session 01b shipped.
5. **No regression in cohort wall.** The added writes are
   I/O-bound and small (one ALTER TABLE on init, one extra
   `sf.write/flush` per agent). End-of-cohort wall delta < 1 %
   vs AMBER v2 baseline.

The plan ships **GREEN-with-stretch** if a session ALSO produces:

6. **Per-episode JSONL emission** (`<output_dir>/episodes/
   <agent_id>.jsonl`) — every PPO rollout episode emits a row
   with the standard env info dict (per-day reward, P&L, bets,
   approx_kl, value_loss, full arb breakdown). At ~2-3 min per
   episode this gives intra-agent visibility for triaging
   divergent runs.

If only (1)+(2) ship and (3) doesn't, the plan is GREEN on
"data exists somewhere on disk live"; the peek script is a
follow-on (it's a tool, not the foundation).

If (4) breaks: stop and triage. The plan does not ship if
self-parity breaks.

## Sessions

### Session 01 — per-agent scoreboard flush + schema widening

Two coupled changes that together unblock the verdict-bar metrics
for any tool that reads `scoreboard.jsonl` or queries
`models.db`:

**(a) Move `scoreboard.jsonl` write inside the per-agent loop on
the sequential branch.**
[`training_v2/cohort/runner.py:262-272`](../../../../training_v2/cohort/runner.py)
currently writes all rows post-loop:

```python
# After all agents in this gen done:
for idx, result in enumerate(results):
    row = _agent_result_to_scoreboard_row(...)
    sf.write(json.dumps(row) + "\n")
    sf.flush()
```

The sequential branch already has every agent's `result` available
inside its own iteration; the post-loop double-iteration is
unnecessary. Move the write inside the loop:

```python
for idx, genes in enumerate(cohort):
    ...
    result = train_one_agent_fn(...)
    results[idx] = result
    total_agents_trained += 1
    # NEW — write immediately so per-agent visibility is live.
    row = _agent_result_to_scoreboard_row(
        result=result, generation=generation, agent_idx=idx,
        eval_day=eval_day, training_days=list(training_days),
    )
    sf.write(json.dumps(row) + "\n")
    sf.flush()
```

The batched branch keeps its post-cluster write — agents in a
batched cluster don't finish independently, so per-agent
visibility within a batched cluster isn't structurally available
without the batched-rollout collector emitting per-agent
sub-events. Out of scope; document it in the session note as a
batched-mode visibility caveat.

**(b) Widen `models.db` evaluation_days schema.**
[`registry/model_store.py`](../../../../registry/model_store.py)
defines the table. Add columns:

| New column | Source field on `EvalSummary` |
|---|---|
| `arbs_closed` | `arbs_closed` |
| `arbs_force_closed` | `arbs_force_closed` |
| `arbs_stop_closed` | `arbs_stop_closed` |
| `arbs_target_pnl_refused` | `arbs_target_pnl_refused` |
| `pairs_opened` | `pairs_opened` |
| `closed_pnl` | `closed_pnl` |
| `force_closed_pnl` | `force_closed_pnl` |
| `stop_closed_pnl` | `stop_closed_pnl` |

Migration runs idempotently in `ModelStore.__init__` via
`ALTER TABLE ... ADD COLUMN ... DEFAULT NULL` for each missing
column. Existing dbs (AMBER v2 baseline, Session 01 cohort,
Session 02 cohort) get the new columns added with NULL on legacy
rows; the writer code defaults the new fields to 0 / 0.0 so
post-plan rows carry honest values.

The `record_evaluation` writer (or whichever method writes the
row — TBD by reading `model_store.py`) is widened to take and
persist the new fields. The `EvalSummary` → SQLite mapping is
already there for the pre-existing fields; this is an extend, not
a rewrite.

**(c) Document the new shape.** Update CLAUDE.md
§"`info["realised_pnl"]` is last-race-only" or a new neighbouring
note that records the SQLite schema as the canonical
per-agent live readout. Without this, a future operator hitting
the same visibility gap will have to re-derive the answer.

End-of-session bar (per "Success bar" §1, §2, §4, §5).

Hard constraints on this session in particular:

- **Self-parity per agent.** Running an agent in a cohort with
  the new write paths must produce bit-identical training metrics
  (loss curves, gradient magnitudes, action distributions) to
  running with the pre-plan code. The change is on the I/O side
  only.
- **Idempotent migration.** Running `ModelStore.__init__` on an
  existing post-plan db must not error or duplicate-add columns.
  Standard `PRAGMA table_info` introspection guard.
- **Default-tolerant readers.** Any downstream reader of
  `evaluation_days` (v1 UI, scoreboard tooling, the side-script
  pattern) must tolerate NULL on the new columns for legacy rows
  without crashing.

Tests:

1. `tests/test_v2_cohort_runner.py::test_scoreboard_writes_per_agent_in_sequential_mode`
   — drive a 2-agent cohort with mocked `train_one_agent_fn`,
   assert scoreboard.jsonl has 1 row after agent 1, 2 rows after
   agent 2.
2. `tests/test_model_store.py::test_evaluation_days_schema_includes_session_02_counters`
   — fresh-init ModelStore exposes the new columns.
3. `tests/test_model_store.py::test_evaluation_days_migration_idempotent`
   — pre-plan db with old schema goes through `ModelStore(...)`
   twice, no errors, columns added once.
4. `tests/test_v2_cohort_worker.py::test_eval_summary_persisted_to_evaluation_days`
   — mocked agent run with non-zero new counters → DB row carries
   the values.
5. `tests/test_v2_cohort_runner.py::test_legacy_rows_default_tolerant`
   — read a row written by pre-plan code path (NULL on new
   columns) without erroring.

Session prompt: `session_prompts/01_per_agent_visibility.md`.

### Session 02 — peek script + verdict-metric library

Build a small reusable peek tool that consumes the new live data
shape and surfaces the verdict-bar metrics any cohort needs:

- mean / median fc_rate (TRUE denominator including stop / closed
  / force_closed)
- positive eval P&L count
- median policy-close fraction
- median stop-close fraction
- mean / max naked-back catastrophe loss

Lives at `tools/peek_cohort.py` (NOT `C:/tmp/...` — the prior peek
was throwaway; this is the canonical version). Reads either
`models.db` (preferred, lowest latency) OR `scoreboard.jsonl`
(more fields per row), picks whichever is more recent for each
agent.

Optional extension: if the per-episode JSONL stretch goal (§6) is
landed, the peek tool also surfaces per-day curves for the most
recent agent ("is this agent's training diverging mid-day?").

End-of-session bar:

1. Running `python -m tools.peek_cohort registry/<run-dir>/` on
   the in-flight Session 02 cohort dir reproduces the full
   verdict table for completed agents.
2. Same tool run on the AMBER v2 baseline dir reproduces the
   table from `findings.md` (consistency check against known
   data).
3. Tool surfaces a clear "X/12 agents complete" header so the
   operator sees progress at a glance.

Test:

- `tests/test_peek_cohort.py` — synthetic 3-row models.db /
  scoreboard.jsonl pair, assert the tool emits the expected
  metrics within float tolerance.

Session prompt: NOT YET WRITTEN. Scaffold once Session 01's
verdict is in.

### Session 03 — verdict + writeup (optional, only if stretch §6 fires)

Per-episode JSONL emission from the cohort path. Hooks into the
trainer's existing per-episode info dict and writes to
`<output_dir>/episodes/<agent_id>.jsonl`. Updates `peek_cohort`
to surface intra-agent curves.

End-of-session bar:

1. Per-episode JSONL files appear at `<output_dir>/episodes/
   <agent_id>.jsonl` while the agent is training, one row per
   PPO rollout episode (training + eval days).
2. Self-parity holds.
3. Cohort wall regression < 1 %.

Session prompt: NOT YET WRITTEN. Trivial; gated on operator
asking for it. Default-disabled flag (cf. throughput-fix
deferred-tensor pattern) so cohorts that don't want the I/O can
opt out.

## Hard constraints

In addition to all rewrite hard constraints
(`plans/rewrite/README.md` §"Hard constraints"), phase-3-cohort
constraints, and inherited from `force-close-architecture`,
`no-betting-collapse`, and `throughput-fix`:

1. **No env edits.** Throughput- and visibility-side work both
   stay in `training_v2/` and `registry/`. `env/betfair_env.py`,
   `env/bet_manager.py`, `env/exchange_matcher.py` are off-
   limits.
2. **No reward-shaping changes.** Same as every other rewrite
   plan; visibility is purely on the side of writing already-
   computed values to disk.
3. **Self-parity is load-bearing.** Two CUDA runs at the same
   seed must produce bit-identical training metrics pre/post
   each session of this plan. The write-shape changes do not
   touch the gradient pathway.
4. **Schema changes are forward-only.** `ALTER TABLE ... ADD
   COLUMN` only. No `DROP COLUMN`, no field renames. Old code
   reading old schemas continues to work.
5. **Migrations are idempotent.** `PRAGMA table_info` checks
   gate every `ALTER TABLE`.
6. **Same `--seed 42` for every cohort.** Cross-cohort
   comparison invariant.
7. **NEW output dirs for every cohort run.** Don't overwrite
   AMBER v2, force-close-architecture, or any other baseline.
8. **No GA gene additions.** Same as every other rewrite plan.
9. **No re-import of v1 trainer / policy / rollout.** Phase 2/3
   constraint.
10. **No removal of existing visibility paths.** The free-form
    cohort.log lines stay; widening the data on disk is
    additive. A future plan can re-shape the log lines if
    desired.

## Out of scope

- Multi-GPU coordination / distributed cohort runs (Phase-5
  question).
- Frontend UI changes — the v1 UI continues to read what it
  reads. A new visualisation pass is a separate plan.
- Websocket emission protocol changes
  (`training_v2/cohort/events.py` is touched only by the optional
  Session 03; default Sessions 01/02 leave it alone).
- Persistent cross-cohort dashboards / time-series reporting
  across multiple registry directories.
- 66-agent scale-up.
- v1 deletion.
- Reward-shape iteration.
- BC pretrain.
- Throughput / speed work (`throughput-fix` owns that).
- Mechanics changes to force-close, stop-close, target-pnl, or
  any other reward path.

## Phase-3-cohort hand-offs

From `plans/rewrite/phase-3-cohort/findings.md`:

1. **CUDA↔CUDA self-parity test** (`tests/test_v2_gpu_parity.py`)
   is the load-bearing correctness foundation. Every change in
   this plan must keep it passing.
2. **`select_days(seed=42)` order is deterministic** per agent.
3. **AMBER v2 baseline cohort dir:**
   `registry/v2_amber_v2_baseline_1777577990/`. Comparison floor.

From `plans/rewrite/phase-3-followups/force-close-architecture/`:

4. **Session 02 cohort dir** (running 2026-05-02):
   `registry/v2_force_close_arch_session02_stop_close_1777718273/`.
   First post-plan target for Session 02's peek tool.
5. **`EvalSummary` already carries the new fields** as of
   commit 3ba8e05. The plumbing gap is from `EvalSummary` →
   SQLite, not from env → `EvalSummary`.

## Useful pointers

- Scoreboard write site:
  [`training_v2/cohort/runner.py:262-272`](../../../../training_v2/cohort/runner.py)
- ModelStore + evaluation_days schema:
  [`registry/model_store.py`](../../../../registry/model_store.py)
- EvalSummary definition:
  [`training_v2/cohort/worker.py`](../../../../training_v2/cohort/worker.py)
  search for `class EvalSummary`.
- `_agent_result_to_scoreboard_row`:
  [`training_v2/cohort/runner.py`](../../../../training_v2/cohort/runner.py)
  search for the helper name.
- Throwaway peek script (precursor to the canonical tool):
  `C:/tmp/peek_session02.py`.
- Existing v2 cohort tests:
  [`tests/test_v2_cohort_runner.py`](../../../../tests/test_v2_cohort_runner.py),
  [`tests/test_v2_cohort_worker.py`](../../../../tests/test_v2_cohort_worker.py).
- Existing model-store tests:
  [`tests/test_model_store.py`](../../../../tests/test_model_store.py).

## Estimate

Per session:

- Session 01: ~1.5 h (30 min schema migration + tests, 30 min
  scoreboard-write move + tests, 30 min CLAUDE.md note + repo
  spot-check on readers).
- Session 02: ~1 h (peek tool + tests). Most of the work is
  schema-aware metric computation; the I/O is trivial.
- Session 03: ~2 h IF stretched (per-episode JSONL hook + tests
  + cohort-wall regression check).

Best case: ~2.5 h (Sessions 01+02). Stretch: ~4.5 h
(Sessions 01+02+03).

If past 3 h on Session 01 excluding tests, stop and check scope
— the change is small; if it's taking longer there's a hidden
schema dependency that needs unbundling into its own follow-on.
