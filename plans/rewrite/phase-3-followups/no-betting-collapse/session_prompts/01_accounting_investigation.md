# Session prompt — no-betting-collapse Session 01: locked/naked accounting investigation

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Phase 3's first 12-agent cohort produced an AMBER verdict
(`plans/rewrite/phase-3-cohort/findings.md` Session 04 live-run
results). Two of three Bar-6 metrics PASSED; **Bar 6c failed: 0/12
agents positive on raw P&L** on the held-out eval day. Every agent
showed `eval_day_pnl = £0.00` exactly, with `eval_locked_pnl =
+X` and `eval_naked_pnl = −X` for matching X (per-agent range
+£3 to +£17). The accounting nets to zero by construction; the
question is whether that's a bug or a correct-by-design property
of the matured-arb cash flow.

**Until this verdict lands, the shaping ablations in Session 02
have no baseline to measure against.** A bug fix here might flip
Bar 6c on its own without any reward-shape change.

End-of-session bar:

1. The cash flow at race-settle for one matured arb pair is traced through the env code path with line references.
2. A unit test (`tests/test_v2_eval_pnl_accounting.py`) builds a synthetic 1-race day, opens one matured pair, and asserts the EXPECTED relationship between `info["day_pnl"]`, `info["locked_pnl"]`, and `info["naked_pnl"]`.
3. Verdict logged as either:
   - **(a) BUG**: minimal fix committed; AMBER baseline cohort re-runs with the fix; new Bar 6c result documented.
   - **(b) BY-DESIGN**: docs change committed (`CLAUDE.md` "Reward function" section + this plan's findings.md) restating Bar 6c interpretation; **no env edit**.

## What you need to read first

1. `plans/rewrite/phase-3-followups/no-betting-collapse/purpose.md`
   — this plan's purpose + constraints + verdict bar.
2. `plans/rewrite/phase-3-cohort/findings.md` Session 04 (last
   section, "Live-run results 2026-04-29 → 2026-04-30") — the
   raw observation table and the diagnostic that motivated this
   investigation. Pay particular attention to the
   `locked_pnl + naked_pnl = 0.00` table — that's the load-bearing
   observation.
3. `CLAUDE.md` "Reward function: raw vs shaped" — the spec for
   how `race_pnl` decomposes into `scalping_locked_pnl +
   scalping_closed_pnl + scaled_naked_sum`.
4. `CLAUDE.md` "Equal-profit pair sizing (scalping)" — the
   spec that says equal-profit pairs lock in `~£X` per pair
   (NOT zero) on either race outcome. This is the load-bearing
   contradiction with the observed zero.
5. `env/betfair_env.py::_settle_current_race` — the function
   that computes `race_pnl`. Trace each component term back to
   its source.
6. `env/bet_manager.py` — the per-pair tracking. Look for
   `locked_pnl`, `naked_pnl`, and any per-pair aggregation.
7. `training_v2/cohort/worker.py::_eval_rollout_stats` — how
   the eval reads `info["day_pnl"]` etc.
8. `registry/v2_first_cohort_1777499178/scoreboard.jsonl` —
   the AMBER baseline data. Re-load and verify the pattern:

   ```python
   import json
   rows = [json.loads(l) for l in
       open("registry/v2_first_cohort_1777499178/scoreboard.jsonl")
       .read().splitlines() if l.strip()]
   for r in rows:
       print(r["agent_id"][:12], r["eval_locked_pnl"],
             r["eval_naked_pnl"], r["eval_day_pnl"])
   ```

   Confirm `locked + naked = 0.00` exactly for all 12 agents
   before assuming the prior session's report.

## What to do

### 1. Trace `day_pnl` end-to-end (~45 min)

Start at the call site:

```python
# training_v2/cohort/worker.py::_eval_rollout_stats
day_pnl = float(last_info.get("day_pnl", 0.0))
```

Trace `info["day_pnl"]` back to where the env populates it
(probably `BetfairEnv.step` at the terminal step). Then trace
each summand of `race_pnl` (per CLAUDE.md):

- `scalping_locked_pnl` — where computed, sourced from which
  per-pair attribute on `Bet`?
- `scalping_closed_pnl` — same.
- `scaled_naked_sum` — same.

For each, confirm:
- The per-pair sum at race level matches the sum at day level.
- The `info["locked_pnl"]` / `info["naked_pnl"]` exposed on the
  step's info dict come from the SAME per-pair source, or from
  a separate (possibly wrong) accumulator.

The hypothesis to test: `info["locked_pnl"]` and
`info["naked_pnl"]` are accumulated from different points in
the per-pair lifecycle, with the matured-pair entry showing up
in BOTH (positive in locked, negative in naked, exactly
cancelling).

### 2. Synthetic regression test (~45 min)

Write `tests/test_v2_eval_pnl_accounting.py`:

```python
def test_one_matured_pair_produces_nonzero_day_pnl():
    """A single matured arb pair should report:

    - day_pnl ≈ +£X (the equal-profit lock value, ≥ £0.01)
    - locked_pnl ≈ +£X
    - naked_pnl == 0.0  (no UNPAIRED naked outcome)
    - day_pnl == locked_pnl + naked_pnl + closed_pnl
    """
    ...
```

Use a hand-crafted minimal `BetfairEnv` setup (one race, two
runners, scripted ladder). Open one back leg, then on the next
step open a passive lay leg at the equal-profit price; confirm
both fill (matched). Step until race settles. Assert the four
relationships above.

If the test FAILS as written (i.e. `day_pnl == 0` despite a
matured pair) — that's the bug, in stark form, in CI. Use this
as your reproducer.

If the test PASSES as written — then the cohort's pattern has
a different source. Tighten the test until you can reproduce
the cohort's `locked + naked = 0.00` pattern in unit form.

### 3. Verdict (a) — BUG path (~60 min)

If the trace + test confirm an accounting bug:

- Find the smallest minimum-impact fix. Preferably a one-line
  change in the env's per-pair tracking (e.g. "matured pair
  should NOT also accumulate to naked").
- Add a regression test that pins the FIXED relationship.
- Run the existing v2 cohort tests (`tests/test_v2_cohort_*.py`,
  `tests/test_v2_websocket_events.py`) — they should still pass
  (the fix only changes accounting under matured-pair conditions).
- **Re-run the AMBER baseline cohort** with the fix:

  ```
  python -m training_v2.cohort.runner \
      --n-agents 12 --generations 1 --days 8 \
      --device cuda --seed 42 \
      --output-dir registry/v2_amber_rerun_$(date +%s)
  ```

  ~3.1 h GPU. Watch for crashes. When done, re-run the analysis:

  ```
  python C:/tmp/v2_phase3_bar6.py registry/v2_amber_rerun_<ts>
  ```

  If Bar 6c flips to ≥ 1/12 → the bug fix on its own recovers
  the rewrite. **Plan ships GREEN; Session 02 + 03 become
  documentation only.**

  If Bar 6c stays 0/12 even with non-zero `day_pnl` readings,
  the accounting bug was real but cosmetic — proceed to
  Session 02 to test shaping ablations.

### 4. Verdict (b) — BY-DESIGN path (~30 min)

If the trace + test confirm the zero is correct (e.g.
matured-arb cash flow is by-design zero under the equal-profit
lock; only `close_signal` produces realised cash):

- Document the trace in `CLAUDE.md` "Reward function" section
  (one paragraph, max).
- Update Bar 6c's interpretation in this plan's findings.md:
  "Bar 6c measures cash from CLOSE_SIGNAL events, not from
  matured arbs. Re-evaluate gen-0 agents on `eval_closed_pnl >
  0` not `eval_day_pnl > 0`."
- If the existing scoreboard rows have `eval_closed_pnl`
  populated, recompute Bar 6c on that field. If they don't, the
  re-run is needed.
- Either way, this changes the shape of Session 02's ablation
  (it'd test which shaping term incentivises `close_signal`
  use, not which one un-collapses betting).

## Stop conditions

- Cannot reproduce the cohort's `locked + naked = 0.00` pattern
  in a unit test → **stop**. Either the cohort's data is from a
  different code path than the env you're tracing, or there's a
  meta-bug in the eval rollout's `info` propagation. Triage that
  first.
- Trace finds the bug but the fix breaks 5+ existing tests →
  **stop**. The fix is not minimal-impact; design a smaller fix
  or escalate (this becomes a multi-session investigation).
- Re-run cohort takes > 5 hours → **stop, kill the cohort, file
  for the throughput-fix follow-on**. The 3.1-hour envelope is
  load-bearing; a regression in throughput here means something
  unrelated broke.

## Hard constraints

- **No env edits beyond the minimum-impact accounting fix.** No
  reward-shape changes. No new shaping terms. No new genes.
- **No reward-shape introductions.** Session 02 owns shaping
  ablations; Session 01 is accounting only.
- **Same `--seed 42` for the re-run.** Cross-cohort comparison
  is the load-bearing test.
- **Re-runs go to NEW output dirs.** Don't overwrite the AMBER
  baseline directory `registry/v2_first_cohort_1777499178/`.

## Out of scope

- Shaping ablations (Session 02).
- Throughput fix (separate plan).
- 4-generation runs (Phase 3 follow-on protocol locks at 1
  generation).
- New genes / schema changes.
- v1 comparison runs (the AMBER baseline IS the comparison).

## Useful pointers

- AMBER baseline scoreboard:
  `registry/v2_first_cohort_1777499178/scoreboard.jsonl`.
- Per-pair lifecycle tracking:
  `env/bet_manager.py` (BetManager class), `env/scalping_math.py`.
- Race-level reward decomposition:
  `env/betfair_env.py::_settle_current_race`.
- Eval rollout: `training_v2/cohort/worker.py::_eval_rollout_stats`.
- Bar 6 analysis tool: `C:/tmp/v2_phase3_bar6.py`.

## Estimate

3 hours.

- 45 min: trace `day_pnl` through the code.
- 45 min: synthetic regression test.
- 60 min: bug-fix path (if applicable).
- 30 min: by-design docs path (if applicable).
- + 3.1 hour cohort re-run if the fix lands.

If past 5 hours (excluding the cohort re-run wall time), stop
and check scope. The most likely overrun is the trace itself
— the env's per-pair accounting has accreted across multiple
plans (`scalping-naked-asymmetry`, `naked-clip-and-stability`,
`arb-curriculum`); pin the source-of-truth one piece at a time
rather than trying to load the whole thing.
