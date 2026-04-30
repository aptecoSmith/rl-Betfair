# Session prompt — Phase 3, Session 04: frontend wiring + first real cohort

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Take the v2 cohort scaffolding from Session 03 and wire it through
the existing frontend's websocket schema. Run the **first real 12-
agent cohort** on 7 training days + 1 held-out eval day. Compare to
v1 cohort-M and write the Phase 3 verdict.

**This is the rewrite's payoff session.** Either v2 beats v1 on the
three architectural metrics that motivated the rewrite, or it
doesn't and the finding is "the architecture didn't pay."

End-of-session bar:

1. The existing UI (no code changes) renders v2 cohort progress live during the run.
2. 12 agents × 7 days cohort completes successfully and writes the registry.
3. Phase 3 success bar 6 (force-close < 50 %, ρ ≤ −0.5, ≥ 1 agent positive raw P&L) gets a verdict — PASS, AMBER, or FAIL.

## What you need to read first

1. `plans/rewrite/phase-3-cohort/purpose.md` — success bar 6, hard
   constraints.
2. `plans/rewrite/phase-3-cohort/session_prompts/03_cohort_scaffolding.md`
   — the cohort scaffolding this session extends.
3. `frontend/src/` (specifically `frontend/src/services/` or
   wherever the websocket consumer lives) — read for the schema
   it expects, do NOT modify.
4. `agents/websocket_events.py` (or wherever v1 emits training
   events) — the event schema v2 must match. **Read, don't
   import.**
5. `plans/rewrite/README.md` §"Success bar (Phase 3)" — the three
   architectural metrics.
6. v1 cohort-M scoreboard (most recent run in `registry/`) —
   produce the comparison table from this baseline.

## What to do

### 1. `training_v2/cohort/events.py` — websocket event adapter (~60 min)

Translate v2 trainer / worker events into v1's websocket schema.
The schema (read v1 to confirm field names — do NOT guess):

- `cohort_started` — n_agents, n_generations, seed, day list.
- `agent_training_started` — agent_id, gene dict, generation.
- `episode_complete` — agent_id, day_str, episode_idx,
  total_reward, day_pnl, value_loss_mean, approx_kl_mean,
  action_histogram, force_close_count, …
- `agent_training_complete` — agent_id, eval_summary.
- `cohort_complete` — final scoreboard.

The event emitter is plumbed through `worker.train_one_agent` and
`runner.run_cohort` as an optional callback parameter (None →
silent run for tests; a real callback → live websocket events).

**Adapter, not redesign.** v2's data has fields v1 doesn't (e.g.
per-runner advantage stats from Phase 2 Session 03's
`EpisodeStats` extension). The adapter drops these — they don't
show up in the UI. Adding a UI-side field is a follow-on plan
(rewrite hard constraint §"frontend stays as-is").

### 2. Wire the event emitter into the existing dispatch (~30 min)

v1's `training/run_training.py` calls a websocket emitter at known
points. v2's runner.py needs the same call sites:

- Before generation 0: `cohort_started`.
- Before each agent: `agent_training_started`.
- After each (day, episode): `episode_complete`.
- After eval: `agent_training_complete`.
- After all generations: `cohort_complete`.

The websocket transport itself is **shared with v1** — same
endpoint, same connection. Phase 3's hard constraint §3 says no
frontend code changes; the adapter writes to the same socket v1
writes to.

### 3. Live-test the wiring (~30 min)

Start the frontend dev server + the websocket relay (refer to
`CLAUDE.md` operator notes for the exact commands). Run a tiny
2-agent / 1-day cohort:

```
python -m training_v2.cohort.runner \
    --n-agents 2 \
    --generations 1 \
    --days 1 \
    --device cuda \
    --seed 42 \
    --output-dir registry/v2_uitest_$(date +%s) \
    --emit-websocket
```

Verify in the browser:

- The cohort scoreboard updates as agents finish.
- The per-episode chart streams reward / pnl / value_loss.
- The training-curves panel renders.

If anything renders wrong, **the bug is in the adapter, not the
UI**. The UI works for v1; if v2 events break it, the schema is
wrong.

### 4. The 12-agent cohort (~60 min runtime + 30 min setup)

```
python -m training_v2.cohort.runner \
    --n-agents 12 \
    --generations 4 \
    --days 7 \
    --device cuda \
    --seed 42 \
    --output-dir registry/v2_first_cohort_$(date +%s) \
    --emit-websocket \
    --eval-day 2026-04-25
```

Expected wall: 12 agents × 7 days × ~15 s/episode (GPU) × 4
generations = ~84 minutes. Plus eval × 12 × 4 = trivial.

Watch the live UI during the run. If anything spikes wrong (KL
ballooning, value loss exploding, force-close rate climbing), let
the run finish and triage — early termination is operator's call.

### 5. Compare to v1 cohort-M (~60 min)

The Phase 3 success bar table:

| Metric | v1 cohort-M | v2 first cohort | Verdict |
|---|---|---|---|
| Mean force-close rate | ~75 % | ?? | < 50 % = PASS |
| ρ(open_cost-equiv gene, fc_rate) | ~0 | ?? | ≤ −0.5 = PASS |
| Agents positive raw P&L (held-out day) | 0–7/66 | ?? / 12 | ≥ 1/12 = PASS |

For v2 there's no `open_cost` gene (Phase 3 schema doesn't have
it). The "open_cost-equivalent" metric is one of:

- ρ(`entropy_coeff`, fc_rate) — high entropy → more random opens
  → maybe more force-closes.
- ρ(`learning_rate`, fc_rate) — fast learners commit to closes
  earlier?

Pick the one with the cleanest theoretical link to "this gene
should reduce force-close rate" and document it in findings.md.
If neither has theoretical pull, the v2 cohort's evidence on this
metric is weak — flag as AMBER and propose a follow-on plan to
add an open-selectivity gene to the schema.

### 6. Findings writeup (~60 min)

`plans/rewrite/phase-3-cohort/findings.md` (extend the file from
Sessions 01–03):

- Success bar table (1–6 from purpose.md, PASS/FAIL/AMBER).
- Cohort comparison table (the three architectural metrics).
- Per-agent eval-day P&L distribution (histogram).
- Force-close rate distribution.
- Anything surprising during the run.
- Phase 3 verdict: PASS / AMBER / FAIL.
- If FAIL: scope-back proposal. Revert to v1 baseline? Iterate
  inside v2? Step further back to phase −1?
- If AMBER: gap-list for the follow-on plan.
- If PASS: green-light to retire v1 (per rewrite README "after
  Phase 3 success per rewrite README").

The findings.md is the rewrite's deliverable. **It's a one-page
table + 5 short observations + a verdict, not a long analysis.**
Phase-3-followups own the deep analysis if any verdict is below
PASS.

### 7. Tests (~30 min)

`tests/test_v2_websocket_events.py`:

- The adapter produces a valid `cohort_started` event with the
  right field names + types.
- A round-trip `episode_complete` → JSON → parse matches.
- A `cohort_complete` event closes the run cleanly.

Schema validation: read v1's `agents/websocket_events.py` (or
wherever the canonical schema lives), build a fixture, assert
v2's adapter produces matching shape.

## Stop conditions

- 2-agent UI test breaks the UI → **stop**. Adapter schema bug.
  Fix before launching the 12-agent cohort.
- 12-agent cohort crashes mid-run → **stop**, triage, restart
  with the fix. The cohort takes ~90 min; failing at agent 11 is
  expensive but salvageable if the bug is mechanical (e.g. GPU
  OOM at agent N — `torch.cuda.empty_cache()` between agents).
- Phase 3 success bar 6 FAILS on all three metrics → **stop and
  write the FAIL verdict honestly**. Do NOT iterate genes or
  reward shaping (rewrite hard constraint §5). The finding is
  "the architecture didn't pay" and Phase 3-followups own the
  next step.
- Phase 3 success bar 6 mixed (1–2 PASS, 1–2 FAIL) → **AMBER**.
  Document which metrics PASS and which FAIL. Propose follow-on
  plan.

## Hard constraints

- **No frontend code changes.** All v2-side. The UI works for v1
  and must keep working for v2 without modification.
- **No env edits.**
- **No reward shaping additions.** If P&L is poor, that's a
  finding, not a fix.
- **No gene schema changes.** Phase 3's locked schema is what the
  cohort runs on.
- **No retroactive change to v1 cohort-M.** Compare against the
  most-recent v1 cohort-M baseline as-is. If v1 needs a re-run for
  comparison, file as a follow-on; do NOT bundle.
- **Phase 3 verdict is honest.** PASS only if all six bars pass.
  AMBER if any softer; FAIL if architectural bars (force-close,
  P&L) miss. Don't grade on a curve.

## Out of scope

- v1 deletion (gated on PASS verdict; a separate follow-on plan
  owns the deletion).
- 66-agent cohort (Phase 3 caps at 12; follow-on for scale-up).
- Multi-GPU (follow-on).
- New gene additions (locked at Phase 3).
- BC pretrain (rewrite removes it).
- UI redesign for v2-specific metrics (follow-on).
- Curriculum day ordering (follow-on if multi-day flat ordering
  shows order-dependence).

## Useful pointers

- v1 websocket emitter: search `agents/`, `training/`, `frontend/`
  for "websocket" or "ws" or "events".
- v1 cohort-M scoreboard: `registry/scoreboard.jsonl` (most-
  recent rows where `arch_name` matches v1's pattern).
- Phase 3 cohort scaffolding: `training_v2/cohort/{worker,runner,
  genes}.py` (Session 03 deliverables).
- Phase 3 multi-day train: `training_v2/discrete_ppo/train.py`
  (Session 02 deliverable).
- Phase 3 GPU pathway: `training_v2/discrete_ppo/trainer.py`
  (Session 01 deliverable).
- Frontend: don't touch.

## Estimate

5 hours.

- 60 min: events.py adapter.
- 30 min: wire emitter into runner.
- 30 min: 2-agent UI test.
- 90 min: 12-agent cohort runtime (background while writing).
- 60 min: comparison table + findings.
- 30 min: tests.

If past 7 hours, stop and check scope. The most likely overrun is
the 12-agent cohort itself if GPU throughput is lower than
expected. If the cohort doesn't fit in a session, run it in
background overnight and resume the comparison + findings the
next day.

This is the last session of Phase 3. After this:

- **PASS verdict** → green-light to retire v1 (separate follow-on
  plan).
- **AMBER verdict** → follow-on plan addresses the gap.
- **FAIL verdict** → step back. Probably: iterate inside v2 with
  a clearly-articulated hypothesis, OR revert to v1 baseline and
  rethink the rewrite premise.
