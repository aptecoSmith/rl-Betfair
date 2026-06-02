# SESSION HANDOFF — rl-betfair recipe-expansion campaign

> You're picking up an autonomous-experiment campaign on this codebase.
> Read this file end-to-end before doing anything. Do NOT relitigate
> the locked-in methodology lessons — they cost hours of GPU to learn
> and are documented for a reason.

## Read these first, in order

1. **`plans/recipe-expansion-and-robustness/findings.md`** — canonical
   campaign state: held-out leaderboard, what works/doesn't, per-open
   economics framework, recommended next steps. This is the most
   important doc.
2. **`plans/recipe-expansion-and-robustness/autonomous_loop_v2.md`** —
   standing operating instructions (methodology rules, levers to try,
   launch reliability, stop conditions).
3. **`plans/recipe-expansion-and-robustness/monitoring_notes.md`** —
   chronological log of decisions (skim recent entries for context).
4. **`plans/EXPERIMENTS.md`** — campaign-wide chronological digest
   (the most recent block from 2026-05-27 onward).
5. **`plans/EXPLORATIONS.md`** — strategic analysis essays. The
   2026-05-28 entry "CORRECTION — naked-EV was eval-window
   overfitting" and the 2026-05-30 entry "Per-open economics —
   why selectivity alone can't reach profitability" are the most
   important.
6. **`plans/recipe-expansion-and-robustness/BREAKTHROUGH.md`** — marked
   SUPERSEDED. Read the warning header. Do NOT redo fc=0 sweeps.

## TL;DR of campaign state (as of 2026-05-30 ~07:30 BST)

- **Goal:** find a deployable true-scalping recipe (agent opens pairs
  that mature, both legs fill, lock spread profit). Evaluated on
  HELD-OUT days only.
- **Best held-out so far:** N4 — full-aug + pwin BAND 0.20–0.50,
  day_pnl **-£78** on 7 unseen days. NOT YET POSITIVE.
- **Mechanism is sound but underpowered:** each matured scalp locks
  +£3–6 after commission. Bottleneck = mat% (only 4–7%, need ~30%).
- **Per-open economics is the central framework**: every recipe at
  the current op-point is **-£1/open net** (locked-side +£0.13 vs
  fc-side −£1.20). Need to flip this sign, not just open fewer pairs.

## Locked-in methodology lessons (DO NOT RELITIGATE)

1. **ALWAYS eval on held-out days, NEVER train on them.** Train days
   = 2026-04-06/08/09. Iteration-eval = 7 odd-dated May days
   (2026-05-07,09,11,13,15,17,19). Final-test = 7 even days
   (2026-05-08,10,12,14,16,18,20), reserved, looked at ONCE at the
   end.
2. **Naked P&L is zero-EV directional variance.** Looked like a
   +£287 edge in-sample, collapsed to -£175 on held-out. Don't be
   tempted again.
3. **`force_close_before_off_seconds=120` (or shorter) ON.** It's
   a safety rail bounding naked variance, not a cost to remove.
4. **Select on held-out LOCKED P&L + mat%, NOT total day_pnl.**
   Total day_pnl on a short window is naked-noise dominated.
5. **Document per-open arithmetic** (mat%, locked/pair,
   fc-cost/pair) — not just day_pnl. That's the unit that defines
   deployability.
6. **Bash polling chains are the durable autonomy mechanism** on
   this Windows + git-bash + Claude Code env. ScheduleWakeup and
   nohup/Popen "supervisor" daemons all failed. Use:
   `nohup bash <wrapper> > /dev/null 2>&1 & disown` from your
   own Bash tool, with chain wrappers (poll for upstream
   completion, then run next round).
7. **Don't celebrate "less negative" as "almost there."** Recipe is
   deployable iff held-out per-open ≥ 0.

## What's running and queued (verify state first via `tail registry/_*_wrapper.log`)

When you start: GPU state was these chains running. **First check
the actual current state** — these may have completed, failed, or
need restarting.

| order | round | direction | status to verify |
|---|---|---|---|
| Q | tight spread + band cap variants (4 cells) | `registry/_roundQ_wrapper.log` should show "roundQ fan-out complete" |
| R (chain Q→R) | band-lever push (7 cells: bands 0.15-0.45, 0.20-0.45, 0.15-0.50, band+fc60, N4 seed replicates) | `registry/_roundR_wrapper.log` |
| S (chain R→S) | **mat%-LIFT path** (8 cells: extreme tight_lock 0.0005-0.0001, fc=30/15, stacked) — explicitly tests Path B | `registry/_roundS_wrapper.log` |

Chain scripts: `plans/recipe-expansion-and-robustness/run_chain_Q_R.sh`,
`run_chain_R_S.sh`. If they died, restart manually.

## Immediate priorities (in order)

1. **Verify GPU is busy and Q/R/S chain is alive.**
   `ps -ef | grep cohort.runner | grep -v grep | head` — should show
   one cohort training. If idle, the chain broke; restart whichever
   round didn't run via `nohup bash <wrapper> > /dev/null 2>&1 &
   disown`.

2. **When Q/R/S complete, analyse held-out results per round.** Pull
   metrics with the python+json pattern (see EXPERIMENTS.md
   examples). Focus on **mat%, locked/pair, fc-cost/pair**, then
   day_pnl. Append a dated entry to monitoring_notes.md and update
   findings.md's leaderboard if anything beat -£78.

3. **Build the mature_prob open-gate (Path C).** This is the
   highest-leverage untried mechanism. Implementation sketch:
   - Add a policy-side mask layer in `agents_v2/discrete_policy.py`
     that, at forward-pass time, zeros OPEN_BACK/OPEN_LAY logits
     per-runner where the policy's own `mature_prob_head` output is
     below a threshold.
   - Add `--mature-prob-open-threshold FLOAT` CLI flag in
     `training_v2/cohort/runner.py`, plumbed through worker.py.
   - Mirror the existing `predictor_p_win_back_threshold` plumbing
     pattern (it's at known wire points — grep the codebase).
   - Estimate ~1h work. The pwin_back gate code is the template.
   - Test with a held-out probe (Round T): threshold sweep e.g.
     0.30/0.40/0.50/0.60, on top of the N4 base recipe.

4. **If the mature_prob gate doesn't lift held-out per-open ≥ 0:**
   write a follow-up findings note + suggest the next direction
   (Path D liquidity gate, scale-up cohort, or a structural rethink
   per findings.md "Open questions").

5. **For ANY candidate that hits held-out per-open ≥ 0:** confirm on
   the final-test set (May even days 8-20, never touched) before
   declaring deployable.

## Recipe templates (reference)

The base "best so far" recipe (N4, held-out -£78):

```
--n-agents 4
--generations 1
--device cuda
--seed 42
--strategy-mode arb
--training-days-explicit 2026-04-06 2026-04-08 2026-04-09
--cohort-eval-days 2026-05-07 2026-05-09 2026-05-11 2026-05-13 2026-05-15 2026-05-17 2026-05-19
--rotating-eval-sample 0
--direction-head-manifest models/direction_head/sweep_c11
--predictor-lean-obs
--use-race-outcome-predictor
--use-direction-predictor
--predictor-bundle-manifests <champion> <ranker> <direction>
--reward-overrides force_close_before_off_seconds=120
--reward-overrides close_feasibility_max_spread_pct=0.05
--reward-overrides matured_arb_expected_random=0.0
--bc-pretrain-steps 500
--bc-include-negative-samples
--bc-include-close-hold-samples
--predictor-p-win-back-threshold 0.20
--predictor-p-win-back-max-threshold 0.50
```

Predictor manifest paths:
- `C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json`
- `C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json`
- `C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json`

## Useful helper scripts

- `tools/launch_detached.py` — Python wrapper for `subprocess.Popen`
  with `CREATE_BREAKAWAY_FROM_JOB` for true Windows detachment.
  Useful when the standard `nohup ... & disown` from your Bash tool
  isn't enough. Note: pass `env=os.environ.copy()` AND ensure the
  child has python on PATH (we hit this — use full
  `/c/Python314/python.exe` or `export PATH=/c/Python314:$PATH` at
  top of wrappers).
- `tools/babysit_loop.py` — DEPRECATED. Don't use. Failed multiple
  times under Task Scheduler job-object semantics.
- `tools/gpu_supervisor.sh` — written but unreliable (bash env
  issues). Use direct chain wrappers instead.

## Cohort runner basics

Every cell:
```bash
nohup bash <wrapper>.sh > /dev/null 2>&1 &
disown
```

Each round wrapper has a `run_cell` function that calls
`python -m training_v2.cohort.runner` with the recipe flags. The
output goes to `registry/_<round>_<cell>_<ts>/` with scoreboard.jsonl
that contains the per-agent eval metrics.

## Update these docs as you go

- `monitoring_notes.md` — every analysis decision (append-only)
- `findings.md` — update leaderboard + recommendations when results change
- `plans/EXPERIMENTS.md` — append new rounds with intention/result
- `plans/EXPLORATIONS.md` — append a new entry when a strategic
  insight emerges (don't just dump data — synthesize)

## When to stop

- A recipe achieves held-out per-open ≥ 0 across 3+ seeds → document
  candidate, run final-test on May even days, tell the operator.
- All four paths (selectivity, mat-lift, mature_prob gate, liquidity
  gate) tested with no per-open ≥ 0 → write a "recipe space
  exhausted at probe scale" findings summary, recommend scale-up
  (12-24 agents, multi-gen) or architectural redesign, tell the
  operator.
- Otherwise: keep looping. Run rounds, analyse, design, queue, repeat.
  Update docs each cycle.
