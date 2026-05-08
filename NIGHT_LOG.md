# Phase-15 overnight session log (autonomous)

Operator left at ~23:30 on 2026-05-08. Resumes ~17h later.

## Current status (snapshot at 23:53)

**v8 LANDED** (registry/_phase15_smoke_md_1778279309). BOTH agents
positive eval pnl:

| Agent | bets | matured | mat% | eval_pnl |
|---|---|---|---|---|
| 1 | 36 | 13 | 72% | **+£20.04** |
| 2 | 94 | 28 | ~30% | **+£39.80** |

This is the breakthrough. Phase-14 baseline was mean -£73; phase-15
v7 was -£3; v8 is +£30 mean.

## Auto-chain queued (background tasks)

| Task | Description | Expected completion |
|---|---|---|
| bygpgdm23 | v8 → v9 (auto-launch when v8 done) | running v9 |
| b3sen31lb | benchmark (24 loss×arch combos) when v9 done | ~00:30 |
| b5rb6oyfa | scan 04-29/04-30 + big run when benchmark done | ~06:30 |
| b02lnai0z | analyse big run + v10 (low-gate) when big run done | ~07:00 |
| blnrz7uj8 | v11 gate sweep (T=0.5/0.6/0.7) after v10 | ~09:00 |
| b2dcvnj0v | v12 force_close sweep (30/60/90s) after v11 | ~11:00 |

## v9 update (00:00)

V9 vanilla BCE result: agent NOOPs entirely.
- post_bc_dir_bce_back=0.4357 (probe-range calibration)
- BUT: head outputs natural distribution (~0.22 mean), gate at
  0.85 blocks everything → 0 bets → 0 pnl
- Confirms: pos_weight=true (v8 config) is correct for T=0.85.
  v10 tests vanilla BCE with T=0.55 (matched gate threshold).

## Files committed

- `agents_v2/discrete_policy.py`: phase-15 (LayerNorm + slice + detach)
- `training_v2/discrete_ppo/bc_pretrain.py`: extended BC pretrainer
  for direction_prob_head + pos_weight knob
- `training_v2/cohort/worker.py`: BC plumbing + freeze post-BC
- `env/betfair_env.py`: direction_bce_use_pos_weight whitelisted
- `tools/direction_loss_benchmark.py`: offline loss/arch benchmark
- `tools/phase15_summary.py`: smoke result summarizer
- `tools/phase15_compare.py`: multi-smoke comparison
- `scripts/phase15_big_run.sh`: 8 agents × 2 gens × 5 train + 3 eval
- `scripts/phase15_gate_sweep.sh`: T={0.7, 0.85, 0.95} sweep

## Decision log

- v8 used pos_weight=true (default). Worked. Big run uses same.
- v9 testing pos_weight=false (vanilla BCE). If v9 is better, may
  switch big run config (but big run already started by then).
- Big run uses --enable-gene direction_gate_threshold so GA evolves
  per-agent threshold in [0.5, 0.95].

## What to do when operator returns

1. Read this file.
2. Check `registry/_phase15_big_*.log` (latest match) for big run
   results.
3. Run `python -m tools.phase15_summary <log>` for quick agent table.
4. Look at scoreboard rows in `registry/_phase15_big_*/scoreboard.jsonl`
   for full eval metrics (pairs_opened, force_closed, locked_pnl).

## If something failed

- v9 fail (negative pnl): pos_weight=true was load-bearing; big run
  using same is fine.
- benchmark fail: just diagnostic; doesn't block big run.
- big run fail: check `registry/_phase15_big_*.log` for traceback.
  Most likely cause: oracle cache missing for a day.
