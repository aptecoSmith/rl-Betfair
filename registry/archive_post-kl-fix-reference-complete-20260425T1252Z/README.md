# post-kl-fix-reference, BOTH gens complete (2026-04-25 12:52Z)

Plan: `post-kl-fix-reference` (`dcb97886…`).
Run IDs: `9bbe0aeb` (gen 0) → `9bc12e09` (gen 1).
Wall time gen 0: 8474s. Wall time gen 1: 7475s. Total ~4h25m.

## What this run validated

**The Session-02 KL fix + threshold bump + state_dict drift fix
all worked.** First clean two-generation run since the plan
landed.

| metric | this run | pre-fix gen-1 | post-Session-01 partial | last attempt (Session-02 first try) |
|---|---|---|---|---|
| KL median (per mini-batch) | 0.23 | 12,740 | 3-20 | 0.043 |
| KL early-stop threshold | 0.15 | 0.03 | 0.03 | 0.03 |
| mini-batches per PPO update (median) | 15 | 1 | 1 | 5 |
| mini-batches per PPO update (max) | 724 | 1 | 1 | 13 |
| State_dict drift on survivor load | 12 detected, 0 crashed | n/a | 8/12 crashed | 8/12 crashed |
| Agents trained gen-0 | 12 | 50+ | 12 | 12 |
| Agents trained gen-1 | 12 | 42 | n/a | 4 (post-crash collapse) |
| Best fitness gen-0 | 0.34 | varied | 0.36 (stuck) | 0.36 |
| Best fitness gen-1 | **0.46** | varied | n/a | 0.36 (no change) |
| Top-1 PnL | **+£17.45** | varied | +£2.51 | +£2.51 |
| Top-1 architecture | ppo_time_lstm_v1 | mixed | ppo_time_lstm_v1 | ppo_time_lstm_v1 |

## What this run revealed about the agent

**Force-close rate stayed at 76 % across all three architectures
despite PPO actually training.** The threshold bump produced
real PPO updates (median 15 mini-batches per update, occasional
runs of 700+) and gen-1 produced a meaningfully better top
agent. But the agent's selectivity didn't move.

| metric | per race |
|---|---|
| pairs_opened | ~620–650 |
| matured | 58–65 |
| closed (agent-initiated) | 16–22 |
| naked | 69–74 |
| force-closed | 471–495 |
| force-close rate | **76 %** |
| `scalping_force_closed_pnl` | −£385 to −£396 |
| total_pnl mean | −£343 to −£381 |

The fact that one agent reached +£17 PnL despite the 76% force-
close rate is striking — selection is finding good policies by
accident. Direct shaping pressure to be selective is needed.

**`plans/selective-open-shaping/` IS justified.** Promotes
straight to Session 02 (gene-sweep probe).

## Bookkeeping anomaly (now fixed)

Plan finalised at status="failed" despite both generations
completing successfully. Root cause: `_check_dead_thread` race
during the inter-session gap (data loading for session N+1
takes 4+ seconds while the session-N training thread is dead).
The grace-poll mechanism couldn't span the gap, so the dead-
thread handler ran `set_status("failed")` mid-run. Eventual
run_complete couldn't recover because the BAD set_status fired
AFTER the auto-continue had already started session N+1.

Fix landed in commit alongside this archive. Future runs of the
plan won't see this issue.

## Contents

- `models.db` — 128 KB, 12 models' training + evaluation records
  + the plan outcomes.
- `weights/` — 12 `.pt` files totalling ~48 MB. Models from
  gen-1 (the survivors-plus-children that ended the run).

## Don't reuse the weights

Different gradient regime if/when `selective-open-shaping`
lands a new probe with `open_cost > 0`. Cross-loading would
create a confused starting state. Treat these as historical
reference only.
