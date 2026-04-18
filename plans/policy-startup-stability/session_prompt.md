# Policy Startup Stability — Session prompt pointer

Current session: **Session 01 — Per-batch advantage
normalisation + tests**.
Detailed brief:
[`session_prompts/01_advantage_normalisation.md`](session_prompts/01_advantage_normalisation.md).

Before starting, read:

- [`purpose.md`](purpose.md) — root cause (PPO update collapses
  action heads on first rollout via large-magnitude rewards),
  the agent-`3e37822e-c9fa` evidence trajectory, the proposed
  fix (literature-standard per-batch advantage normalisation),
  and what success looks like.
- [`hard_constraints.md`](hard_constraints.md) — 21 non-
  negotiables. §1 (single conceptual change), §3 (no schema
  bumps), §4 (env untouched), §5–§7 (the math), §8 (one
  function in `agents/ppo_trainer.py`), §15–§17 (mandatory
  tests).
- [`master_todo.md`](master_todo.md) — two-session breakdown
  with deliverables, exit criteria, acceptance criteria for
  each.
- `CLAUDE.md` — "Reward function: raw vs shaped" (Session 02
  augments this).

Session order:

| # | File | What lands |
|---|---|---|
| 01 | [`01_advantage_normalisation.md`](session_prompts/01_advantage_normalisation.md) | Per-batch advantage normalisation in `agents/ppo_trainer.py`. Synthetic spike-prevention test. Smoke run with policy_loss < 100 on ep 1. Optional first-update LR warmup if normalisation alone is insufficient (decision rule in the brief). |
| 02 | [`02_docs_and_reset.md`](session_prompts/02_docs_and_reset.md) | CLAUDE.md note + four activation plans reset to draft. Docs + JSON edits only. |

Both sessions are designed to be runnable unattended end-to-end —
no manual decision points an agent can't resolve from the brief
alone.
