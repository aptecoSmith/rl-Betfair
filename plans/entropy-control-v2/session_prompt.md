# Entropy-Control-v2 — Session 01 prompt

Current session: **Session 01 — Target-entropy controller
(learned log_alpha)**.

Detailed brief:
[`session_prompts/01_target_entropy_controller.md`](session_prompts/01_target_entropy_controller.md).

Before starting, read:

- [`purpose.md`](purpose.md) — Baseline-A entropy drift
  evidence (139.6 → 201.3 across 64 agents × 15 episodes),
  the diagnosis (fixed-coefficient vs sparse policy
  gradient), and the controller design sketch.
- [`hard_constraints.md`](hard_constraints.md) — 26
  non-negotiables. §4–§10 (controller semantics), §11–§12
  (inherited agents and checkpoint format), §18–§19
  (testing), §21 (NOT a reward-scale change).
- [`master_todo.md`](master_todo.md) — three-session scope
  and per-session exit criteria.
- `CLAUDE.md` — "PPO update stability — advantage
  normalisation" section; this plan adds a new "Entropy
  control" paragraph downstream of that one.
- `plans/naked-clip-and-stability/progress.md` — the
  2026-04-19 Validation entry with the A-baseline drift
  table and criterion scorecard.
- `plans/naked-clip-and-stability/lessons_learnt.md` — the
  2026-04-19 entry on endpoint-vs-slope test design (Session
  02 of this plan implements the fix).
- `agents/ppo_trainer.py` — the file being edited. Locate
  the `entropy_coeff` / `entropy_coefficient` references
  and the existing `_entropy_coeff_base` scaffolding
  (arb-improvements Session 2) — the controller replaces
  the latter.
