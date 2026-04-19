# Arb Curriculum — Session 01 prompt

Current session: **Session 01 — Offline arb oracle scan +
per-day density metric**.

Detailed brief:
[`session_prompts/01_oracle_scan.md`](session_prompts/01_oracle_scan.md).

Before starting, read:

- [`purpose.md`](purpose.md) — structural diagnosis from
  `reward-densification-probe` 2026-04-19, design sketch
  for the four coordinated interventions, success criteria.
- [`hard_constraints.md`](hard_constraints.md) — 40
  non-negotiables. §6–§9 (oracle semantics), §24–§25
  (telemetry + invariant), §26–§32 (testing).
- [`master_todo.md`](master_todo.md) — seven-session scope
  and per-session exit criteria.
- `plans/arb-improvements/session_6_oracle_scan.md` — the
  2026-04-14 scoping for this session. Most of it carries
  over; deltas are captured inline in the Session 01
  prompt.
- `plans/arb-improvements/hard_constraints.md` — the
  original hard constraints on oracle output; everything
  about "offline-only", "targets must be env-reachable",
  "output is deterministic" carries forward unchanged.
- `env/exchange_matcher.py` — the filter predicates the
  oracle mirrors.
- `env/scalping_math.py` —
  `locked_pnl_per_unit_stake(P_back, P_lay, commission)`.
  The oracle's profitability check calls this directly;
  no re-implementation.
- `CLAUDE.md` "Order matching: single-price, no walking".
- `plans/reward-densification/progress.md` 2026-04-19
  entry — the failure evidence motivating this whole plan.
