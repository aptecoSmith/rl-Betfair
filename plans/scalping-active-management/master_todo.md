# Master TODO — Scalping Active Management

Ordered session list. Tick boxes as sessions land. Each
session's full brief is in
[`session_prompts/`](./session_prompts/); the short summary
here is a nav aid, not the authoritative spec.

When a session completes:

1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.
4. Update [`session_prompt.md`](./session_prompt.md)'s
   "Next session" pointer to the following session.
5. Note cross-repo follow-ups in `ai-betfair/incoming/` per
   the postbox convention.

---

## Phase 1 — Active re-quote mechanic

- [x] **Session 01 — Re-quote action + env plumbing** (2026-04-16)

  Per-runner `requote_signal` action dim; cancels + re-places
  the open paired passive at a new tick offset computed from
  CURRENT LTP. Two new obs features
  (`seconds_since_passive_placed`,
  `passive_price_vs_current_ltp_ticks`). No archived prompt
  — see Session 01 in `progress.md` for what landed.

## Phase 2 — Self-awareness via auxiliary heads

- [x] **Session 02 — Fill-probability head** (2026-04-16)

  📄 Prompt: [`session_prompts/02_fill_prob_head.md`](./session_prompts/02_fill_prob_head.md)

  Supervised aux head predicting `P(passive fills before
  race-off | state)`. BCE loss at
  `fill_prob_loss_weight` (default `0.0` — plumbing-off).
  Prediction captured per-`Bet`, inherited onto the paired
  passive via `pair_id`, persisted through parquet.

- [ ] **Session 03 — Risk / predicted-variance head**

  📄 Prompt: [`session_prompts/03_risk_head.md`](./session_prompts/03_risk_head.md)

  Second aux head: per-runner Gaussian NLL on locked-P&L
  (mean + clamped log-variance). `risk_loss_weight` default
  `0.0`. Structurally parallel to Session 02; same
  capture→attach, same migration helper pattern.

---

### 🔌 Activation gate — `activation_playbook.md`

Sessions 02 and 03 land as **plumbing-off**
(`fill_prob_loss_weight = 0.0`, `risk_loss_weight = 0.0`).
Before the Phase 3 UI sessions below can show meaningful
confidence / risk badges, and before Session 07 can measure
the plan's net effect, the weights need to be turned up,
swept, and promoted.

See [`activation_playbook.md`](./activation_playbook.md) for
the full protocol: zero-weight sanity run → per-head weight
sweep → joint verification → promote into
`config/scalping_gen1.yaml`.

This is NOT numbered as a session because it's an operator
procedure, not a code change (Step E does commit a config-
value update). It must complete before Session 07's
validation run or Session 07 will be measuring the re-quote
mechanic in isolation.

---

## Phase 3 — UI

- [ ] **Session 04 — Bet Explorer confidence / risk badges**

  📄 Prompt: [`session_prompts/04_bet_explorer_badges.md`](./session_prompts/04_bet_explorer_badges.md)

  Surface the per-`Bet` predictions captured in Sessions 02
  + 03 in the existing Bet Explorer: confidence chip
  (green / amber / red) and risk tag (`±£X.XX`) next to the
  pair-class badge. Hides gracefully on pre-Session-02 bets
  and when the head is near its untrained default.

- [ ] **Session 05 — Model-detail calibration card**

  📄 Prompt: [`session_prompts/05_calibration_card.md`](./session_prompts/05_calibration_card.md)

  New card on the model-detail page: four-bucket reliability
  diagram, MACE summary number, risk-vs-realised scatter.
  Diagnostic — doesn't feed ranking. Reports only on
  held-out eval days (hard_constraints §13).

- [ ] **Session 06 — Scoreboard MACE column**

  📄 Prompt: [`session_prompts/06_scoreboard_mace_column.md`](./session_prompts/06_scoreboard_mace_column.md)

  Sortable MACE column on the Scoreboard's Scalping tab.
  Diagnostic only — does NOT feed composite ranking
  (hard_constraints §14 is explicit about this;
  ranking-invariant test is the tripwire).

## Phase 4 — Validation

- [ ] **Session 07 — Training run + analysis**

  📄 Prompt: [`session_prompts/07_validation_run.md`](./session_prompts/07_validation_run.md)

  **Prerequisite:** `activation_playbook.md` Steps A–E
  completed. Full-population training run on the Gen 1 day
  range at the promoted weights; compare against the Gen 1
  baseline (commit `7a3968a`) across four targets
  (fill-rate lift, MACE ≤ 5 % per bucket, risk Spearman ρ >
  0.3, `arbs_naked` trends down). Produces CSVs + markdown
  verdict; honest accounting — no "fix it in another
  session within this plan" moves.

---

## Summary

| # | Session | Phase | Prompt |
|---|---|---|---|
| 01 | Active re-quote mechanic | 1 | (archived in progress.md) |
| 02 | Fill-probability head | 2 | [02](./session_prompts/02_fill_prob_head.md) |
| 03 | Risk / variance head | 2 | [03](./session_prompts/03_risk_head.md) |
| — | Activation playbook (operator procedure) | — | [playbook](./activation_playbook.md) |
| 04 | Bet Explorer badges | 3 | [04](./session_prompts/04_bet_explorer_badges.md) |
| 05 | Model-detail calibration card | 3 | [05](./session_prompts/05_calibration_card.md) |
| 06 | Scoreboard MACE column | 3 | [06](./session_prompts/06_scoreboard_mace_column.md) |
| 07 | Training run + analysis | 4 | [07](./session_prompts/07_validation_run.md) |

7 sessions + 1 operator procedure. Sessions 01–03 are code-
heavy. Sessions 04–06 are UI surfaces that reuse the data
captured by 02 / 03. Session 07 proves the whole stack
against the Gen 1 baseline.
