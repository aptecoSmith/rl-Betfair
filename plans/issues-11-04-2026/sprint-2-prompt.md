# Sprint 2 — Training Config (8 sessions)

Two issues: configurable budget (5 sessions) then market type filter
(3 sessions). These enable £10/race training and specialised WIN/EW models.

## Before you start

- Verify Sprint 1 landed: test suite is green, inactivity penalty gene exists.
- Read `plans/issues-11-04-2026/order.md` for context.

## Issue 01 — Configurable Budget & Percentage-Based P&L (5 sessions)

Read the full plan folder: `plans/issues-11-04-2026/01-configurable-budget/`

Start with `purpose.md`, then `hard_constraints.md`, then work through
`session_prompt.md` sessions 01–05 in order.

Summary of sessions:
1. Per-plan `starting_budget` override (training plans carry optional budget)
2. Record `starting_budget` per evaluation run (schema migration)
3. Percentage return in scoreboard (% column alongside raw P&L)
4. Model detail + bet explorer budget context
5. Percentage-based discard threshold

Key insight from the analysis: composite scoring already normalises by
budget — this is mostly display + per-plan configurability.

Has an ai-betfair knock-on file: `knockon_ai_betfair.md` — read it,
but no action needed now (trivial display changes).

**Exit per session:** All tests pass, `progress.md` updated, commit.

---

## Issue 04 — Market Type Filter Gene (3 sessions)

Read the full plan folder: `plans/issues-11-04-2026/04-market-type-filter/`

Start with `purpose.md`, then `hard_constraints.md`, then work through
`session_prompt.md` sessions 01–03 in order.

Summary of sessions:
1. Add `market_type_filter` gene (WIN/EACH_WAY/BOTH/FREE_CHOICE) + env
   filtering in `reset()`
2. Evaluator filtering (same filter applied during eval)
3. Scoreboard + model detail display (badge per model)

Critical constraint: NO observation or action schema changes. The filter
controls which races are presented, not the obs/action shape. Weights
remain cross-compatible across filter values.

Has an ai-betfair knock-on file: `knockon_ai_betfair.md` — radio buttons
on the go-live page. ~1 session in ai-betfair.

**Exit per session:** All tests pass, `progress.md` updated, commit.

---

## Sprint complete

After all eight sessions:
1. Full test suite green.
2. Push: `git push origin master`.
3. Run training at £10 budget with the new market type filter gene enabled.
   Try a population that includes WIN-only, EW-only, and BOTH models.
4. Check the scoreboard shows % return and market type badges.
