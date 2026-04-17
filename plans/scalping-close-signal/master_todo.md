# Master TODO — Scalping Close-Signal

Ordered list of sessions. Each session must land green (full
`pytest tests/ -q`, frontend `ng test`, browser verify for UI
changes) before the next begins. Session 01 is the only session
strictly required to unblock re-running activation-A-baseline;
Sessions 02–03 polish the observability surface.

## Session 01 — `close_signal` action + placement path

**Status:** pending

**Deliverables:**
- `ACTION_SCHEMA_VERSION: 3 → 4`, `SCALPING_ACTIONS_PER_RUNNER: 6 → 7`.
- `close_signal` (7th per-runner action) processed after the re-quote
  pass in `env/betfair_env.py:_process_action`. Threshold `> 0.5`
  matches the `requote_signal` convention.
- New method `env/betfair_env.py::_attempt_close` mirroring the
  `_attempt_requote` shape: locate the open aggressive leg + pair_id,
  cancel the outstanding passive, place an aggressive opposite-side
  bet at current market best with equal-P&L sizing.
- Commission gate bypassed on the close path (per hard_constraints §4).
- Per-episode `arbs_closed` counter wired through `BetManager` and
  `EpisodeStats`, written to `episodes.jsonl`.
- Distinct activity-log line format:
  `"Pair closed at loss: Back £X / Lay £Y on runner Z → realised −£W"`.
- Policy-network migration helper
  `agents.policy_network.migrate_scalping_action_head_v3_to_v4` that
  pads the actor head + `action_log_std` for the new dim, matching
  the Session-01-era `_v2_to_v3` helper pattern.
- Tests: the six cases listed in `hard_constraints.md` §14.

**Exit criteria:**
- Fresh training run (scalping_mode on) produces non-zero
  `arbs_closed` for at least one agent over 10 episodes.
- Legacy checkpoint refused on strict load; migrates cleanly via
  helper.
- Reward invariant `raw + shaped ≈ total_reward` per episode
  unchanged by the new path (CLAUDE.md).
- Pytest + ng test green.

**Acceptance:** a test explicitly verifying that a close-at-loss pair
contributes 0 to raw reward while its cash cost appears in
`info["day_pnl"]`.

## Session 02 — Learning-curves + Bet Explorer surfaces

**Status:** pending

**Deliverables:**
- `EpisodeRecord` TypeScript type in
  `frontend/src/app/learning-curves/agent-diagnostic.ts` gains an
  `arbs_closed?: number` field.
- Analyzer captions gain a "closes rate" line when `arbs_closed > 0`
  — e.g. `"5% of pair attempts closed at loss (1 of 20)"`. Goes into
  `AgentDiagnostic.captions.arbRate` or a new `captions.closes`
  field.
- Verdict thresholds:
  - An agent whose `arbs_closed` trends up while `arbs_naked` trends
    down reads as LEARNING (new signal on top of the existing
    arb_rate check).
- Bet Explorer: new "Closed" status chip for pairs where both legs
  matched via the close path (identifiable by a timing signature the
  Session-01 env records — e.g. a `pair_close_tick` field on the
  aggressive `Bet`, or the simpler heuristic "passive's
  `matched_size > 0` AND `matched_price` is further from ladder-best
  than typical", to be confirmed during implementation).

**Exit criteria:**
- Frontend specs green.
- Browser-verify: pair explicitly closed by agent shows the new chip;
  un-closed completions still show the existing "Locked" chip.

## Session 03 (optional) — CLAUDE.md + progress.md integration

**Status:** pending

**Deliverables:**
- CLAUDE.md "Reward function: raw vs shaped" section gains a note:
  "close_signal (v4 action schema) lets the agent substitute
  closes for nakeds. A closed pair contributes 0 to raw reward
  (locked floors to 0, naked is 0); the cash cost lands in
  `day_pnl` only."
- `plans/scalping-active-management/progress.md` gets an entry
  cross-linking to this plan, explaining that Session 07's
  measurement basis changed.
- `plans/scalping-close-signal/progress.md` gets Session 01 + 02
  entries with commit hashes.

**Exit criteria:**
- Prose merge, no code changes.
- Commits reference both plan folders.

---

## After Session 01: re-run activation-A-baseline

Once Session 01 lands (not waiting for 02/03), reset activation-A-
baseline and re-launch to measure close_signal's impact on the same
reward config that produced the gen-2 population under discussion.
Document the before/after in
`plans/scalping-active-management/progress.md`.
