# Progress — Reward Densification

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows `entropy-control-v2/progress.md`
— "What landed", "Not changed", "Gotchas", "Test suite",
"Next".

---

_Plan folder created 2026-04-19. See `purpose.md` for the
diagnosis flowing from `entropy-control-v2`'s Validation
entry (2026-04-19) and the `fill-prob-aux-probe`
2026-04-19 follow-on: entropy control is not the lever,
aux heads alone don't move the needle, the bottleneck is
reward sparsity. This plan introduces per-step mark-to-
market shaping to densify the training signal without
changing the raw P&L accounting._

---

## Session 01 — Mark-to-market scaffolding (knob at 0 default)

**Date:** 2026-04-19.
**Commit:** _pending_.

**What landed.**

- `BetfairEnv._compute_portfolio_mtm(current_ltps)` — iterates
  open matched bets (`outcome is UNSETTLED`, `matched_stake > 0`),
  applies the §6/§7 formulas, skips runners with missing/invalid
  LTP. Returns portfolio-level sum in pounds.
- Per-step hook in `BetfairEnv.step()` (after the terminal-bonus
  step, before `_get_info()`): computes `mtm_now = _compute_portfolio_mtm(
  _current_ltps())`, emits `mtm_delta = mtm_now − _mtm_prev`,
  multiplies by `mark_to_market_weight`, adds to the step's
  `reward` AND to `_cum_shaped_reward` so the raw+shaped invariant
  holds. `_mtm_prev` updates to `mtm_now`; `_cumulative_mtm_shaped`
  accumulates the weighted contribution for telemetry.
- New reward-config key `mark_to_market_weight`; default 0.0
  (byte-identical to pre-change). Whitelisted in
  `_REWARD_OVERRIDE_KEYS` so per-agent genes can flow through the
  existing `reward_overrides` passthrough.
- `info["mtm_delta"]` (pre-weight), `info["cumulative_mtm_shaped"]`
  and `info["mtm_weight_active"]` populated on every step.
- `EpisodeStats.mtm_weight_active` / `cumulative_mtm_shaped` fields
  added; `_log_episode` writes them to the per-episode JSONL row
  (optional fields — downstream readers tolerate absence on
  pre-change rows, mirroring `alpha`/`log_alpha`).
- 11 new tests in `tests/test_mark_to_market.py` (formula, knob
  default, reward-override passthrough, missing-LTP guard,
  resolved-bet exclusion, telescope property, full-rollout
  invariant at weight 0.05, byte-identical at weight 0, info
  presence).
- `tests/test_forced_arbitrage.py::TestScalpingReward::
  test_invariant_raw_plus_shaped_equals_total_reward`
  parametrised over `mtm_weight ∈ {0.0, 0.05}` — the
  load-bearing regression guard per the 2026-04-18
  units-mismatch lesson now covers the MTM path.
- `CLAUDE.md`: new "Per-step mark-to-market shaping (2026-04-19)"
  paragraph under "Reward function: raw vs shaped".

**Not changed.** Matcher semantics, entropy-control-v2 target-
entropy controller, PPO stability defences (ratio clamp ±5, KL
0.03, advantage normalisation, reward centering, LR warmup),
aux-head losses, smoke-gate assertion, action/obs schemas, raw
P&L accounting, GA gene ranges.

**Gotchas.**

- `_compute_portfolio_mtm` filters by `outcome is BetOutcome.UNSETTLED`.
  That's what makes the telescope close: after `bm.settle_race`
  mutates outcomes to WON/LOST/VOID, those bets stop contributing
  to MTM on the settle step, so the final `mtm_delta = 0 − MTM_{t-1}`
  unwinds the running portfolio value.
- `_current_ltps()` returns `{}` past the last tick of the last
  race; the portfolio MTM is then 0 via the missing-LTP path as
  well as the outcome filter. Either mechanism alone is enough;
  both belt-and-braces.
- The per-step hook runs AFTER step 5 (end-of-day bonus). The
  terminal bonus flows into `_cum_raw_reward`; the MTM flows
  into `_cum_shaped_reward`. No double-counting.

**Test suite.** 2275 passed, 7 skipped, 1 xfailed (same as
pre-change totals + 11 new MTM tests + 1 parametrisation on the
existing invariant guard = net +12 assertions, 11 net new tests).

**Scripted-rollout probe (qualitative).**

- Synthetic day, scalping mode, `mark_to_market_weight=0.05`,
  back bet at price 8.0 held across 5 pre-race ticks where the
  synthetic LTP is 4.0. Observed: `|MTM|` sits around £5 mid-race
  (stake 5.0 × (8−4)/4 = £5), drops to 0 at settle as the
  outcome filter kicks in. `cumulative_mtm_shaped` ends at
  `0.000000` within float tolerance. `raw + shaped ≈ total`.
  (Property covered as an assertion inside
  `test_mtm_telescopes_to_zero_at_settle`.)

**Next.** Session 02 — flip the config.yaml default from 0.0 to
0.05 so the next training run engages the mechanism without
per-plan overrides.

---

## Session 02 — Plan-level default weight (0.05)

**Date:** 2026-04-19.
**Commit:** _pending_.

**What landed.**

- `config.yaml` — added `reward.mark_to_market_weight: 0.05`
  under the reward block (preserves existing ordering). This
  is the project-wide default; agents that don't override via
  hp pick it up.
- `tests/test_config.py::test_mark_to_market_weight_default_matches_session_02`
  pins the value so a future refactor can't silently revert.
- `CLAUDE.md` — dated "Default weight 0.05 (2026-04-19,
  Session 02)" paragraph appended under the Session-01
  "Per-step mark-to-market shaping" subsection. Notes the
  reward-scale change: runs after this commit are NOT
  byte-identical to pre-change; scoreboard rows compare on
  `raw_pnl_reward` but not on `total_reward`.

**Not changed.** The mechanism itself (Session 01), formulas,
telescope property, invariant tests, target-entropy controller,
PPO stability defences, action/obs schemas, gene ranges, smoke-
gate assertion, raw P&L accounting.

**Gotchas.**

- This IS a reward-scale change per hard_constraints §19 —
  documented in the CLAUDE.md paragraph. Pre-change scoreboard
  rows are NOT directly comparable on `total_reward`. The MTM
  shaping lives entirely in `shaped_bonus`; `raw_pnl_reward`
  (race-settled cashflow only) remains comparable.
- The default propagates via `config["reward"]["mark_to_market_weight"]`
  → `BetfairEnv._mark_to_market_weight`; per-plan overrides via
  `reward_overrides.mark_to_market_weight` still work and take
  precedence over the default (same gene passthrough as the
  other reward knobs).

**Test suite.** 2276 passed, 7 skipped, 1 xfailed (+1 new pin
test in test_config.py).

**Next.** Session 03 — archive the existing registry and
redraft the probe training plan. Session 03 is
operator-gated per hard_constraints §21.

---

## Session 03 — Training-plan redraft + archive

**Date:** 2026-04-19.
**Commit:** _pending_.

**What landed.**

- Archived the live registry into
  `registry/archive_20260419T192404Z/`:
  - `models.db` (25 models — partial
    `activation-A-baseline` gen-0 + `fill-prob-aux-probe`
    7 completed / 2 still-running agents).
  - `weights/` (25 `.pt` files).
  - `training_plans/` (snapshot of the 5 historical plans;
    the live copies stay in-place for reference).
- Moved `logs/training/episodes.jsonl` →
  `logs/training/episodes.pre-reward-densification-20260419T192404Z.jsonl`
  (758 rows).
- Fresh registry:
  - New `registry/models.db` via `ModelStore()` — 0 models.
  - `registry/weights/` recreated empty.
  - `logs/training/episodes.jsonl` truncated to 0 bytes.
- New training plan
  `registry/training_plans/3403aaf5-73c0-472f-a719-bda47ba96540.json`:
  - `name: "reward-densification-probe"`
  - 9 agents (3 per arch — same `arch_mix` as
    `fill-prob-aux-probe` for clean comparison).
  - 1 generation, 3 epochs, `auto_continue=false`,
    `generations_per_session=1`.
  - `reward_overrides`: `fill_prob_loss_weight: 0.0`,
    `risk_loss_weight: 0.0` — aux heads off so the MTM
    signal is clean (2026-04-19 `fill-prob-aux-probe`
    showed aux-head weight 0.10 doesn't move the needle on
    its own). No explicit `mark_to_market_weight` override
    — picked up from the config.yaml default (0.05 per
    Session 02).
  - `hp_ranges`: copied from `fill-prob-aux-probe` so genes
    roll identically.
  - `seed: 421` (different from fill-prob-aux-probe's 137).
  - `status: "draft"`; all runtime fields null.
- Plan validates via
  `PlanRegistry('registry/training_plans').list()` — the
  new plan loads alongside the 5 historical plans as
  expected.

**Not changed.** Code / tests (none in Session 03 per
hard_constraints §21). `plans/INDEX.md` already carries the
`reward-densification` row from plan-folder creation.

**Gotchas.**

- The registry archive path is on disk only — the
  `registry/archive_*/` folders are gitignored per the
  project's standard pattern. Only plan docs
  (`progress.md`, `INDEX.md`) show in the diff; the commit
  body documents the archive location for post-mortem
  reference.
- `reward_overrides` does NOT carry
  `mark_to_market_weight` — letting the config.yaml default
  (0.05) apply avoids a duplicate source of truth. If the
  probe's findings motivate a different weight, bumping the
  config and keeping the plan override-free lets the A/B
  swap live in one file.

**Pre-reset state (archived).**

- `registry/archive_20260419T192404Z/models.db` — 25 models.
- `registry/archive_20260419T192404Z/weights/` — 25 `.pt`
  files.
- `logs/training/episodes.pre-reward-densification-20260419T192404Z.jsonl`
  — 758 rows (A-baseline gen-0 partial + fill-prob-aux-probe
  7-completed state).

**Test suite.** Not re-run (Session 03 is registry + plan
JSON only; no code changes). Sessions 01/02 covered the
code-level regression suite.

**Next.** Operator launches the
`reward-densification-probe` plan with "Smoke test first"
ticked. Post-run Validation entry follows here, per
`master_todo.md §After Session 03`.
