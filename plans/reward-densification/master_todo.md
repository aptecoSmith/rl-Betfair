# Master TODO — Reward Densification

Three sessions, one commit per session, hard-constrained by
`hard_constraints.md`. Sessions 01–02 are automatable;
Session 03 is operator-gated.

---

## Session 01 — Mark-to-market scaffolding (knob at 0, byte-identical default)

**Status:** pending

**Deliverables:**

- `env/betfair_env.py`:
  - Read `mark_to_market_weight` from the reward-config
    block; default `0.0`.
  - New method `_compute_portfolio_mtm(current_ltps: dict[int,
    float]) -> float` — iterates `self.bet_manager.bets`
    (open bets only), applies §6 / §7 formulas, sums. Missing
    LTP → that bet contributes 0.
  - State: `self._mtm_prev: float = 0.0` per race; reset on
    race-start.
  - Per-step: after bet placement and matching logic, compute
    `mtm_now = _compute_portfolio_mtm(ltps)`. Delta is
    `mtm_now − self._mtm_prev`. Multiply by weight for shaped
    contribution. Update `self._mtm_prev = mtm_now`.
  - On race settle (`_settle_current_race`): resolved bets no
    longer appear in the MTM sum, so `mtm_now` naturally
    drops to zero for any closed-out position. The delta
    from `mtm_prev` to 0 is the LAST shaped contribution for
    that position on the settle step. Then reset
    `self._mtm_prev = 0.0` for the next race.
  - `info["mtm_delta"]` populated every step with the
    pre-weight delta.
  - `EpisodeStats` (or the equivalent container) gains
    `cumulative_mtm_shaped: float` accumulating `weight ×
    delta` across the episode; exposed via `info["...rollup"]`.
- `agents/ppo_trainer.py`:
  - `_log_episode` writes `mtm_weight_active` and
    `cumulative_mtm_shaped` into the JSONL row (per §13).
- `tests/test_betfair_env.py` (or a new
  `tests/test_mark_to_market.py` if scope warrants):
  - `test_mark_to_market_weight_default_is_zero`.
  - `test_mtm_delta_zero_when_no_open_bets`.
  - `test_mtm_back_formula_matches_spec`.
  - `test_mtm_lay_formula_matches_spec`.
  - `test_mtm_zero_when_ltp_missing`.
  - `test_mtm_telescopes_to_zero_at_settle`.
  - `test_mtm_weight_zero_byte_identical_rollout`.
  - `test_info_mtm_delta_field_present`.
- `tests/test_forced_arbitrage.py` (or wherever the
  raw+shaped invariant test lives):
  - Extend existing
    `test_invariant_raw_plus_shaped_equals_total_reward` to
    also run with `mark_to_market_weight=0.05` set; assert
    invariant still holds.
- `CLAUDE.md`: new paragraph under "Reward function: raw vs
  shaped" describing the MTM shaping. Documented with
  weight=0 being byte-identical to pre-change so historical
  scoreboards stay comparable.

**Exit criteria:**

- `pytest tests/ -q` green. Expect ≈ +8 net tests.
- Pre-existing `test_invariant_raw_plus_shaped_equals_
  total_reward` still green both at weight=0 (byte-
  identical) and at weight=0.05 (invariant holds because
  MTM telescopes to zero at settle).
- Pre-existing
  `test_real_ppo_update_feeds_per_step_mean_to_baseline`
  still green — reward-centering path untouched.
- Pre-existing target-entropy controller tests
  (`TestTargetEntropyController`) still green — the
  reward-path change is orthogonal to the controller.
- A scripted-rollout probe (documented in `progress.md`,
  not a pytest) with `mark_to_market_weight=0.05`: 3-race
  scripted rollout with an open back bet held across ticks
  while LTP moves. Assert cumulative shaped MTM ≈ 0 at
  settle within float tolerance; per-episode
  `raw + shaped` matches `total_reward`.

**Acceptance:** every new test passes; all existing tests
stay green; weight=0 rollouts are byte-identical to
pre-change. At least one test exercises the real
`_ppo_update` path with weight>0 to catch integration
regressions (per the 2026-04-18 units-mismatch lesson in
`naked-clip-and-stability`).

**Commit:** one commit, type `feat(env)`. First line:
`add per-step mark-to-market shaping (weight 0 default =
no-op)`. Body cites entropy-control-v2's Validation
conclusion, the fill-prob-aux-probe's supporting evidence,
and the design constraint that cumulative shaped MTM
telescopes to zero at settle.

**Session prompt:**
[`session_prompts/01_mark_to_market_scaffolding.md`](session_prompts/01_mark_to_market_scaffolding.md).

---

## Session 02 — Plan-level default weight

**Status:** pending

**Deliverables:**

- `config.yaml`:
  - Add `reward.mark_to_market_weight: 0.05` under the
    reward block. This is the project-wide default; agents
    that don't override via hp pick it up.
- `tests/test_config_defaults.py` (or similar):
  - `test_mark_to_market_weight_default_matches_session_02`
    — pin the value so future refactors can't silently
    revert.
- `CLAUDE.md`: dated paragraph noting the 0.05 default
  landed (separate from the mechanism landing in Session
  01). Rationale: "MTM deltas are O(pence-to-pounds) per
  tick; 0.05 × cumulative scales the shaped contribution
  to order-of-magnitude-comparable with per-race raw P&L
  without dominating it."

**Exit criteria:**

- `pytest tests/ -q` green, no regressions.
- `config.yaml` change is byte-minimal (one new line under
  `reward`), doesn't reorder other keys.

**Acceptance:** the default knob is now non-zero so a plain
`python -m training.worker` run engages the densification
mechanism without requiring per-plan overrides. The
Validation run picks it up automatically.

**Commit:** one commit, type `chore(config)` or `feat(env)`
(arguably a small feature). First line:
`default mark_to_market_weight to 0.05`. Body cites the
purpose.md magnitude-calibration rationale.

**Session prompt:**
[`session_prompts/02_default_weight_and_gene.md`](session_prompts/02_default_weight_and_gene.md).

---

## Session 03 — Training-plan redraft + archive (operator-gated)

**Status:** pending

**Deliverables:**

- Archive the current registry + episodes log:
  - `registry/models.db` →
    `registry/archive_<isodate>Z/models.db`.
  - `registry/weights/` →
    `registry/archive_<isodate>Z/weights/`.
  - `logs/training/episodes.jsonl` →
    `logs/training/episodes.pre-reward-densification-<isodate>.jsonl`.
  - `registry/training_plans/` copied into the archive for
    audit-trail. The `fill-prob-aux-probe` plan JSON stays
    in-place (completed, keep for reference).
- Fresh registry:
  - New `registry/models.db` via `ModelStore()`.
  - `registry/weights/` recreated empty.
  - `logs/training/episodes.jsonl` truncated.
- New training plan JSON in `registry/training_plans/`:
  - Name: `reward-densification-probe`
  - 9 agents (3 per architecture, same arch_mix as
    `fill-prob-aux-probe`).
  - 1 generation, 3 epochs, auto_continue=false,
    generations_per_session=1.
  - `reward_overrides`: empty OR just
    `mark_to_market_weight: 0.05` as a belt-and-braces
    pin (the config.yaml default also sets it; either
    way, the run runs with 0.05).
  - `hp_ranges`: copy from `fill-prob-aux-probe` so
    genes roll identically. No `mark_to_market_weight`
    in the gene ranges (§11).
  - Different `seed` from `fill-prob-aux-probe` (say
    `seed=421`) so initial gene rolls differ.
  - `status="draft"`, all runtime/status fields null.
- `plans/INDEX.md`: new row for `reward-densification`.

**Exit criteria:**

- New `registry/models.db` has `select count(*) from models`
  → **0**.
- `episodes.jsonl` is 0 bytes.
- The new plan JSON validates (loadable via `PlanRegistry`;
  passes the existing schema checks in `test_training_plan.py`).
- Pre-reset state (scoreboard rows from `fill-prob-aux-probe`)
  captured in the archive folder.
- `git status` clean except for the gitignored archive
  folders.

**Acceptance:** operator can tick "Smoke test first" in the
UI, select `reward-densification-probe`, and launch. The
probe runs cleanly against the fresh registry with
`mark_to_market_weight=0.05` engaged.

**Commit:** one commit, type `chore(registry)`. First line:
`archive pre-reward-densification registry + redraft probe
plan`. Body cross-references Sessions 01–02 commit hashes
and notes the archive location.

**Session prompt:**
[`session_prompts/03_validation_launch.md`](session_prompts/03_validation_launch.md).

---

## After Session 03: launch + validate

Once Session 03 lands and the registry is reset:

1. **Operator launches `reward-densification-probe`** with
   the smoke-test-first checkbox ticked.
2. **Smoke test runs.** Expected: passes the tracking-error
   gate at target=150 (the controller is untouched; only
   the reward shape changes). If smoke fails on the
   existing `entropy_tracks_target` assertion, that's
   unexpected — entropy dynamics shouldn't depend strongly
   on the new shaping over 3 episodes. Capture in
   `lessons_learnt.md`.
3. **Full population trains** if smoke passes. Watch the
   learning-curves panel for the success criteria from
   `purpose.md §What success looks like`:
   - At least 50 % of agents remain active through ep15.
   - `policy_loss` stays O(1)+ through ep15.
   - At least one agent reaches reward > −500 by ep15.
   - At least one agent's `arbs_closed / arbs_naked`
     ratio clears 10 %.
   - `raw + shaped ≈ total` invariant holds
     episode-by-episode.
4. **Capture findings** in `progress.md` under a
   "Validation" entry. Same shape as the Validation entries
   in `naked-clip-and-stability/progress.md` and
   `entropy-control-v2/progress.md`.
5. **Green light for scale** if validation succeeds — a
   follow-on session (or a follow-on plan) lifts the probe
   to a 16-agent 4-generation run matching the
   `activation-*` plan scale. Otherwise open the queued
   observation-space plan if diagnostics point there, or
   a tuning session if the mechanism works but the weight
   is wrong.

## Queued follow-ons (for context; not in this plan)

- **`reward-densification-gene`** — if the probe shows the
  mechanism works but the magnitude is wrong, add
  `mark_to_market_weight` to `hp_ranges` so the GA can
  mutate it per-agent. Range probably `[0.01, 0.2]` based
  on probe results.
- **`observation-space-audit`** — if the probe fails on the
  same passive/bleeding bifurcation as A-baseline and
  fill-prob probe, the policy's observation is the next
  suspect. Audit what runner-state / ladder-state features
  the policy sees; the reward may be fine, the state
  representation insufficient.
- **`mark-to-market-smoothing`** — if probe shows high
  policy-gradient variance traced to per-tick LTP noise,
  replace raw LTP with an EMA reference. Second-pass
  refinement only.
