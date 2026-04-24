---
plan: force-close-sizing-review
status: draft
---

# Master todo — force-close-sizing-review

Two-session shape: **Session 01 = written design review**, **Session
02+ = implement operator's pick**. Session 02+ contents depend on
the decision so they're sketched generically.

## Session 01 — design review & recommendation

### Read first

- [ ] `plans/ppo-stability-and-force-close-investigation/findings.md`
      §"Problem 2 — force-close sizing" (source of truth)
- [ ] CLAUDE.md sections:
  - "Force-close at T−N (2026-04-21)"
  - "Order matching: single-price, no walking"
  - "Equal-profit pair sizing (scalping)"
- [ ] `plans/arb-signal-cleanup/hard_constraints.md` §7, §14
      (force-close exclusions from matured and close_signal bonuses)
- [ ] `env/betfair_env.py`:
  - `_force_close_open_pairs` (~line 2430) — call-site
  - `_attempt_close` — relaxed matcher path under
    `force_close=True`
  - `_settle_current_race` — `scalping_force_closed_pnl`
    accumulation
  - `_get_info` — refusal counters
    (`force_close_refused_no_book`, `_place`, `_above_cap`)
- [ ] `env/exchange_matcher.py::ExchangeMatcher._match` —
      `force_close` flag handling
- [ ] `env/bet_manager.py::place_back/place_lay` — overdraft
      semantics under `force_close=True`

### Quantify the baseline

- [ ] From `logs/training/episodes.jsonl` cohort W rows, compute:
  - Per-race distribution of `arbs_force_closed` (already in
    findings.md: min 0, mean 182, max 834).
  - Per-race `scalping_force_closed_pnl` (mean −£213).
  - Per-race `force_close_refused_*` counts (the "would-have-
    closed-but-couldn't" surface).
  - The ratio of force-closes to `arbs_completed +
    arbs_closed + arbs_naked` — how much of the total pair
    lifecycle ends in a force-close.
- [ ] Per-agent table: force-close count vs total_reward rank.
      Confirms (or refutes) the "top agents bet less" observation
      from findings.md.

### Replay probe (read-only, no new training)

Build a deterministic replay that reloads a single gen-1
checkpoint weight file and re-runs 3 days of races, once per
option, without touching the live worker. Options need an env-
level switch to toggle for the replay only; the switch does NOT
need to be a gene, just an env-init argument.

- [ ] Option 0 — baseline: current behaviour (force_close=True,
      max_back_price=50, fractional k=1).
- [ ] Option 1 probe: `max_back_price` on force-close path ∈ {50,
      30, 20, 15}. Table of per-race aggregate cost.
- [ ] Option 3 probe: fractional k ∈ {1.0, 0.5, 0.25}.
- [ ] Option 4 probe: budget cap ∈ {none, 100, 50, 20}.
- [ ] Option 2 probe: harder — requires a reward-accounting
      change to score. Skip for session 01; describe the
      expected behaviour qualitatively and defer measurement to
      session 02 if Option 2 is picked.

Store probe outputs under `plans/force-close-sizing-review/
probe_outputs/` (JSON summaries, not raw logs).

### Write `design_review.md`

For each option: mechanism, implementation surface, per-race cost
measured in the replay, scoreboard-comparability notes, risks,
effort estimate (in sessions).

- [ ] Include a table at the top: option × (mean fc cost / race,
      mean naked count / race, estimated effort).
- [ ] Include an explicit recommendation to the operator with
      ONE preferred path and the conditions under which an
      alternative becomes better.
- [ ] Include the "do nothing until PPO fix lands" option
      (Option 5) with the specific re-review gate.

### Hand off to operator

- [ ] Summarise in a hand-off note: "operator picks session 02
      scope from options 1..5". Plan goes to `status: pending-
      operator-review` (or the project's equivalent).

## Session 02+ — implement operator's pick (contents depend on pick)

### If Option 1 (tighter cap) picked

- [ ] Add force-close-specific `max_back_price_force_close` to
      `config.yaml::betting_constraints`. Default = current
      `max_back_price` so the plan is byte-identical when the
      knob isn't set.
- [ ] Thread it through `ExchangeMatcher` so the hard cap used
      under `force_close=True` differs from the one used under
      strict match.
- [ ] New test: `test_force_close_respects_tighter_cap` — strict
      matcher accepts, relaxed matcher refuses at the lower cap.
- [ ] Smoke probe: confirm force-close aggregate cost drops;
      confirm `arbs_naked` doesn't explode.
- [ ] Update CLAUDE.md.

### If Option 2 (time-phased escalation) picked

- [ ] New constraint
      `close_signal_preferred_window_seconds` — e.g. 5.
- [ ] New shaped-reward term: +£X per `close_signal` success
      INSIDE the window (zero elsewhere). X bounded by the per-
      race expected force-close spread cost × some < 1 factor so
      the incentive aligns without overshooting.
- [ ] Update `hard_constraints.md §7, §14` exclusion text to note
      the window's shaping.
- [ ] Reward-shape change → scoreboard break → document in
      CLAUDE.md under "Reward function".
- [ ] Tests: in-window close gets bonus; at-window-boundary close
      gets bonus; force-close triggered close does NOT get bonus;
      BC targets still work.
- [ ] Smoke probe.

### If Option 3 (fractional sizing) picked

- [ ] New constraint `force_close_fractional_k` ∈ (0, 1], default
      1.0 = byte-identical.
- [ ] `_attempt_close` with `force_close=True` scales the
      equal-profit stake by `k` and leaves the residual open.
- [ ] Confirm accounting: matched portion → `arbs_force_closed`;
      residual → existing naked accounting at settle.
- [ ] Tests: k=1 byte-identical; k=0.5 halves close stake; k=0
      is an error (not supported).
- [ ] Smoke probe.

### If Option 4 (budget cap) picked

- [ ] New constraint `force_close_per_race_cap`. Default 0 =
      disabled = byte-identical.
- [ ] `_force_close_open_pairs` tracks per-race count; past the
      cap, skip with a new refusal reason `force_close_refused_
      budget`.
- [ ] Selection rule under cap: worst-spread-first OR random
      (operator choice during session 01).
- [ ] Tests: cap=0 is no-op; cap=1 force-closes exactly one pair;
      skipped pairs settle naked.
- [ ] Smoke probe.

### If Option 5 (do nothing) picked

- [ ] Document the decision in `lessons_learnt.md`.
- [ ] Plan `status: complete` with a follow-up gate: "re-review
      after `ppo-kl-fix` has landed and one full GA cycle has
      run under the fix".
- [ ] No code changes.

## Close-out (whichever option)

- [ ] CLAUDE.md updated under "Force-close at T−N" subsection.
- [ ] `lessons_learnt.md` written — include the force-close-
      magnitude vs reward-signal framing so future design
      reviews don't re-discover the selection-pressure trap.
- [ ] Plan `status: complete`.
