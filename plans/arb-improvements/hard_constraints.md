# Hard Constraints — Arb Improvements

These are the invariants that must hold through every session. If
any session's design forces one of these to break, stop and escalate
rather than regressing a constraint.

## From `CLAUDE.md` — inherited project invariants

- **No ladder walking.** Bets match at one price only. Any oracle
  generation, feature computation, or BC target that simulates "what
  the env would do" must never peek past the single post-filter best
  level.
- **LTP-aware junk filter.** Ladder levels > `max_price_deviation_pct`
  from LTP are dropped. Feature functions and oracle scan both apply
  the filter before computing arb spreads — a £1000 parked lay is
  not a valid arb leg.
- **Hard price cap enforced after filter.** `betting_constraints.max_back_price`
  / `max_lay_price` gate on the *best post-filter* price, never the
  raw top-of-book.
- **Reward shaping stays zero-mean for random policies.** No
  asymmetric positive-per-bet bonus sneaks in via new features, BC
  loss, or aux head.
- **`raw + shaped ≈ total_reward`** per episode. Reward clipping is
  a *training-signal* transform; it does not enter the `raw` or
  `shaped` accumulators.
- **`info["day_pnl"]` is authoritative.** `info["realised_pnl"]` is
  last-race-only and must not be consumed by BC / oracle code.
- **`env.all_settled_bets` for whole-day bet history.**
  `env.bet_manager.bets` is last-race-only. Oracle scanning against
  the env must not confuse the two.
- **Obs schema version bump is loud.** Adding any RUNNER_KEYS /
  MARKET_KEYS entry increments `OBS_SCHEMA_VERSION`; old checkpoints
  refuse to load. No silent zero-pad.

## Plan-specific invariants

### Stabilisation (Phase 1)

- **Reward clip affects the training signal only.** `EpisodeStats`,
  `info["day_pnl"]`, the log line, and the monitor progress events
  all carry unclipped values. Add `clipped_reward_total` as a
  *separate* field, not a replacement.
- **Grad clip stays on.** The fix is to stop outliers entering the
  update, not to remove the safety net downstream.
- **Entropy floor raises the coefficient, not the distribution.**
  The floor controller scales `entropy_coefficient`; it never
  overrides the policy's action distribution directly.
- **Action-bias warmup adds to the *mean* of a continuous head.** It
  is a soft prior, not a hard override. Decays linearly to zero by
  the warmup epoch.
- **All Phase-1 knobs default to 0 / off.** Byte-identical training
  when untouched.

### Features (Phase 2)

- **Pure functions, stdlib only.** Live in `env/features.py`
  alongside the existing six (`compute_microprice`, `compute_obi`,
  …). Duck-typed `PriceLevel` inputs. Must be vendorable into
  `ai-betfair` without modification.
- **Zero when unpriceable.** No LTP / one-sided book / empty book
  → `0.0`. Never `NaN`, never a fabricated negative that reads as
  "crossed book exists".
- **Single commission constant.** Post-commission arb profit feature
  and `BetManager.get_paired_positions` import the same module-level
  constant. No duplicate literal `0.05` in the feature module.
- **Features are always on.** Unconditional on `scalping_mode`. The
  cost is a handful of floats per tick; directional models benefit
  too. This simplifies the schema story — one `OBS_SCHEMA_VERSION`
  bump covers both modes.
- **`data/feature_engineer.py` stays in lockstep.** Cached features
  on disk match the env's RUNNER_KEYS / MARKET_KEYS exactly, or
  the env refuses them.

### Oracle + BC (Phase 3)

- **Oracle is offline-only.** Never runs inside a training epoch.
  Produces deterministic `.npz` per date; training reads, never
  writes. Scan time is free.
- **Oracle targets must be reachable by the real policy.** If the
  env would reject the paired order (junk filter, price cap,
  budget, max_bets_per_race), the oracle does not emit it. A BC
  target the env rejects at rollout time is a phantom target that
  teaches nothing.
- **BC is per-agent.** Each agent in a generation pretrains
  independently from its own initialisation. Sharing BC weights
  across a population collapses behavioural diversity on
  generation 0.
- **BC is skippable.** Empty oracle dataset → skip BC for that
  day, log warning, continue. Do not crash.
- **BC only trains heads the oracle ground-truths.** Currently
  `signal` (cross-entropy) and `arb_spread` (MSE). Other heads
  stay at their init; the policy learns them via PPO.

### Auxiliary head (Phase 4)

- **Gated by default.** `training.aux_arb_head=False` means no
  forward-pass change, no memory cost, no loss term.
- **Target comes from the same oracle scan.** No second source of
  ground truth. One oracle, two consumers (BC and aux).

### UI

- **No knob is finished until it's editable in the wizard.** Every
  new config key opened by a session must be in `ui_additions.md`
  with a tick-box for Session 8 to consolidate.
- **Frontend `ng build` stays clean** through every session that
  touches the frontend.

## Cross-repo

- **`env/features.py`, `env/exchange_matcher.py`, `env/tick_ladder.py`**
  stay dependency-free and vendorable into `ai-betfair`. New feature
  functions obey this rule.
- **`env/features.py` changes are mirrored to the live-inference
  project's postbox** (`feedback_incoming_postbox.md` in
  user memory) when they ship, so live inference picks them up.

## Test gates

- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `cd frontend && ng build`.
- Per-plan fast tests live in `tests/arb_improvements/`; run them
  first in every session (<30 s total target).
- `git push all` after each session commit.
