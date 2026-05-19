# robust-phenotype — hard constraints

These are the rules that must hold across all R1-R5 work. Each is
load-bearing — violating one is grounds to reject a PR / cohort
run.

## §1 — Defaults preserve byte-identity

Every new reward-override or env kwarg added in this plan has a
default value that leaves training byte-identical to pre-plan when
unset. Specifically:

- ``composite_score_mode=sortino`` is an opt-in mode; existing
  ``day_pnl_per_std`` and other modes remain unchanged.
- ``naked_loss_quadratic_beta=0.0`` default = no contribution to
  shaped (existing ``naked_variance_penalty_beta`` symmetric form
  untouched).
- ``opposite_side_depth_floor=None`` default = no gate.
- ``open_mask_max_ltp_velocity=None`` default = no mask.

Cohort scoreboard rows from pre-plan runs remain directly
comparable to post-plan runs on ``raw_pnl_reward`` unless an
override is explicitly set.

## §2 — Reward-channel separation preserved

The "raw + shaped ≈ total_reward" invariant from CLAUDE.md
"Reward function: raw vs shaped" must hold per episode. R3's
quadratic naked-loss penalty MUST land in the shaped channel, not
raw. The CLAUDE.md note on this is load-bearing — any new term
that breaks the invariant violates the documented contract and
will silently break the trainer's reward centering.

## §3 — R3 is asymmetric BY CONSTRUCTION

The quadratic naked-loss penalty fires only on the loss side:
``shaped -= β × sum(min(0, p)² for p in per_pair_naked)``. Naked
WINNERS contribute zero to this term (the existing
``naked_winner_clip`` already neutralises 95% of them; adding
another winner-side term would over-clip and starve the policy of
gradient on real arbitrage opportunities).

The asymmetry is the entire point. A symmetric quadratic penalty
would just be a tighter version of the existing
``naked_variance_penalty_beta`` and has already been tested
(2026-05-15 tnv1 finding: variance compression alone doesn't
clear bands).

## §4 — R1's downside_deviation handles zero gracefully

``downside_deviation = sqrt(mean(min(0, day_pnl)²))``. If an agent
has zero negative days, downside_deviation = 0 and the score
division blows up. The implementation must:

- Floor the denominator at some small constant (e.g. £1) — protects
  against infinity AND rewards the rare "no bad days" agent
  appropriately.
- Or: switch to ``mean(day_pnl) - λ × downside_deviation`` (a
  Sortino-shaped score additively instead of as a ratio).

The additive form is more numerically stable and easier to reason
about. Default to that unless ratio gives clearly better
discrimination at probe scale.

## §5 — R4 must read post-junk-filter depth

The opposite-side ladder depth check MUST happen AFTER the matcher's
±50% junk-filter applies. Reading raw ladder depth would include
the £1-£1000 stale parked orders that the matcher already discards
(CLAUDE.md "Order matching: single-price, no walking" §junk filter).
Pre-filter depth on a thin runner could read "+£1000 available"
when post-filter is "+£3 available."

R4's implementation must call the same ``pick_top_price`` /
post-filter accessors that E3 uses.

## §6 — R5's velocity check uses an existing predictor feature

The velocity threshold reads ``ltp_velocity_30`` from the runner
observation (already produced by feature_engineer.py). Don't add
a new tick-level feature for R5; the existing one is sufficient.

## §7 — Tests guard every new reward-override path

Each of R1-R5 lands with at least:

- A default-off byte-identical test (the override unset gives the
  same result as the pre-plan path)
- A direct mechanism test (the override engaged produces the
  expected gradient / refusal / mask)
- An info-dict telemetry test (any counter or scale field
  surfaces correctly)

Pattern mirrors the E1-E6 tests in
``tests/test_forced_arbitrage.py``. New tests go in the same file
or a sibling ``tests/test_robust_phenotype.py``.

## §8 — Compatibility with E3's mechanism

R4 (liquidity-floor) EXTENDS E3's close-feasibility gate; it is
not a replacement. The recommended deployment combines E3
(close_feasibility_max_spread_pct=0.05) with R4
(opposite_side_depth_floor=£X). Both default-off; both engaged
in the R1+R3+R4 probe.

## §9 — Probe-scale ablation discipline

The R1+R3+R4 probe runs THREE mechanisms simultaneously by design
(orthogonal: selection vs reward vs env). If it bites, the
follow-on probes should ablate individually:

- R1 alone vs baseline
- R3 alone vs baseline
- R4 alone vs baseline

Don't escalate to a full cohort until at least ONE of the three
individual ablations also bites — otherwise we're going to scale
with a confounded mechanism stack.
