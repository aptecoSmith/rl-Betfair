# Recipe Expansion and Robustness — Round 5

## Why

Phase B (`plans/bc-label-augmentation/`) demonstrated that BC label
augmentation **mechanically works**: F2 (close+hold labels) restored
cls% from BC's collapsed 5-26% range back up to 41.2% — actually
HIGHER than the no-BC baseline C2 (38.7%). The fundamental
"BC kills close_signal" problem is solved.

The trade-off F2 showed is selection regression — mat% dropped to
3.2% and opens fell to 93 (below the 100-180 target band). F3 / F3b
(full L2+L3a+L4 stack, currently running) are expected to restore
selection by adding L2's NOOP-at-oracle-negative labels back.

Whatever falls out of F3/F3b will be the **base recipe** for Round
5's expansion. The goal of this round is:

1. **Robustness** — confirm the recipe isn't seed-lucky.
2. **Dose-response** — explore around the recipe's current
   defaults (BC steps, pwin_back threshold, positive_weight) to
   find optima.
3. **Lifetime** — does multi-generation PPO continue lifting
   metrics on top of the base, or does the recipe plateau at 1
   gen?
4. **Re-test dropped levers** — with a strong base, do gates that
   regressed in earlier rounds (lay-side pwin, race confidence)
   start helping?
5. **Information probes** — does the direction predictor signal
   actually help, or is it inert (D4-equivalent from
   `direction-predictor-mechanism/purpose.md`)?

## Hard constraints (same as rounds 1–4)

- Train days: 2026-04-06, 2026-04-08, 2026-04-09.
- Eval days: 2026-04-10, 2026-04-17, 2026-04-21, 2026-05-03, 2026-05-06.
- 4 agents × 1 generation (except Group 4 which varies generations).
- All 3 predictors loaded; lean obs.
- Frozen C11 direction head loaded.
- Policy-side direction gate OFF (D-cells decided).
- `close_feasibility_max_spread_pct=0.05`,
  `matured_arb_expected_random=0.0`,
  `force_close_before_off_seconds=120`.

## Base recipe (filled in once F3/F3b lands)

The `BASE_RECIPE` (cohort-wide flag list) is set in `run_round5.sh`
after the Phase B results are interpreted. Plausible values:

- `--predictor-p-win-back-threshold 0.20`
- `--bc-pretrain-steps 500`
- `--bc-include-negative-samples`
- `--bc-include-close-hold-samples`
- `--bc-positive-weight X` (X ∈ {1.0, 2.0} from Phase B winner)

## Experiments

7 groups, ~25 cells total. Independent — each group can be run
in any order. Total estimated wall ~10-12h on a single GPU at
~25 min per single-gen cell.

### Group 1 — Robustness across seeds (5 cells, ~2.1h)

Re-run the base recipe with 4 additional seeds plus a re-run of
seed 42 to confirm intra-cohort reproducibility. n=20 effective
agents per "recipe variance" estimate.

| cell | --seed |
|------|-------:|
| R1_seed42_repeat | 42 |
| R1_seed43       | 43 |
| R1_seed44       | 44 |
| R1_seed45       | 45 |
| R1_seed46       | 46 |

**Decision rule:** the base recipe is "robust" if all 5 cells'
day_pnl spreads less than ±£60 and mat% spreads less than ±2pp.

### Group 2 — BC dose-response on augmented pool (4 cells, ~1.7h)

With L2+L3a+L4 in the BC pool, the optimal BC step count may have
shifted from the pre-augmentation 500. Sweep around 500:

| cell | --bc-pretrain-steps |
|------|--------------------:|
| R2_bc100  |  100 |
| R2_bc200  |  200 |
| R2_bc1000 | 1000 |
| R2_bc2000 | 2000 |

(BC=500 covered by Group 1.) Decision: if R2_bc1000 lifts mat%
without hurting cls%, BC dose-response is favourable to longer
training.

### Group 3 — pwin_back threshold sweep (4 cells, ~1.7h)

We've been pinned at 0.20. May be sub-optimal with the augmented
BC pool:

| cell | --predictor-p-win-back-threshold |
|------|---------------------------------:|
| R3_pwin015 | 0.15 |
| R3_pwin025 | 0.25 |
| R3_pwin030 | 0.30 |
| R3_pwin035 | 0.35 |

(pwin=0.20 covered by Group 1.) The expected shape is U-curve:
too tight → fewer opens, miss good ones; too loose → bad opens
slip through. Find the bottom of the U.

### Group 4 — Multi-generation training (3 cells, ~3.1h)

Single-generation results are 1 gen of PPO on the BC-warmstarted
policy. With BC providing strong selection, multi-gen PPO may
unlock further mat%/cls% gains — or may diverge (the failure mode
to watch for).

| cell | --generations | wall |
|------|--------------:|------|
| R4_2gen | 2 |  ~50min |
| R4_3gen | 3 |  ~75min |
| R4_5gen | 5 | ~125min |

If R4_5gen is monotonically better than R4_2gen on day_pnl, we
may want to push to even more gens in a follow-up.

### Group 5 — Direction signal value (2 cells, ~50min)

D4-equivalent owed from `direction-predictor-mechanism/findings.md`.
Tests whether the C11 head's obs signal is load-bearing.

| cell | flags |
|------|-------|
| R5_dir_gain_zero    | (base) + `--direction-signal-gain 0` (mute the 12 direction obs columns) |
| R5_no_direction     | (base) − `--use-direction-predictor` (disable the predictor entirely + free its obs columns) |

**Implementation note:** `--direction-signal-gain` doesn't exist as
a CLI flag yet. Pre-implementation: check if the env exposes a
mechanism to zero specific obs columns; if not, write the smallest
patch needed. If the patch is non-trivial, defer R5_dir_gain_zero
and run only R5_no_direction. Decision: if the no-direction cell
matches the base on day_pnl, the direction predictor's obs signal
isn't earning its place.

### Group 6 — Re-test dropped lay-side levers (3 cells, ~75min)

C3 (lay-pwin) and C4 (race confidence) regressed in Round 1 on the
weak baseline. With Round 5's strong base, the lay-side gates may
finally bite — the "remove bad lays" mechanism could compose with
BC's selection lift.

| cell | flags added to base |
|------|---------------------|
| R6_pwin_lay050 | `--predictor-p-win-lay-threshold 0.50` |
| R6_race_conf035 | `--race-confidence-threshold 0.35` |
| R6_lay_price_max20 | `--lay-price-max 20` |

### Group 7 — Deploy-time safety probes (2 cells, ~50min)

These vary deploy-time concerns:

| cell | flags added to base |
|------|---------------------|
| R7_fc60      | `--reward-overrides force_close_before_off_seconds=60` |
| R7_tight_lock | `--arb-spread-target-lock-pct 0.005` (tighter passive target → more passive fills) |

R7_fc60 tests whether reducing the naked-exposure window further
reduces force-close P&L drag at deploy time. R7_tight_lock tests
whether tighter target lock improves mat% by making more pairs
fill in-window.

## Acceptance and decision rules

Each cell evaluated against the round-3 acceptance bar:

| metric        | target          |
|---------------|-----------------|
| opens/day     | 100–180         |
| mat%          | ≥ 5%            |
| fc%           | ≤ 50%           |
| day_pnl       | > -£100         |
| locked/σ_naked | > 0.5          |

A cell passing 4 of 5 with day_pnl > -£50 = deploy candidate.

After all cells land (~ early afternoon next day), write
`findings.md` and decide on the next cohort or production candidate.

## Out of scope

- Engineering for new label classes (L3b, L5, L6) — out of scope
  unless Phase B's F3/F3b shows L3a is insufficient (e.g., mat%
  drops because L3a is too aggressive). Then a Phase B' might
  upgrade to L3b.
- Larger cohort scale (12+ agents, multiple generations) — that's
  the **production cohort** post-Round 5, not a probe.
- Gradient knob pinning (`learning_rate`, `gae_lambda`) — still
  owed from Round 3, still requires new CLI flags. Independent
  engineering effort.
- Bigger train-day set — keeping 3 days for direct comparability.
  Larger train sets are deploy-stage decisions.

## Estimated wall time

~10-12h sequential. Leaves ~6h headroom in the 18h budget for
follow-up cells driven by what the data shows.
