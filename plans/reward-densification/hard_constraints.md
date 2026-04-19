# Hard constraints — Reward Densification

Non-negotiable rules. Anything that violates one gets rejected
in review before destabilising the next training run.

## Scope

**§1** This plan makes three coordinated changes, described
in `purpose.md`:

1. Per-step mark-to-market P&L shaping in the env: compute
   open-position MTM at each tick, emit the Δ between ticks
   as a shaped-reward contribution.
2. Training-plan redraft: new plan JSON in
   `registry/training_plans/` tuned for the 9-agent 1-gen
   probe pattern, identical shape to
   `fill-prob-aux-probe` + the new `mark_to_market_weight`
   override.
3. Validation launch: operator-gated, writes a Validation
   entry into `progress.md`.

Anything NOT in that list is out of scope. Examples
explicitly out of scope: controller changes, matcher
changes, action/obs schema bumps, gene-range edits (the
GA-gene step is a follow-on session OR a follow-on plan;
this plan ships with a plan-level default only), PPO
stability reworks (clamps / KL thresholds / warmup),
aux-head reweighting, smoke-gate reworks.

**§2** No changes to matcher semantics. `env/exchange_matcher.py`
stays single-price, no-walking, LTP-filtered. The MTM
calculation reads LTP but never places hypothetical hedges
against the ladder (would reintroduce phantom-profit risk).

**§3** No changes to PPO numerical stability defences. Ratio
clamp ±5 stays. KL early-stop 0.03 stays. Per-arch LR stays.
5-update LR warmup stays. Advantage normalisation stays.
Reward centering EMA stays.

**§4** The `entropy-control-v2` target-entropy controller
stays wired in, targeting 150, using SGD proportional
control with `alpha_lr=1e-2`. The Validation entry from that
plan concluded the controller mechanism was working correctly;
nothing to revert.

## Mark-to-market semantics

**§5** Mark-to-market is computed per-open-bet using the
runner's current LTP. When the LTP is absent (the runner is
unpriceable per CLAUDE.md's matcher rule), that bet's MTM
contribution for the current tick is zero. Ditto for resolved
bets that aren't yet settled but no longer have ladder
exposure (e.g. cancelled, withdrawn).

**§6** MTM formula for an open back bet of matched stake `S`
at average matched price `P_matched`, when the runner's LTP
is `P_current`:

    mtm_back = S * (P_matched − P_current) / P_current

Interpretation: positive when LTP has fallen below the
matched price (market agrees the runner is less likely to
win than when we backed it — good for a backer). Reduces to
zero when `P_current == P_matched`. Commission is NOT
deducted at MTM time — it's applied at realised settle, so
adding it to MTM would double-count. The `S * (... / P_current)`
form is the exchange-value formula, not the "if I closed
right now what would I lock" formula; v1 uses the simpler
form as it preserves the raw/shaped invariant cleanly.

**§7** MTM formula for an open lay bet of matched stake `S`
at average matched price `P_matched`, when the runner's LTP
is `P_current`:

    mtm_lay = S * (P_current − P_matched) / P_current

Symmetric to the back formula with sign flipped. Same
commission and unpriceable-runner rules as §6.

**§8** Portfolio MTM at tick `t` is the sum across all open
bets:

    MTM_t = sum(mtm_back) + sum(mtm_lay)

The per-step shaped contribution is:

    shaped_mtm_t = mark_to_market_weight * (MTM_t − MTM_{t-1})

with `MTM_0 = 0` by convention (nothing open at the start of
a race). This delta form has the telescoping property:
cumulative `sum(shaped_mtm)` over a race equals
`mark_to_market_weight × MTM_final`. When all positions
resolve at settle, `MTM_final = 0` (resolved bets don't
contribute to MTM), and the cumulative shaped contribution
is exactly zero. Together with the raw P&L arriving at
settle, `raw + shaped` is unchanged per race (the shaping
only redistributes through time).

**§9** When a bet resolves (settles, or is actively closed via
`close_signal`), its contribution to `MTM_t` drops to zero
immediately. That drop IS the last mark-to-market delta the
shaped term sees for this bet — which MUST equal the
negative of the bet's last unrealised P&L so the telescope
closes. The raw P&L (win − stake, or stake × (price − 1),
etc.) arrives simultaneously on the same step. No
double-counting because MTM and realised P&L are
accumulated into different buckets (`shaped_bonus` and
`raw_pnl_reward`).

## Knob and default

**§10** New reward-config key: `mark_to_market_weight` on the
`BetfairEnv` config. Default `0.0`. When `0.0`, the entire
per-step MTM computation is a no-op; episodes are
byte-identical to pre-landing behaviour. Same
zero-by-default migration pattern as Session 1 of
`arb-improvements`.

**§11** `mark_to_market_weight` is NOT added to GA
`hp_ranges` in this plan. Session 02 lands a plan-level
default (likely `0.05` per purpose.md); gene-ing it is
a follow-on plan if/when the mechanism is proven.

## Telemetry and invariant

**§12** The existing
`test_invariant_raw_plus_shaped_equals_total_reward`
integration test MUST stay green. The MTM shaping lives
entirely in `shaped_bonus`; raw P&L accounting is
untouched. Any test failure here is a blocker — fix
before landing, not in validation.

**§13** Per-episode JSONL rows gain optional fields
(entropy-control-v2 pattern):

- `mtm_weight_active: float` — the weight the env used this
  episode (plan-level, not per-step).
- `cumulative_mtm_shaped: float` — the total
  `sum(shaped_mtm_t)` across the episode. Should equal
  `weight × MTM_final = 0` at settle (within floating-point
  tolerance); surfacing this value lets us catch telescope-
  break bugs in production without a regression test.

Downstream JSONL readers must tolerate absence on pre-change
rows (same backward-compat pattern as `alpha` / `log_alpha` in
entropy-control-v2).

**§14** `info` dict on every env step gains an optional
`mtm_delta: float` field carrying that step's
`(MTM_t − MTM_{t-1})` BEFORE weight is applied. For
diagnostics / learning-curves panel. When weight is 0, this
field still reports the raw delta (agents can be debugged
even in no-op mode).

## Testing

**§15** Each session commit ships with new tests. Full
`pytest tests/ -q` MUST be green on every session commit.

**§16** Session 01 (scaffolding) tests:

- `test_mark_to_market_weight_default_is_zero` — fresh env
  from default config has weight 0.
- `test_mtm_delta_zero_when_no_open_bets` — reward path is
  byte-identical to pre-change when no bets held.
- `test_mtm_back_formula_matches_spec` — spec-level unit
  test on the `mtm_back` closed-form.
- `test_mtm_lay_formula_matches_spec` — ditto for lay.
- `test_mtm_zero_when_ltp_missing` — no LTP → no shaping
  contribution (§5 guarantee).
- `test_mtm_telescopes_to_zero_at_settle` — construct a
  scripted race: open a back bet, move LTP around for N
  ticks, settle. Cumulative `shaped_mtm` across the race
  must be zero (to floating-point tolerance).
- `test_invariant_raw_plus_shaped_with_nonzero_weight`
  (NEW): run a full scripted rollout with `weight=0.05`,
  assert `raw + shaped ≈ total` holds per episode.
- `test_mtm_weight_zero_byte_identical_rollout` — diff two
  rollouts with weight 0 vs the old env (pre-change): per-
  episode reward must be identical to float-eps.
- `test_info_mtm_delta_field_present` — env step info dict
  has `mtm_delta` key when any open position exists.

**§17** Session 02 (default weight) tests:

- `test_mark_to_market_weight_default_matches_session_02` —
  whatever value we pick, pin it so a future refactor
  can't silently revert.

**§18** Session 03 (registry / plan redraft) is non-code.
No new tests required; correctness is verified by the
Validation entry post-launch.

## Reward-scale change protocol

**§19** This IS a reward-scale change for any run where
`mark_to_market_weight > 0`. CLAUDE.md "Reward function: raw
vs shaped" gets a new dated paragraph. Scoreboard rows from
pre-change runs and post-change weight>0 runs are NOT
directly comparable on per-episode reward magnitudes —
though per-race `raw_pnl_reward` (race-settled P&L only)
stays comparable because the shaping lives entirely in
`shaped_bonus`.

**§20** For weight=0 migrations (the Session 01 default),
runs remain byte-identical to pre-change. The scoreboard
rule above applies only when the knob is turned up.

## Cross-session

**§21** Sessions land as separate commits, in order 01 → 03.
Session 03 (plan redraft + launch) is a **manual operator
step**, NOT something an agent runs autonomously. The plan
folder's Session 03 prompt is instructional; execution is
operator-gated. Same rule as
`naked-clip-and-stability` Session 05 /
`entropy-control-v2` Session 03.

**§22** If Session 01 fails (invariant test red, any
MTM-math test red), Session 02 does not start. The knob is
untuned until the mechanism is correct.

**§23** Do NOT bundle the re-launch into the Session 03
commit. Session 03 is "redraft plan + docs"; the launch is
a follow-on operator action that writes back into
`progress.md` as a Validation entry.

**§24** Archive artefacts from the `fill-prob-aux-probe`
2026-04-19 run stay available for comparison:

- `logs/training/episodes.jsonl` snapshot at plan start —
  archive to
  `logs/training/episodes.pre-reward-densification-<isodate>.jsonl`
  before Session 03 launches.
- The `fill-prob-aux-probe` training-plan JSON itself stays
  in the registry for reference (status `completed`).

Post-mortem comparison at validation time diffs the reward-
densification run's entropy / alpha / arbs-ratio / reward
trajectories against the A-baseline and fill-prob-probe
baselines documented in `entropy-control-v2/progress.md`.
