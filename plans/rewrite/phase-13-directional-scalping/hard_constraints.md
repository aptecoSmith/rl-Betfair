---
plan: rewrite/phase-13-directional-scalping
parent_purpose: ./purpose.md
---

# Hard constraints

These are load-bearing invariants. Violating any of them invalidates a
session's deliverable. Each constraint is followed by the failure mode
it prevents.

## Label generation (S02)

§1 **Offline only.** The direction-label generator never runs inside
   the training loop. Labels are produced by an offline CLI and
   loaded from disk at trainer init. Same precedent as
   `training_v2/arb_oracle.py` and phase-12's
   `training_v2/fill_label_scan.py`.
   *Failure mode:* on-line label generation introduces a per-tick
   compute cost on every cohort run, defeats the determinism
   guarantee, and couples the trainer to data-loading paths that
   should stay isolated.

§2 **Determinism.** Same data + same config → byte-identical labels.
   Sort labels by `(tick_index, runner_idx)` before writing.
   `header.json` carries every config key that affects the label
   value (`obs_schema_version`, `direction_horizon_ticks`,
   `direction_threshold_ticks`, `force_close_before_off_seconds`,
   junk-filter %, price caps, label-semantics-version).
   `load_labels(strict=True)` raises `ValueError` on any mismatch.
   *Failure mode:* labels silently shifting under config changes
   produces uninterpretable cohort comparisons.

§3 **Match env-matcher priceability rules at the OPEN tick.** A
   label of `1.0` requires that the env's `ExchangeMatcher` would
   have accepted the corresponding open at the open tick (junk
   filter, price cap, minimum stake budget). Otherwise the label
   teaches the actor about opens it cannot make.
   *Failure mode:* the head learns to predict price moves on
   un-openable runners, contaminating its actor_head input column.

§4 **V1 label semantics: threshold-crossing on `last_traded_price`.**
   The label is binary: did `ltp` cross the favourable threshold
   within the close horizon, NOT the magnitude of the move.
   `direction_horizon_ticks` and `direction_threshold_ticks` are
   the only label-defining knobs; both go in `header.json`.
   Magnitude-target labels (V2) require a separate cache namespace.
   *Failure mode:* mixing semantics in the same cache file leads to
   silent corruption when the loss function changes.

§5 **Cache filename embeds the invalidating keys.** Path:
   `data/direction_labels/{date}/horizon{H}_thresh{T}_fc{F}.npz`
   plus a sibling `_header.json`. Different config tuples coexist
   on disk. Filename mismatch is caught by the path resolver before
   `load_labels` is even called.
   *Failure mode:* shared-name cache collisions across cohort
   variants.

## Architecture / actor wiring (S03)

§6 **Architecture-hash break.** Adding two per-runner columns
   (P(direction_back), P(direction_lay)) to `actor_input` changes
   `actor_head[0].weight.shape[1]` by `+2`. This breaks
   `load_state_dict(..., strict=True)` against pre-plan checkpoints,
   exactly as `fill_prob_in_actor` and `mature_prob_in_actor` did.
   The shape mismatch IS the variant identity — no new explicit
   version field is added; the existing weight-shape check carries
   the contract.
   *Failure mode:* silent truncation of input columns produces
   garbled actions. Strict load is the correct-by-default behaviour.

§7 **Gradient flows through `direction_prob_head` from
   `actor_head`.** Do NOT detach the sigmoid output. Same
   precedent as `fill_prob_head` and `mature_prob_head` post the
   `fill_prob_in_actor` / `mature_prob_in_actor` plans. The actor
   may learn discriminative direction features beyond what the BCE
   target alone supervises; this is desirable.
   *Failure mode:* detaching kills the alpha pathway through the
   actor. The head still trains via BCE but the policy can't
   integrate its output.

§8 **Default `direction_prob_loss_weight = 0.0` is byte-identical
   to pre-plan.** With weight 0 there is no BCE auxiliary term
   contributing to the trainer's loss, the head still runs in the
   forward pass (so its near-`sigmoid(0) = 0.5` near-constant
   output column lands in actor_input), and the policy outputs
   match pre-plan to floating-point tolerance on a fresh-init
   network. Same precedent as fill-prob / mature-prob default.
   *Failure mode:* breaking byte-identity on default config makes
   pre-plan probe rows non-comparable.

§9 **The two extra columns are PER-SIDE, not aggregated.**
   `actor_input` gains exactly `+2` per-runner dims, in the order
   `[..., fill_prob, mature_prob, direction_back, direction_lay]`.
   `runner_embed_dim + lstm_hidden + 4` is the new
   `actor_head[0].weight.shape[1]` (or `+ d_model + 4` for the
   transformer). Aggregating to a single direction scalar
   (`max(back, lay)`) loses the side-discrimination signal that
   makes back-first vs lay-first different actions.
   *Failure mode:* a single direction scalar gives the policy no
   way to represent "I think the price will rise" vs "I think the
   price will fall" as different decisions.

§10 **All three policy classes get the head.** `DiscreteLSTMPolicy`
    (v2), and the v1 `PPOLSTMPolicy` / `PPOTimeLSTMPolicy` /
    `PPOTransformerPolicy` if v1 is still in active use at the time
    S03 lands. The head structure (`nn.Linear(hidden,
    max_runners × 2)` + sigmoid) is identical across classes. No
    gene-gating; the head is unconditionally present (default
    weight 0 covers opt-in).
    *Failure mode:* policy-class divergence makes cohort comparisons
    across architectures invalid.

§11 **BCE label sourcing.** The trainer reads direction labels from
    the offline cache only — same pattern as phase-12 S02. Per-
    transition cached label is keyed by `(date, race_idx,
    tick_index)` and the trainer raises a clear error if the cache
    is missing for a training day.
    *Failure mode:* on-the-fly label computation inside the trainer
    breaks determinism and re-introduces the on-line-label
    coupling §1 prevents.

## Reward / accounting

§12 **No raw-reward changes in S03 / S05.** The direction head and
    direction-targeted BC affect ONLY the actor-input pathway and
    the BC pretrain pathway respectively. `race_pnl`,
    `scalping_locked_pnl`, `scalping_closed_pnl`,
    `scalping_force_closed_pnl`, `naked_pnl`, and the shaped
    accumulators are unchanged. The `raw + shaped ≈ total_reward`
    invariant holds.
    *Failure mode:* reward-shape drift makes `raw_pnl_reward`
    non-comparable across plans, breaking the operator's primary
    cross-cohort gauge.

§13 **S04 is opt-in.** The MTM-loss stop-loss adds either an env
    trigger or a new action dim. Its activation knob
    (`mtm_stop_loss_threshold = 0.0` for env trigger, or the new
    action dim's gating gene for the action variant) defaults to
    DISABLED, with the disabled path byte-identical to pre-plan.
    *Failure mode:* a stop-loss that fires by default re-shapes
    every cohort run silently and invalidates all baseline rows.

§14 **S04 stop-loss accounting.** Closes triggered by the stop-loss
    flow through `_attempt_close` like any other close. The bet
    flag is `stop_loss_close = True` (NEW, distinct from
    `force_close = True`) so settlement routes the pair into a new
    `scalping_arbs_stop_loss_closed` counter. The agent did not
    *actively choose* the moment in the env-trigger variant, so —
    matching the force-close precedent in CLAUDE.md — the matured-
    arb bonus and the `+£1 close_signal` shaped bonus BOTH exclude
    stop-loss closes. (If S04 picks the action-dim variant, the
    agent IS choosing and the shaped bonus DOES apply — this is a
    decision point inside S04.)
    *Failure mode:* leaking stop-loss closes into matured/closed
    counters double-counts them in the matured-arb bonus and the
    operator's lifecycle telemetry.

## BC pretrain (S05)

§15 **Direction BC layers with the oracle BC; does not replace.**
    The trainer's BC pretrain step receives BOTH targets: the
    arb-oracle target from phase 8 AND the direction target from
    S05. Mixing weight is a per-agent gene
    (`bc_direction_target_weight`, default 0.0 = oracle-only =
    byte-identical to phase-8). With weight > 0, the BC loss is
    `(1 − w) × bce_oracle + w × bce_direction`.
    *Failure mode:* replacing the oracle target loses the arb-spread
    signal; the post-BC policy then has direction confidence but no
    sense of "is there an arb opportunity here at all".

§16 **BC pretrain is per-agent, never shared.** Inherited from
    phase-8 hard_constraints (which inherits from the
    `arb-improvements` lessons). Sharing BC weights collapses GA
    diversity.
    *Failure mode:* the population converges on identical post-BC
    initialisations and the GA loses its exploration.

## Validation (S06)

§17 **Cohort comparisons require matched config tuples.** The
    direction-on arm and direction-off arm must use the same
    `arb_spread_ticks`, `force_close_before_off_seconds`,
    `mark_to_market_weight`, training days, and seed schedule.
    Only `direction_prob_loss_weight` differs.
    *Failure mode:* unmatched config produces a Goodhart-style "the
    knob did something" conclusion that doesn't isolate the
    intervention.

§18 **The success gate is force-close rate, not P&L.** Plan-level
    success is "force-close rate drops by ≥ 5 pp on the direction-on
    arm". Raw P&L is a NON-REGRESSION check (must not drop by more
    than 10 %) but is not the win condition. Profitability is a
    sequel question.
    *Failure mode:* selecting on P&L over a 3-gen cohort cherry-
    picks runs whose noise is favourable; force-close rate is the
    cleaner signal at this scope.

§19 **If S03 trains but S06 fails the force-close gate, escalate to
    a follow-on plan, not a parameter sweep.** A trained head whose
    output the policy ignores is a representational / architectural
    issue (precedent: `selective-open-shaping` Sessions 03–04). The
    response is to think harder about what's missing — adding stop-
    loss (S04), running the magnitude-target V2 head, or revisiting
    the feature audit findings — not to sweep
    `direction_prob_loss_weight`.
    *Failure mode:* parameter sweeps on a saturated mechanism waste
    cohort budget.

## Cross-cutting

§20 **Pre-plan checkpoints fail to load against post-plan policies
    by design.** The architecture-hash check is the load-bearing
    test that catches operator error. Lessons from
    `plans/fill-prob-in-actor` and `plans/mature-prob-in-actor`
    apply directly. Document the break in
    `agents_v2/discrete_policy.py`'s docstring and add a regression
    test in `tests/test_policy_network.py::TestDirectionProbInActor`
    that asserts pre-plan weights raise on
    `load_state_dict(..., strict=True)`.

§21 **Reward-scale rows from pre-plan runs remain comparable on
    `raw_pnl_reward`.** They are NOT comparable on `total_reward`
    if S04's stop-loss is active (cash flows through stop-loss
    closes change `race_pnl`'s composition, even though the
    accounting category is new). Document the comparability
    boundary in the validation cohort write-up.

§22 **No silent fallbacks.** If the offline direction-label cache
    is missing for a training day and `direction_prob_loss_weight
    > 0`, the trainer raises a clear `FileNotFoundError` with the
    expected cache path. Do NOT default to "skip the BCE term for
    this day" — silent skips corrupt the head's training signal.
