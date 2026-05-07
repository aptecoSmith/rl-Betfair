---
plan: rewrite/phase-14-direction-gate
parent_purpose: ./purpose.md
---

# Hard constraints

Load-bearing invariants. Each constraint is followed by the failure
mode it prevents.

## Architecture / actor wiring (S01)

§1 **Per-runner direction head architecture.**
   `direction_prob_head` becomes a small MLP that operates per-slot
   on `(slot_emb, lstm_last)` and emits 2 logits per slot
   `(direction_back_logit, direction_lay_logit)`. Mirrors
   `actor_head`'s pattern. The single
   `Linear(hidden, max_runners*2)` is REMOVED.
   *Failure mode:* keeping the single Linear keeps the failure mode
   that produced phase-13's NULL. The probe established this is
   THE bottleneck; keeping the old shape repeats the experiment.

§2 **Architecture-hash break protocol.** The new
   `direction_prob_head.0.weight` shape (per-runner MLP) is
   incompatible with phase-13's `Linear(hidden, max_runners*2)`.
   `load_state_dict(strict=True)` refuses pre-S01 checkpoints. No
   new explicit version field — the weight shape carries the
   variant identity (precedent: `fill-prob-in-actor` /
   `mature-prob-in-actor` / `direction-prob-in-actor`).
   *Failure mode:* silent shape coercion produces garbled head
   outputs.

§3 **Gradient flow through `direction_prob_head` from
   `actor_head`.** The sigmoid output continues to feed
   `actor_input`. Do NOT detach. Same precedent as
   `fill_prob_head` / `mature_prob_head`.
   *Failure mode:* detaching kills the alpha pathway through the
   actor; head trains via BCE but policy can't act on it.

§4 **Default `direction_prob_loss_weight = 0.0` is byte-identical
   to phase-13.** The architecture change in S01 alone changes the
   weight-shape fingerprint, but with weight 0 the head's outputs
   are still produced (near-`sigmoid(0) ≈ 0.5` constant) and the
   actor_input columns stay benign. PPO loss is unchanged from
   phase-13 at weight 0.
   *Failure mode:* breaking byte-identity at the disabled config
   makes pre-S01 cohort comparisons impossible.

## Feature extension (S02)

§5 **`OBS_SCHEMA_VERSION` bump.** Adding 8 features to
   `RUNNER_KEYS` (RUNNER_DIM 115 → 123) increases obs dim. Bump
   `OBS_SCHEMA_VERSION` from 6 to 7. Old obs-schema checkpoints,
   oracle caches, and direction-label caches all refuse load —
   correct-by-default behaviour.
   *Failure mode:* silent zero-padding of the new feature columns
   on old checkpoints corrupts the policy's input.

§6 **Cache regeneration is mandatory before S04 cohort.** The
   oracle cache (`data/oracle_cache_v2/`) and direction-label cache
   (`data/direction_labels/`) both carry `obs_schema_version` in
   their headers. After S02 lands, both must be re-scanned via
   their respective CLIs before any cohort launch.
   *Failure mode:* cohort training trips a `FileNotFoundError` /
   `ValueError` mid-run. Acceptable but wastes wall time.

§7 **`engineer_day` outputs the 8 new features at every priceable
   tick.** No silent zero-fill on edge cases (e.g. tick_idx < 30
   for `ltp_velocity_30`). When the lookback is unavailable, emit
   the feature with value 0.0 explicitly — the lookback availability
   itself is implicitly encoded by other obs features (e.g.
   `seconds_since_last_tick`). Document in the engineer's docstring.
   *Failure mode:* sentinel NaN values flow into the policy and
   produce gradient-NaN training crashes.

§8 **TradedVolumeLadder defensive fallback.** Some snap_json rows
   don't carry the ladder (~15% per the probe). When absent, the 4
   ladder-derived features are set to 0.0 (same as the no-data
   case). This is the SAME convention `RUNNER_KEYS` already uses
   for missing values.
   *Failure mode:* engineer_day raises on missing-ladder rows and
   crashes the cohort.

§9 **`RunnerSnap.traded_volume_ladder` as a new field.** The
   episode_builder gains a new optional field on `RunnerSnap`
   (default `[]`). The parquet ingest path populates it from
   `snap_json["MarketRunners"][k]["Prices"]["TradedVolumeLadder"]`.
   Existing callers that don't read the field are unaffected.
   *Failure mode:* re-walking snap_json downstream of
   episode_builder duplicates parse cost.

## Hard-mask gate (S03)

§10 **`direction_gate_threshold` is a per-agent gene clamped to
    [0.5, 0.95].** The lower bound is the gate-disabled semantic
    (at 0.5 with positive-class density ~22% the gate barely
    filters). The upper bound prevents an agent drawing 0.99+ that
    never opens (which would starve PPO).
    *Failure mode:* an agent draws 0.99+, opens 0 pairs, PPO has
    no reward gradient, agent collapses to an inert NOOP-only
    policy.

§11 **The mask is applied INSIDE `DiscreteLSTMPolicy.forward`.**
    The policy reads its own `direction_back_prob` /
    `direction_lay_prob` outputs and masks `OPEN_BACK_i` /
    `OPEN_LAY_i` logits where `max(P_back_i, P_lay_i) <
    threshold`. The env / shim does NOT reach into the policy's
    internals.
    *Failure mode:* the env tries to invoke a partial policy
    forward to compute the mask, coupling env to policy
    architecture.

§12 **Mask is multiplicative on logits, not on the categorical
    sample.** Add `-inf` to masked logits before
    `Categorical(logits=...)`. PyTorch's softmax handles `-inf`
    cleanly (zero probability at masked positions; non-masked
    positions normalise correctly).
    *Failure mode:* post-hoc rejection (sample, then re-sample if
    masked) introduces sampler bias the PPO update can't account
    for.

§13 **Mask interacts cleanly with the existing legality mask.**
    `_apply_mask` already supports a `mask: torch.Tensor` argument
    (the env-side legality mask). The direction-gate mask AND-s
    with that — both legality AND direction confidence must pass
    for an `OPEN_*` action to remain legal.
    *Failure mode:* the new mask overrides legality and the policy
    samples illegal actions (e.g. opens at a tick where action is
    blocked by env-side rules).

§14 **NOOP action stays legal regardless of threshold.** The mask
    NEVER touches the NOOP logit. An agent at threshold 0.95 with
    no high-confidence runners just emits NOOP — that's the
    selectivity working.
    *Failure mode:* masking NOOP means the policy MUST act every
    tick, defeating the gate.

§15 **CLOSE_i actions are NOT gated.** The direction gate
    selectively GATES OPENS. CLOSE actions reduce existing
    positions and are needed for risk management; gating them
    would trap the agent in losing positions when direction
    confidence is low.
    *Failure mode:* mid-race the agent's direction confidence
    drops below threshold, all CLOSE actions are blocked, the
    agent can't bail out of a losing pair before force-close.

§16 **`direction_gate_threshold = 0.5` is byte-identical to no-gate
    AT THE COHORT BASELINE.** With positive-class density 0.22 and
    a fresh-init policy outputting ~sigmoid(0)=0.5 ± noise on
    direction logits, ~50% of rows have `max(P_back, P_lay) ≥ 0.5`.
    So 0.5 IS a gate, not a no-op. **The byte-identical default
    is achieved via a separate `direction_gate_enabled: bool`
    flag, defaulting False.** The threshold gene only takes effect
    when the flag is True. With the flag False the policy's mask
    path is byte-identical to phase-13.
    *Failure mode:* cohort runs without explicit opt-in inherit a
    silent gate that changes their training dynamics, breaking
    backward comparison with phase-13 baselines.

## Cohort & validation (S04)

§17 **Two-arm cohort design.** Arm A: gate-disabled
    (`direction_gate_enabled=False`, equivalent to phase-13
    architecture-with-fix-from-S01-S02 baseline). Arm B:
    gate-enabled with `direction_gate_threshold` evolved as a
    Phase 5 gene in [0.5, 0.95]. Both arms have S01 + S02 active
    and `direction_prob_loss_weight = 0.1`.
    *Failure mode:* mixing the architecture/feature changes with
    the gate change in a single arm makes the gate's effect
    unmeasurable.

§18 **Multi-day eval is mandatory.** Per phase-13 S06's lessons,
    `--n-eval-days` must be ≥ 3 to average down per-day naked-luck
    variance. Spec-spec'd eval is 3 held-out days at the end of
    the day window.
    *Failure mode:* single-day eval, ±£600 day-pnl spread on
    identical-gene agents (per phase-13 lesson) drowns the
    intervention's signal.

§19 **Pre-cohort smoke probe.** Before launching the validation
    cohort, run a 1-2 agent × 1-gen smoke to confirm:
    (a) policy constructs (architecture-hash break works),
    (b) `direction_back_bce_mean` is non-zero with weight 0.1,
    (c) gate-on agent opens > 50 pairs/day,
    (d) no shape mismatches in PPO update.
    *Failure mode:* a wiring bug eats the full cohort's wall time
    before surfacing.

§20 **Cohort runs on GPU.** `--device cuda`. Per the phase-13
    `feedback_always_gpu.md` memory note. CPU is a foot-gun.
    *Failure mode:* 10× wall-time penalty on a 12 × 4 cohort.

## Cross-cutting

§21 **All probe-derived numbers are reproducible.** The probes
    (`tools/direction_*_probe*.py`) stay in the repo as the
    artefact behind this plan's claims. If S04 produces results
    that diverge from the probe predictions, the disagreement
    itself is a signal — re-run the probes against the cohort's
    obs to see where the gap is.
    *Failure mode:* probes thrown away after this plan, no way to
    audit the strategic claims later.

§22 **Empirical per-pair P&L is in lessons_learnt.md.** The
    £3.37 / £1.80 / 34.8% break-even numbers come from the n=92
    cohort scoreboards processed by
    `tools/cohort_per_pair_pnl_summary.py`. They live in
    `lessons_learnt.md` and are re-derivable. They are NOT
    hard-coded into env or trainer config.
    *Failure mode:* env starts assuming a cost ratio it doesn't
    realise from the actual matcher.

§23 **No new policy classes.** All changes land on
    `DiscreteLSTMPolicy`. v1 policies (`agents/policy_network.py`)
    are unchanged. Phase 14 ships v2-only.
    *Failure mode:* parallel-tree maintenance burden on a v1 stack
    no current cohort uses.
