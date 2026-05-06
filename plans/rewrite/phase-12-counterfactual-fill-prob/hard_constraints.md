---
plan: rewrite/phase-12-counterfactual-fill-prob
---

# Hard constraints

§1  **Offline only.** The label generator never imported or
    invoked inside the training loop. Same posture as the arb
    oracle — any code path that could reach the scan during a
    live training run is a blocking bug.

§2  **Deterministic.** Same date + same config + same matcher
    rules → byte-identical `.npz` output. Sort labels by
    `(tick_index, runner_idx, side)` before writing.

§3  **Env-matcher consistent.** A label of 1.0 must require that
    the env's `ExchangeMatcher` would actually accept the
    corresponding open at tick T. Junk filter, price cap,
    `MIN_BET_STAKE` budget, and `max_back_price` all apply
    identically to how the live matcher treats them.

§4  **Cache header carries the full label-generation contract.**
    Header.json must include:
    - `obs_schema_version`
    - `arb_spread_ticks` (the spread used to derive P_lay / P_back)
    - `force_close_before_off_seconds`
    - `max_price_deviation_pct` (junk filter)
    - `max_back_price` / `max_lay_price` caps
    - `commission` (informational; doesn't affect fill, only
      eventual P&L if matched)
    - The label semantics version ("v1_price_reachable" or
      "v2_queue_aware").

    `load_labels(strict=True)` raises `ValueError` on any
    mismatch.

§5  **fill_prob_head label source is mutually exclusive.** Once
    Phase 12 ships, the BCE loss reads from offline labels only.
    The previous "agent rollout outcome" label path is removed,
    not blended. Mixing two label sources on one head produces
    ambiguous gradients.

§6  **mature_prob_head is untouched.** Phase 9's per-transition
    credit still drives that head. Phase 12 owns fill_prob only.
    The two heads answer different questions (will the passive
    fill? vs will the pair lock profit favourably?) and need
    independent label paths.

§7  **V1 conservative label.** First version uses
    `best_back_at_tick_t ≤ P_lay` (price-reachable). Realistic
    queue-position + traded-volume fill simulation is V2.
    Discipline: no V2 work until V1 passes its S03 gate.

§8  **Architecture-hash break is permitted.** Widening
    `fill_prob_head` outputs to `(max_runners × 2)` for per-side
    prediction breaks `load_state_dict(strict=True)` against
    pre-Phase-12 weights. Acceptable — same precedent as
    Phase 7's actor_head extensions. Document the break in
    lessons_learnt.

§9  **Performance budget.** Scan-day must complete in ≤ 5 minutes
    per day on CPU. Typical day = ~9000 pre-race ticks ×
    ~14 runners × 2 sides = ~250k labels. If S01 measures > 5
    min/day, stop and profile before continuing — likely needs
    a vectorised forward-pass over the price ladders rather than
    a per-label inner loop.

§10 **Cache regenerates on any matcher / spread change.** Changing
    `arb_spread_ticks`, `force_close_before_off_seconds`, the
    junk-filter %, or the matcher's price cap invalidates the
    cache. The header check at load time enforces this — runs
    that try to use stale labels hard-fail.

§11 **Class imbalance is acknowledged.** Most labels (≥ 75 %)
    will be 0. The BCE loss must use a positive-class weight
    (`pos_weight = N_neg / N_pos`) computed per cache; otherwise
    the head collapses to "always predict 0" within the first
    few mini-batches and the actor never sees useful signal.

§12 **Per-cohort cache regen for arb_spread sweeps.** If a cohort
    pins `arb_spread_scale ≠ 1.0`, its labels must be generated
    for the corresponding `arb_spread_ticks`. Mixing cohorts
    with different spreads on one cache is forbidden by §4 (the
    header check rejects it).
