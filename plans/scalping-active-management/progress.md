# Progress — Scalping Active Management

One entry per completed session. Most recent at the top.

---

## Session 02 — Fill-probability aux head (2026-04-16)

**Landed.**

- Every policy architecture grows a `fill_prob_head = nn.Linear(backbone,
  max_runners)`: `PPOLSTMPolicy` / `PPOTimeLSTMPolicy` take `lstm_last`
  as backbone, `PPOTransformerPolicy` takes the last-tick transformer
  output `out_last`. Sigmoid applied inside `forward`, so
  `PolicyOutput.fill_prob_per_runner` is in `[0, 1]` with a same-shape
  `0.5`-tensor default on the dataclass (stub policies / legacy callers
  that construct `PolicyOutput` positionally keep working). Head init:
  orthogonal gain `0.01` on the weight, zeros on the bias — matches the
  actor-head small-init pattern so predictions start ≈ 0.5 ("unsure").
  The head shares the backbone, NOT the actor head (hard_constraints §8:
  conditions on state, not on sampled action).
- **Capture → attach flow.** In `agents/ppo_trainer.py::_collect_rollout`
  each tick snapshots the per-runner fill-prob prediction into a numpy
  array, walks `info["action_debug"]` for `aggressive_placed=True`
  entries, resolves the slot via a new
  `BetfairEnv.current_runner_to_slot()` accessor, and stamps
  `fill_prob_at_placement` on the newest matching `Bet` in
  `bm.bets`. A `pair_to_transition: dict[pair_id, (tr_idx, slot_idx)]`
  map records which transition owns each pair's label cell. At episode
  end the trainer walks `env.all_settled_bets`, groups by `pair_id`
  (≥2 bets = completed → label `1.0`; 1 bet = naked → label `0.0`),
  and writes the label into `Transition.fill_prob_labels[slot_idx]`.
  Pairs still unresolved (e.g. race aborted) stay NaN so the BCE mask
  rejects them — no fake supervision.
- **Paired passive inherits the aggressive partner's prediction.**
  `PassiveOrderBook.on_tick` at fill-time looks up the matching
  aggressive leg in `bm.bets` by `pair_id` and copies
  `fill_prob_at_placement` into the newly-created passive `Bet`. Per
  hard_constraints §10 the value is the decision-time capture, never
  recomputed later.
- **BCE aux loss in `_ppo_update`.** A new module-level
  `_compute_fill_prob_bce(preds, labels)` helper does masked BCE with
  an ε-clamp (`1e-7`) to avoid `log(0)` on contrived extreme
  predictions; refactored into a helper so the gradient-direction and
  zero-loss-on-perfect-predictions tests can exercise it directly.
  Batch labels are built once at the top of `_ppo_update` from each
  transition's `fill_prob_labels` (NaN-padded to `(n, max_runners)`),
  moved to device alongside `obs_batch`, and sliced by `mb_idx` inside
  the minibatch loop. Aux term is added to the total loss as
  `self.fill_prob_loss_weight * fill_prob_loss`; the per-update mean is
  reported in `loss_info["fill_prob_loss"]` so operators see the head's
  behaviour even when the weight is 0.
- **Plumbing-off default — reward scale unchanged.**
  `fill_prob_loss_weight` defaults to `0.0` (read from `hp` first, then
  `config["reward"]`). With weight `0.0` the aux term contributes
  exactly nothing to the optimised loss. Verified by
  `test_fill_prob_weight_zero_is_noop_on_total_loss`. The `raw + shaped
  ≈ total_reward` invariant still holds — verified by
  `test_raw_plus_shaped_invariant_still_holds_with_aux_loss` which
  injects `fill_prob_loss_weight=0.5` via `reward_overrides` and checks
  the env accumulators are undisturbed.
- **Gene passthrough.** `fill_prob_loss_weight` added to
  `agents/ppo_trainer.py::_REWARD_GENE_MAP` and to
  `env/betfair_env.py::_REWARD_OVERRIDE_KEYS`. Mirrors the `reward_clip`
  precedent: env never reads the key, but whitelisting suppresses the
  "unknown overrides" debug log.
- **Checkpoint back-compat strategy: explicit migration helper.** New
  `migrate_fill_prob_head(state_dict, fresh_policy)` in
  `agents/policy_network.py` injects fresh `fill_prob_head.*` weights
  (cloned off the target policy) into any pre-Session-02 state-dict so
  `load_state_dict(..., strict=True)` succeeds. Chosen over
  `strict=False` because strict=False would silently swallow legitimate
  missing-key errors from unrelated migrations — keeping the session-02
  migration explicit preserves the audit trail.
- **`Bet.fill_prob_at_placement: float | None = None`** added
  (defaults to `None`, so existing positional-arg constructions in tests
  keep working). Mirrored on
  `registry/model_store.py::EvaluationBetRecord`. The parquet writer
  emits the new column; the reader uses a `has_fill_prob` guard in the
  same style as the existing `has_pair_id` check so older files without
  the column load with `fill_prob_at_placement=None`.
  `training/evaluator.py` forwards `bet.fill_prob_at_placement` into
  each `EvaluationBetRecord`.
- **Tests.** `TestFillProbHead` class (12 tests) covering: shape / range
  on all 3 architectures; decision-time stamp onto a placed `Bet`;
  passive-leg inheritance from the aggressive partner; BCE ≈ 0 on
  perfect predictions; BCE gradient sign in both directions;
  `weight=0` noop on total loss; `weight=1` lifts the policy-parameter
  gradient norm relative to `weight=0`; parquet back-compat on missing
  column; parquet roundtrip preserving the value; legacy state-dict
  migration + strict load; raw+shaped invariant with aux weight on;
  BCE exclusion of NaN-labelled samples. Full suite: `pytest tests/ -q`
  → **1955 passed, 7 skipped, 1 xpassed**. Net +12 vs Session 01
  (1943 → 1955).
- **Incidental fix.** `test_gradients_flow_through_actor` in
  `tests/test_policy_network.py` was asserting every non-critic /
  non-log-std parameter receives gradient from `action_mean.sum()`.
  The new `fill_prob_head` is a sibling aux head — by design it
  doesn't receive gradient from actor-mean-only loss
  (hard_constraints §8). Narrowed the filter to skip
  `fill_prob_head.*` with the same pattern used for `critic` and
  `action_log_std`.

**Back-compat summary.**

- Scalping OFF or scalping ON without the aux head enabled: byte-
  identical total loss to Session 01 (`fill_prob_loss_weight` defaults
  to 0.0, BCE term contributes exactly 0). Reward invariant untouched.
- Pre-Session-02 checkpoints: strict-loadable after a single call to
  `migrate_fill_prob_head(state_dict, fresh_policy)`. The
  `fill_prob_head.*` keys are the only addition; all other parameter
  shapes are unchanged.
- Pre-Session-02 parquet files: readable unchanged — the optional
  column returns `None` per hard_constraints §11.

---

## 📋 When you open this plan for the first time

Read in order:

1. `purpose.md` — why this exists, the Gen 1 evidence, the
   four changes.
2. `hard_constraints.md` — 20 non-negotiables (invariant,
   matcher, aux-loss isolation, back-compat).
3. `master_todo.md` — session sequencing.
4. `session_prompt.md` — brief for the immediate next session.

Also re-read `CLAUDE.md`, especially "Order matching:
single-price, no walking" (the matcher is load-bearing) and
"Reward function: raw vs shaped" (the invariant still applies).

---

## Session 01 — Re-quote action + env plumbing (2026-04-16)

**Landed.**

- `SCALPING_ACTIONS_PER_RUNNER` bumped 5 → 6. New dim is
  `requote_signal ∈ [-1, 1]`, read from
  `action[5 * max_runners + slot_idx]` in `_process_action`.
- `SCALPING_POSITION_DIM` bumped 2 → 4. Two new per-runner obs
  features live at offsets +2, +3 of the scalping-extra block:
  - `seconds_since_passive_placed` — elapsed real seconds since
    the paired passive was posted, normalised by that race's
    first/last tick span and clamped to [0, 1].
  - `passive_price_vs_current_ltp_ticks` — signed tick distance
    from current LTP to the resting price, normalised by
    `MAX_ARB_TICKS` and clamped to [-1, 1].
  `PassiveOrder` gained a `placed_time_to_off` field so the
  elapsed computation is drift-free across varying race lengths.
  `_race_durations` is pre-computed once per race in
  `_precompute`, matching the existing static-obs / slot-map
  caching pattern.
- Schemas bumped: `OBS_SCHEMA_VERSION: 5 → 6`,
  `ACTION_SCHEMA_VERSION: 2 → 3`. These invalidate existing
  checkpoints for strict validation, same convention as prior
  schema bumps. A migration helper
  `agents.policy_network.migrate_scalping_action_head` pads the
  actor-head final layer + `action_log_std` for code paths that
  explicitly opt into loading a pre-Session-01 state dict (the
  requote row initialises fresh, existing rows are preserved).
- Re-quote dispatch is a dedicated SECOND PASS over slots after
  the main placement loop so a slot that `continue`'d (no bet
  signal, below-min stake, below min_seconds_before_off) can
  still re-quote if its runner has a paired passive to manage.
  The pass:
    1. Finds the first open paired order for the slot's `sid`.
       If none → no-op with `requote_reason="no_open_passive"`
       (hard_constraints §5: never opens a naked leg).
    2. Computes `arb_ticks` from this tick's `arb_raw` (current
       LTP, not the original fill price).
    3. Computes the new resting price with the same direction
       rule as `_maybe_place_paired` (back → lay below, lay →
       back above).
    4. Applies the junk-filter window explicitly — paired
       `PassiveOrderBook.place` bypasses it for the auto-paired
       path, but an active re-quote sitting outside ±max_dev
       from current LTP IS stale-parked-order risk. On failure
       we cancel the old passive and set
       `requote_reason="junk_band"`, leaving the aggressive
       leg naked for that runner.
    5. Cancels the existing passive via a new
       `PassiveOrderBook.cancel_order(order, reason)` — budget
       reservation is released before the new reservation is
       taken (hard_constraints §6). On Lay-after-Back pairs the
       `place()` freed-budget offset recomputes against the
       aggressive leg still in `bm.bets`, so the net
       `available_budget` change equals the liability delta
       between old and new passive prices.
    6. Re-places via `PassiveOrderBook.place(..., price=...)`
       with the same `pair_id` — ledger continuity is preserved
       and the re-quoted passive, if it fills, shows up as a
       completed pair alongside the aggressive bet.
    7. Records `requote_attempted`, `requote_placed`, and/or
       `requote_failed`/`requote_reason` on
       `action_debug[sid]` for diagnostics.
- Tests: new `TestScalpingRequote` class covering the 11
  scenarios in the session prompt. Existing
  `test_obs_space_grows_when_scalping` updated for the new
  `SCALPING_POSITION_DIM`. Three `test_p1?_*` schema-refusal
  tests had hard-coded `OBS_SCHEMA_VERSION == 5` assertions;
  relaxed to `>= N` so they stay locked on the refusal
  behaviour rather than on a frozen version number.
- Full suite green: `pytest tests/ -q`
  → 1943 passed, 7 skipped, 1 xfailed.

**Back-compat strategy.**

- Scalping OFF: action vector stays 4-per-runner, obs stays
  `SCALPING_POSITION_DIM=0` extras. Byte-identical to pre-
  Session-01. Verified by
  `test_action_space_size_grows` (scalping=False branch).
- Scalping ON with a pre-Session-01 checkpoint: strict
  `validate_action_schema` refuses the version-2 checkpoint.
  Use `migrate_scalping_action_head(state_dict, max_runners,
  old_per_runner=5, new_per_runner=6)` to widen the actor head
  + log-std before `load_state_dict`. Tested by
  `test_legacy_checkpoint_loads`.

**No reward-scale change.** `raw + shaped ≈ total_reward`
invariant verified by the new
`test_raw_plus_shaped_invariant_holds` under active re-quoting.

---

_Plan created 2026-04-16 off the back of the Gen 1 training-run
analysis. The Gen 1 run confirmed the reward-signal fix but
also exposed that even the top scalper only completes 14.5 %
of pair attempts — the rest become accidental directional bets
because the passive never fills. This plan gives the agent the
tools to manage passives actively and to know its own fill
probability and risk._
