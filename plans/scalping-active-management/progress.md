# Progress — Scalping Active Management

One entry per completed session. Most recent at the top.

---

## Session 04 — Bet Explorer confidence + risk badges (2026-04-16)

**Landed.**

UI-only session. Surfaces the per-`Bet` aux-head predictions
from Sessions 02 + 03 in the Bet Explorer as a confidence chip
and a risk tag, one per row, next to the existing pair-class
badge.

**API contract.**

Three optional fields added to `ExplorerBet` in
[api/schemas.py:260](../../api/schemas.py) and forwarded from
`EvaluationBetRecord` in [api/routers/replay.py:139](../../api/routers/replay.py):

- `fill_prob_at_placement: float | None`
- `predicted_locked_pnl_at_placement: float | None`
- `predicted_locked_stddev_at_placement: float | None`

All default to `None` so existing clients are unaffected and
bets written before Session 02 keep returning without these
fields populated. Confirmed by the three new
`TestBetExplorer::test_scalping_aux_head_*` tests in
[tests/test_api_replay.py](../../tests/test_api_replay.py).

**Frontend chip logic.**

Thresholds pinned as named constants in
[frontend/src/app/bet-explorer/bet-explorer.ts](../../frontend/src/app/bet-explorer/bet-explorer.ts):

- `CONFIDENCE_HIGH_THRESHOLD = 0.7` → green "High" chip.
- `CONFIDENCE_MED_THRESHOLD = 0.4` → amber "Med" chip for
  `[0.4, 0.7)`.
- Below `0.4` → red "Low" chip.

**Near-default hide rule.** When
`|fillProbAtPlacement − 0.5| < CONFIDENCE_NEAR_DEFAULT` (=
0.02), the chip is suppressed. This is the untrained-head
fallback: until `activation_playbook.md` Step E lands and the
aux heads are actually receiving gradients, every prediction
is ≈ 0.5 and a chip would be noise. The sunset is explicit:
once Step E completes, trained predictions move out of the
band and the chip starts rendering automatically. No code
change required to flip the switch — it's data-driven.

**Risk tag.** Plain text (no colour), formatted `±£X.XX` with
two special cases: returns `null` (hides the tag) if either
`predicted_locked_pnl_at_placement` or
`predicted_locked_stddev_at_placement` is missing, and
renders `±£<0.01` for a non-zero stddev that would round to
`£0.00`. Present-but-zero stddev renders faithfully as
`±£0.00` (that's the model saying "no uncertainty", not a
rounding artefact).

**Layout.**

Extended the shared `$bet-grid` from 11 to 14 columns in
[frontend/src/app/bet-explorer/bet-explorer.scss](../../frontend/src/app/bet-explorer/bet-explorer.scss):
`… 3.5rem 4.5rem 2.5rem 3.75rem 4.5rem 1.5rem` — slots added
for pair-class / chip / risk respectively. Rows without a chip
or risk render a zero-size placeholder span in each slot so
the P&L and replay columns stay in fixed positions regardless
of which bets carry predictions. Verified in browser with a
fabricated fixture exercising all three chip buckets plus the
hidden-row case.

**Theme parity.** The existing pair-class-badge hard-codes
colours (no CSS variables exist in the bet-explorer SCSS).
The new `.confidence-chip` and `.risk-tag` follow the same
pattern rather than introducing a theming scheme — scope-
conservative, and anything broader belongs in a dedicated
theming plan. The dark-only app renders both consistently.

**Tests.**

- Frontend: 10 new cases in
  [frontend/src/app/bet-explorer/bet-explorer.spec.ts](../../frontend/src/app/bet-explorer/bet-explorer.spec.ts)
  covering the `confidenceBucket` helper (null / near-default
  / high / med / low / threshold constants), `formatRiskTag`
  (missing / normal / near-zero / exactly-zero), and full
  DOM-level rendering assertions for all six observable
  states of the chip + risk tag. `ng test --watch=false` →
  491 passed, 24 skipped.
- API: 3 new contract tests on `TestBetExplorer`. Full pytest
  suite → 1971 passed, 7 skipped, 1 xfailed.

**Browser verification.**

Fabricated three bets on a throwaway `SESSION04DEMO` model:
- `fill_prob=0.85` → green "High" chip, tooltip
  "85.0 % predicted fill rate at placement", risk tag
  `±£1.25`, tooltip "Predicted locked P&L: £3.50 ± £1.25 …".
- `fill_prob=0.30` → red "Low" chip, risk tag `±£0.80`.
- `fill_prob=None` → no chip, no risk tag; row unchanged
  from pre-Session-04 appearance.

All three rendered correctly on the Bet Explorer at
`http://localhost:4202/bets`; the P&L and replay button
stayed aligned across all three row types thanks to the
placeholder spans. Fixture + synthetic parquet were cleaned
up after verification.

**No reward-scale change.** Pure UI session — env, trainer,
and aux heads all untouched.

---

## Session 03 — Risk / predicted-variance aux head (2026-04-16)

**Landed.**

- Every policy architecture grows a `risk_head = nn.Linear(backbone,
  max_runners * 2)` alongside the Session-02 `fill_prob_head`:
  `PPOLSTMPolicy` / `PPOTimeLSTMPolicy` take `lstm_last` as backbone,
  `PPOTransformerPolicy` takes the last-tick transformer output
  `out_last`. Forward reshapes the linear output to
  `(batch, max_runners, 2)`, splits into `mean` + `log_var`, and
  clamps `log_var` to `[RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX] = [-8, 4]`
  before writing to `PolicyOutput`. The clamp lives at the forward-pass
  boundary so downstream consumers (UI, parquet, NLL) never see an
  unsafe log-var — `exp(log_var)` stays finite inside the Gaussian NLL
  regardless of what the raw head emits. Head init: orthogonal gain
  `0.01` on the weight, zeros on the bias → at init the raw mean output
  ≈ 0 and raw log_var ≈ 0 (stddev ≈ £1 on £100 budget — a reasonable
  "unsure" prior). Shares the backbone, NOT the actor or fill-prob head
  (hard_constraints §8: state-conditioned, not head-conditioned).
- **`PolicyOutput` grows two fields**, both defaulting to
  zero-tensors shape `(1, 1)` so stub policies / legacy callers that
  construct the dataclass positionally keep working:
  `predicted_locked_pnl_per_runner` (the clamped-input mean channel)
  and `predicted_locked_log_var_per_runner` (the clamped log-var
  channel). Zero / zero → mean=0, log_var=0 → stddev=1, matching the
  "unsure" prior convention the Session-02 fill-prob default uses.
- **Capture → attach flow.** In `_collect_rollout` each tick snapshots
  both the per-runner mean and `exp(0.5 * log_var)` (computed once at
  capture so parquet consumers don't replicate the math) into numpy
  arrays alongside the existing fill-prob snapshot. The
  `aggressive_placed=True` walk over `info["action_debug"]` now stamps
  `predicted_locked_pnl_at_placement` and
  `predicted_locked_stddev_at_placement` on the same newest-matching
  `Bet` it already stamps `fill_prob_at_placement` on. The existing
  `pair_to_transition` map is reused — one dict, two labels per pair.
  At episode end the backfill loop computes realised `locked_pnl` for
  completed pairs by inlining the same commission-aware math
  `BetManager.get_paired_positions` uses (races share
  `env.all_settled_bets` but per-race BetManagers are gone by backfill
  time, so inlining avoids reconstructing them). Naked pairs
  contribute nothing — `risk_labels` stays NaN and the NLL mask
  rejects it.
- **`Transition.risk_labels: np.ndarray` shape `(max_runners,)`**,
  NaN default, parallel to `fill_prob_labels`. Same NaN-as-mask idiom;
  same 1-element fallback for directional runs keeps them cheap.
- **Gaussian NLL aux loss in `_ppo_update`.** A new module-level
  `_compute_risk_nll(means, log_vars, labels)` helper mirrors
  `_compute_fill_prob_bce`: masked NLL on the constant-free form
  `0.5 * (log_var + (target - mean)^2 / exp(log_var))`, defensive
  zero-return on all-NaN mask. Batch labels are built alongside the
  existing fill-prob labels at the top of `_ppo_update`, moved to
  device, sliced by `mb_idx`. A new aux term
  `self.risk_loss_weight * risk_loss` lands in the total loss, and
  `loss_info["risk_loss"]` (pre-weight mean per update) is emitted
  every update so operators can watch head behaviour even when the
  weight is 0.
- **Plumbing-off default — reward scale unchanged.**
  `risk_loss_weight` defaults to `0.0` (read from `hp` first, then
  `config["reward"]`). With both aux weights at `0.0` this session's
  total loss is numerically identical to Session 02. Verified by
  `test_risk_weight_zero_is_noop_on_total_loss`. The
  `raw + shaped ≈ total_reward` invariant still holds — verified by
  `test_raw_plus_shaped_invariant_still_holds_with_risk_loss`, which
  threads both `fill_prob_loss_weight=0.5` and `risk_loss_weight=0.5`
  through `reward_overrides` and checks the env accumulators are
  undisturbed.
- **Gene passthrough.** `risk_loss_weight` added to
  `agents/ppo_trainer.py::_REWARD_GENE_MAP` and to
  `env/betfair_env.py::_REWARD_OVERRIDE_KEYS`. Same rationale as the
  Session-02 fill-prob key: env never reads it, but whitelisting keeps
  the "unknown overrides" debug log quiet on gene passthrough.
- **Checkpoint back-compat: explicit migration helper.** New
  `migrate_risk_head(state_dict, fresh_policy)` in
  `agents/policy_network.py` mirrors `migrate_fill_prob_head` exactly
  but keyed on `"risk_head."` — injects fresh weights for any missing
  keys so `load_state_dict(..., strict=True)` succeeds. Chose a
  parallel helper over extending the Session-02 one so each session's
  migration story stays independently auditable. Test
  `test_legacy_checkpoint_loads_with_risk_head` confirms a
  pre-Session-03 dict (risk_head.* stripped) strict-loads after
  migration and the fresh init contract (non-zero weight, zero bias)
  holds.
- **`Bet` + `EvaluationBetRecord` + parquet.** `Bet` gets
  `predicted_locked_pnl_at_placement: float | None = None` and
  `predicted_locked_stddev_at_placement: float | None = None`.
  `PassiveOrderBook.on_tick` fill-path extends the existing
  Session-02 inheritance lookup — one `pair_id` scan now copies
  three fields (fill-prob + risk mean + risk stddev) from the
  aggressive partner onto the newly-created passive `Bet`. Captured,
  not recomputed (hard_constraints §10).
  `registry/model_store.py::EvaluationBetRecord` mirrors both fields;
  the parquet writer emits two nullable-float columns; the reader
  uses `has_predicted_locked_pnl` + `has_predicted_locked_stddev`
  guards in the same style as the existing `has_pair_id` /
  `has_fill_prob` checks. Back-compat: missing columns read as
  `None` (hard_constraints §11). `training/evaluator.py` forwards
  both fields into each `EvaluationBetRecord` construction.
- **Tests.** `TestRiskHead` class (13 tests) covering: shape +
  clamp-band on all 3 architectures; decision-time stamp onto a
  placed `Bet` (mean any float, stddev > 0); passive-leg inheritance
  of both risk fields; analytic-value NLL on perfect predictions with
  log_var at the clamp minimum; NLL gradient sign in both mean and
  log_var directions; `risk_weight=0` noop on total loss;
  `risk_weight=1` lifts the policy-parameter gradient norm relative
  to 0; parquet back-compat on both missing columns; parquet
  roundtrip preserving both values; legacy state-dict migration +
  strict load with fresh-init contract; raw+shaped invariant with
  both aux weights on; NLL exclusion of NaN-labelled samples; log-var
  clamp enforcement inside `forward` (overwritten head bias forced
  outside ±100, tensor reaching `PolicyOutput` still within clamp
  band). Full suite: `pytest tests/ -q` → **1968 passed, 7 skipped,
  1 xfailed**. Net +13 vs Session 02 (1955 → 1968) — matches exit-
  criteria target exactly.
- **Incidental fix.** `test_gradients_flow_through_actor` in
  `tests/test_policy_network.py` already skipped `fill_prob_head.*`;
  widened the filter to also skip `risk_head.*` for the same reason
  (hard_constraints §8 — sibling aux head, no gradient from an
  actor-mean-only loss). And updated
  `tests/test_model_store.py::test_parquet_schema_correct` to include
  the two new columns in the expected schema set so strict schema
  checks stay locked on the intended column list (rather than stale
  pre-Session-03 expectations).

**Back-compat summary.**

- Scalping OFF or scalping ON with both aux weights `0.0`: byte-
  identical total loss to Session 02. Reward invariant untouched.
- Pre-Session-03 checkpoints: strict-loadable after a single call to
  `migrate_risk_head(state_dict, fresh_policy)`. The `risk_head.*`
  keys are the only addition; all other parameter shapes are unchanged.
  Sessions 01 + 02 migrations are still needed for pre-Session-01 /
  pre-Session-02 dicts; stack the helpers in order.
- Pre-Session-03 parquet files: readable unchanged — the two new
  optional columns return `None` per hard_constraints §11.

**Activation.** Session 03 exits plumbing-off.
`plans/scalping-active-management/activation_playbook.md` is the
protocol for turning both aux weights up — run the zero-weight
baseline (Step A), sweep fill-prob (Step B), sweep risk (Step C),
verify joint (Step D), promote to `config/scalping_gen1.yaml`
(Step E). Do not promote a weight that fails the ± 20 % reward-delta
bar.

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
