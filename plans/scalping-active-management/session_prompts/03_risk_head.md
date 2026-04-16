# Scalping Active Management — Session 03 prompt

Work through session 03 fully (code + tests + progress.md
entry) before moving on. Commit after the session.

## Before you start — read these

- `plans/scalping-active-management/purpose.md` §"3. Risk /
  predicted-variance head" — why this head exists, the three
  gains (variance-aware sizing, risk badges in the UI, future
  CVaR hook), and what "risk predictions correlate with
  realised locked-P&L variance" means (Spearman ρ > 0.3
  within a run's bets).
- `plans/scalping-active-management/hard_constraints.md` —
  especially:
  - §1–3 — PPO reward invariant untouched. Risk loss is a
    SEPARATE loss term. `raw + shaped` accumulators unchanged.
  - §8 — aux head shares the backbone (LSTM / transformer
    output), NOT the policy head or the fill-prob head.
    Conditions on STATE, not on sampled action.
  - §9 — pre-Session-03 checkpoints must still load. New
    head initialises fresh.
  - §10 — per-bet predictions are captured at decision time,
    not recomputed later.
  - §11 — new parquet columns are optional; missing columns
    on old data read as `None`.
- `plans/scalping-active-management/progress.md` — Session 01
  and Session 02 entries. Session 02's capture→attach flow
  (per-tick prediction → stamped onto `Bet` → inherited by
  paired passive via `pair_id` lookup) is the template you're
  mirroring. Same backfill, same migration-helper pattern.
- `plans/scalping-active-management/session_prompts/02_fill_prob_head.md`
  — the just-landed fill-prob session. Session 03 is
  structurally near-identical; the deltas are: 2-output head
  instead of 1, Gaussian NLL instead of BCE, log-var clipping,
  and the label is realised `locked_pnl` (a float) instead of
  a binary `completed/naked` flag.
- `plans/scalping-active-management/lessons_learnt.md` — in
  particular the Session 02 findings on:
  - BCE-with-probabilities vs logits (risk head analogue:
    raw mean + raw log-var vs clamped log-var; you want the
    clamped variant for numerical sanity, see below).
  - NaN-as-mask on `Transition.fill_prob_labels` — same idiom
    applies to the new `risk_labels`.
  - `create_policy(name=..., ...)` — keyword is `name`.
- `CLAUDE.md` — "Reward function: raw vs shaped" (risk NLL
  also sits outside both accumulators) and "Bet accounting:
  matched orders, not netted positions" (one Bet per fill —
  same supervision-per-pair setup as Session 02).
- `plans/scalping-active-management/activation_playbook.md` —
  familiarise yourself. Session 03's exit state leaves
  `risk_loss_weight` at 0.0, and the activation playbook is
  where that weight (and Session 02's `fill_prob_loss_weight`)
  gets turned up. Nothing to do in this session on activation
  itself; just make sure your defaults line up with what the
  playbook expects.

## Before you touch anything — locate the code

```
grep -rn "fill_prob_head\|fill_prob_per_runner\|fill_prob_at_placement" agents/ env/ registry/ training/ tests/
grep -rn "migrate_fill_prob_head\|fill_prob_loss_weight" agents/ env/ registry/
grep -rn "locked_pnl\|get_paired_positions" env/ agents/ registry/
```

Identify:

1. The Session-02 additions in `agents/policy_network.py`:
   `fill_prob_head` on all three architectures, the
   sigmoid-applied output in `PolicyOutput`, the
   `migrate_fill_prob_head` helper. You're adding the exact
   sibling of each — `risk_head` (two outputs per runner),
   `predicted_locked_pnl` + `predicted_locked_stddev` on
   `PolicyOutput`, `migrate_risk_head` helper.
2. Session-02's capture→attach flow in
   `agents/ppo_trainer.py::_collect_rollout` — the per-tick
   `fp_per_runner_t` snapshot, the `action_debug` walk, the
   `pair_to_transition` dict, and the episode-end backfill.
   Session 03 reuses the same `pair_to_transition` map (one
   dict, two labels per pair): mean-target is the realised
   `locked_pnl` of the completed pair; naked pairs don't
   contribute to risk labels (realised_locked_pnl is
   undefined for them — mask with NaN).
3. The aux-loss block in `agents/ppo_trainer.py::_ppo_update`
   — the pattern is
   ``loss + self.fill_prob_loss_weight * fill_prob_loss``.
   You're adding a second term
   ``+ self.risk_loss_weight * risk_loss``.
4. `env/bet_manager.py::Bet` — already has
   `fill_prob_at_placement`. Add two more fields for risk.
5. `env/bet_manager.py::BetManager.get_paired_positions` —
   this is the source of `locked_pnl` per pair (see existing
   tests `TestPairedPositions` for shape). The episode-end
   backfill uses it to compute the realised target for each
   completed pair.
6. `registry/model_store.py::EvaluationBetRecord` — mirror
   the two new fields. Parquet back-compat pattern is the
   same (`has_predicted_locked_pnl` guard, mirroring
   `has_fill_prob`).

Write what you find into your scratchpad before editing.
In particular: does the NLL benefit from operating on
`log_var` directly (stable gradient, cheap) or on `var =
exp(log_var)` (more intuitive but can blow up)? Answer:
log-var, clamped to `[-8, 4]` per master_todo.md. Note that
the "stddev" exposed on `PolicyOutput` and `Bet` is the
`exp(0.5 * log_var)` value — what a human would read — but
the loss operates on log-var directly.

---

## Session 03 — Risk / predicted-variance head

### Context

Session 02 gave the policy direct supervision on "will this
pair complete?". Session 03 adds direct supervision on "what
is the P&L distribution of this pair given that it does
complete?". Together they let a later session build
variance-aware sizing (stake ∝ 1 / predicted_stddev) and a
UI risk badge ("+£5 ± £15" vs "+£1 ± £1.50").

The loss is Gaussian NLL on realised locked_pnl:

```
risk_loss = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
```

where `mean` and `log_var` are per-runner outputs of the new
head, and `target` is the realised locked_pnl of the pair
placed at that tick-slot. Pairs that went naked (no realised
locked_pnl to supervise against) are masked out of the NLL
the same way unresolved fill-prob samples were in Session 02.

Like Session 02, the head lands **plumbing-off**:
`risk_loss_weight` defaults to `0.0`, so the total loss is
numerically identical to Session 02 unless someone opts in.
The `activation_playbook.md` covers the turn-on procedure.

### What to do

1. **Extend every architecture with a risk head.**
   - In `agents/policy_network.py`, add a small linear head
     to `PPOLSTMPolicy`, `PPOTimeLSTMPolicy`, and
     `PPOTransformerPolicy`.
   - Input: same backbone tensor as the fill-prob head
     (`lstm_last` / `out_last`). NOT the fill-prob head's
     output (hard_constraints §8: state-conditioned, not
     head-conditioned).
   - Output: `(batch, max_runners, 2)` — first channel is
     `mean`, second channel is `log_var`. Apply log-var
     clamping inside `forward` so the tensor that reaches
     `PolicyOutput` is already in `[-8, 4]`; this keeps
     downstream consumers (UI, parquet) safe without them
     having to remember the clamp bounds.
   - Init: orthogonal gain=0.01 on the weight, zeros on the
     bias — matches the actor-head + fill-prob-head pattern.
     Bias of zero plus small orthogonal weight means the
     initial `mean` output ≈ 0 and initial `log_var` output
     ≈ 0 (i.e. stddev ≈ £1 on a £100 budget — a reasonable
     "unsure" prior).

2. **Extend `PolicyOutput`** with two new fields:
   - `predicted_locked_pnl_per_runner: torch.Tensor` shape
     `(batch, max_runners)` — the `mean` channel.
   - `predicted_locked_log_var_per_runner: torch.Tensor`
     shape `(batch, max_runners)` — the clamped log-var.
   Default values for stub policies / legacy callers: both
   zero-tensors shape `(1, 1)` (matches the "unsure" prior
   interpretation — mean=0, log_var=0 → stddev=1).

3. **Capture at placement time.** In the rollout loop, after
   stamping `fill_prob_at_placement` on the newest matching
   `Bet`, stamp the same `Bet` with:
   - `predicted_locked_pnl_at_placement: float | None` — the
     per-slot mean.
   - `predicted_locked_stddev_at_placement: float | None` —
     `exp(0.5 * log_var)`, computed once at capture time so
     parquet consumers don't need to replicate the math.
   Use the same `env.current_runner_to_slot()` resolution
   and the same "newest matching Bet" scan-backwards pattern
   Session 02 uses.

4. **Inherit on the paired passive leg.** In
   `PassiveOrderBook.on_tick` at fill-time, extend the
   existing lookup that copies `fill_prob_at_placement` from
   the aggressive partner: copy the two risk fields at the
   same time. Same `pair_id`-lookup, same one-for-loop
   structure.

5. **Label backfill.** In `_collect_rollout` at episode end,
   extend the existing `pair_to_transition` backfill loop:
   - Completed pair (≥2 bets with the pair_id): compute the
     realised `locked_pnl` from `env.all_settled_bets`
     grouped by pair_id. `BetManager.get_paired_positions`
     is the reference implementation; you can either call it
     (requires a `bm` per market) or inline the same math
     (back_pnl + lay_pnl for the pair's two bets — see that
     helper for commission handling).
   - Naked pair: target is undefined. Write NaN into the
     risk label cell; the NLL mask rejects it.
   - Add a new `Transition.risk_labels` field:
     `np.ndarray` shape `(max_runners,)`, NaN default,
     parallel to `fill_prob_labels`.

6. **NLL loss helper.** Add a module-level
   `_compute_risk_nll(means, log_vars, labels)` in
   `agents/ppo_trainer.py`, mirroring
   `_compute_fill_prob_bce`. Masked NLL:
   ```
   mask = ~torch.isnan(labels)
   safe = torch.where(mask, labels, zeros)
   per_elem = 0.5 * (log_vars + (safe - means) ** 2 / log_vars.exp())
   return (per_elem * mask).sum() / mask.sum()
   ```
   Defensive zero-return on all-NaN mask, same as the BCE
   helper.

7. **Aux loss in the PPO update.** In the update block add a
   fourth term:
   ```
   risk_loss = _compute_risk_nll(out.predicted_locked_pnl_per_runner,
                                 out.predicted_locked_log_var_per_runner,
                                 mb_risk_labels)
   loss = loss + self.risk_loss_weight * risk_loss
   ```
   Track in `loss_info["risk_loss"]` so operators can watch
   it even when the weight is 0.

8. **Config knob wiring.**
   - `agents/ppo_trainer.py`: `self.risk_loss_weight =
     float(hp.get("risk_loss_weight",
     config.get("reward", {}).get("risk_loss_weight", 0.0))
     or 0.0)`.
   - Add `risk_loss_weight` to `_REWARD_GENE_MAP`.
   - `env/betfair_env.py`: add `risk_loss_weight` to
     `_REWARD_OVERRIDE_KEYS` (same rationale as
     `fill_prob_loss_weight` — env doesn't read it, but
     whitelisting keeps the unknown-key debug log quiet).

9. **`Bet` + `EvaluationBetRecord` + parquet.**
   - `env/bet_manager.py::Bet` — two new fields,
     `predicted_locked_pnl_at_placement: float | None = None`
     and `predicted_locked_stddev_at_placement: float | None
     = None`.
   - `registry/model_store.py::EvaluationBetRecord` — mirror.
   - Parquet writer emits both columns; reader uses
     `has_predicted_locked_pnl` + `has_predicted_locked_stddev`
     guards (same pattern as the existing `has_pair_id` /
     `has_fill_prob` checks). Back-compat: missing columns
     read as `None`.
   - `training/evaluator.py` forwards both new fields into
     `EvaluationBetRecord` construction.

10. **Checkpoint back-compat.** Add a sibling helper
    `migrate_risk_head(state_dict, fresh_policy)` in
    `agents/policy_network.py`, mirroring
    `migrate_fill_prob_head` exactly but keyed on
    `"risk_head."`. Document why (auditable migration vs
    `strict=False`, same reasoning as Session 02).

### Tests (add to tests/test_forced_arbitrage.py)

Add a new `TestRiskHead` class. Suggested coverage —
structurally parallel to `TestFillProbHead`:

1. **`test_risk_output_shape_and_range`** — build each of
   the three architectures; assert
   `predicted_locked_pnl_per_runner.shape == (batch, max_runners)`
   and `predicted_locked_log_var_per_runner` is in the
   clamp band `[-8, 4]`.

2. **`test_risk_recorded_on_bet_at_placement`** — one step
   of the scalping env with an aggressive trigger; assert the
   placed aggressive `Bet` has non-`None`
   `predicted_locked_pnl_at_placement` (any float) and
   `predicted_locked_stddev_at_placement` > 0.

3. **`test_risk_inherited_by_paired_passive`** — force the
   paired passive to fill (same harness shortcut as
   `TestFillProbHead.test_fill_prob_inherited_by_paired_passive`);
   assert the passive `Bet`'s two risk fields equal the
   aggressive leg's. Captured, not recomputed.

4. **`test_risk_nll_zero_on_perfect_predictions`** —
   contrive a batch where `mean == target` exactly and
   `log_var` is the clamped minimum `-8.0`. Loss equals
   `0.5 * (-8.0 + 0)` summed and normalised — a known
   analytic value. Assert within 1e-5.

5. **`test_risk_nll_gradient_direction`** — single sample
   where `mean = 5.0`, `target = 0.0`, `log_var = 0.0`. NLL
   wrt `mean` should be positive (push mean down), wrt
   `log_var` should be negative (widen the variance to
   explain the residual). Assert both signs.

6. **`test_risk_weight_zero_is_noop_on_total_loss`** — mirror
   the Session-02 equivalent. Weight=0, any other hp, assert
   total loss equals the Session-02 sum (policy + value +
   entropy + fill_prob * 0 + risk * 0) within 1e-7.

7. **`test_risk_weight_positive_changes_gradient_norm`** —
   weight=1.0 produces a strictly larger total-param gradient
   norm than weight=0.0, with same seed and data.

8. **`test_risk_parquet_backcompat_missing_columns`** —
   write `EvaluationBetRecord` with both risk fields set,
   drop the two columns from the parquet file, read back,
   both fields read as `None`.

9. **`test_risk_parquet_roundtrip`** — write with
   `predicted_locked_pnl=1.23`,
   `predicted_locked_stddev=2.5`; read back; both values
   within float tolerance.

10. **`test_legacy_checkpoint_loads_with_risk_head`** —
    build a pre-Session-03 state dict (strip
    `risk_head.*` keys), migrate via `migrate_risk_head`,
    load strict=True, assert success + non-zero weight init
    + zero bias init.

11. **`test_raw_plus_shaped_invariant_still_holds_with_risk_loss`**
    — re-run the Session-02 invariant test with
    `risk_loss_weight=0.5` (and optionally
    `fill_prob_loss_weight=0.5`). `raw + shaped ≈
    total_reward` on the env side — aux losses don't modify
    the accumulators.

12. **`test_risk_excluded_from_loss_when_outcome_unresolved`**
    — minibatch with one NaN-labelled sample. Mutate that
    sample's mean prediction; NLL must be unchanged. Same
    invariant as fill-prob, different head.

13. **`test_log_var_clamped_in_forward`** — synthesise a
    forward pass where the raw head output (before clamp)
    would fall outside `[-8, 4]`. Assert
    `PolicyOutput.predicted_locked_log_var_per_runner` is
    strictly within the clamp band regardless. Without this,
    `exp(log_var)` can overflow during NLL and poison the
    optimiser.

### Exit criteria

- All new tests pass.
- `python -m pytest tests/test_forced_arbitrage.py
  tests/test_bet_manager.py tests/test_betfair_env.py
  tests/test_policy_network.py tests/test_ppo_trainer.py
  tests/test_model_store.py -q` — all green.
- `python -m pytest tests/ -q` — 1968+ passed
  (Session 02 baseline 1955 + 13 new risk tests), no new
  failures.
- `progress.md` updated with a Session 03 entry summarising
  what landed — the risk head's 2-output-per-runner shape,
  the log-var clamp rationale, the reuse of the
  Session-02 `pair_to_transition` backfill for the new
  `risk_labels` array, and the chosen migration strategy
  (parallel `migrate_risk_head` helper).
- `lessons_learnt.md` appended with anything surprising
  (Gaussian NLL gotchas, log-var clamp behaviour under the
  optimiser, whether the combined
  `fill_prob_loss_weight + risk_loss_weight` total loss has
  any minibatch-level interaction you didn't expect).
- `ai-betfair/incoming/` note: live inference needs to flow
  `predicted_locked_pnl_at_placement` and
  `predicted_locked_stddev_at_placement` through to its
  recommendations payload, same pattern as the
  `scalping-fill-prob-head.md` note dropped in Session 02.
- Commit referencing `plans/scalping-active-management/`
  and session 03. Message explicitly calls out:
  `risk_loss_weight` defaults to 0.0 → reward scale
  unchanged. Refer to `activation_playbook.md` for the
  turn-on procedure.

---

## Cross-session rules for the whole plan

- Run `pytest tests/ -q` after the session. All tests must
  pass.
- Do NOT touch `env/exchange_matcher.py`. See CLAUDE.md
  "Order matching: single-price, no walking" — three
  independent regressions last time this was relaxed.
- Do NOT touch the existing `arb_spread` action dim, the
  `arb_spread_scale` gene, the `requote_signal` dim, or the
  fill-prob head landed in Session 02. This plan adds new
  heads / dims; it does not re-tune existing ones.
- Do NOT "improve" unrelated code you happen to read. Scope
  is tight.
- Commit after the session. Call out any reward-scale
  changes in commit messages. (Session 03 should NOT change
  reward scale — the risk head is plumbing-off by default.)
- Knock-on work for `ai-betfair` — drop a note in
  `ai-betfair/incoming/` per the cross-repo postbox
  convention. Live inference is small-surface (doesn't
  train); the relevant new fields are the two risk
  predictions on the recommendations payload.
