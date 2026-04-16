# Scalping Active Management — Session 02 prompt

Work through session 02 fully (code + tests + progress.md
entry) before moving on. Commit after the session.

## Before you start — read these

- `plans/scalping-active-management/purpose.md` §"2. Fill-
  probability head" — why this head exists, the three gains
  (direct gradient to `arb_spread`, re-quote decision input, UI
  transparency), and what success looks like (±5 % calibration
  per bucket).
- `plans/scalping-active-management/hard_constraints.md` —
  especially:
  - §1–3 — the PPO reward invariant is untouched. The fill-
    probability loss is a SEPARATE loss term added to the
    training objective; it does NOT modify `raw` or `shaped`
    accumulators in `env/betfair_env.py`.
  - §8 — the aux head shares the backbone (LSTM / transformer
    output), NOT the policy head. It conditions on STATE, not on
    sampled action.
  - §9 — pre-Session-02 checkpoints must still load. New head
    initialises fresh.
  - §10 — the prediction captured on each `Bet` is the
    decision-time output, NOT recomputed later.
  - §11–12 — new parquet columns are optional; missing column on
    old data reads as `None`.
- `plans/scalping-active-management/progress.md` — Session 01
  notes. The 6th per-runner action dim (`requote_signal`) and
  the two new obs features are now live; they are inputs for
  Session 02's head, not work to redo.
- `plans/scalping-active-management/lessons_learnt.md` — in
  particular the "synthetic-market queue_ahead=0 auto-fill trap"
  lesson. Any new test that drives a pair through to settlement
  needs the same fence.
- `CLAUDE.md` — reinforce:
  - "Reward function: raw vs shaped" — aux loss stays outside
    both accumulators.
  - "Bet accounting: matched orders, not netted positions" —
    one `Bet` per fill. Every paired passive that fills is a
    positive supervised example; every paired passive that
    cancels at race-off is a negative one.

## Before you touch anything — locate the code

```
grep -rn "class PPOLSTMPolicy\|class PPOTimeLSTMPolicy\|class PPOTransformerPolicy\|PolicyOutput" agents/ env/
grep -rn "EvaluationBetRecord\|fill_prob\|aux_loss" agents/ registry/ api/
grep -rn "def update\|compute_loss\|loss.backward\|value_loss_coeff" agents/ppo_trainer.py
```

Identify:

1. The three policy architectures in `agents/policy_network.py`
   (`PPOLSTMPolicy`, `PPOTimeLSTMPolicy`, `PPOTransformerPolicy`)
   and the shared `PolicyOutput` dataclass at line ~215. Every
   architecture must grow a fill-probability head, and
   `PolicyOutput` must carry its per-runner output.
2. The PPO update loop in `agents/ppo_trainer.py` — look for the
   block starting `policy_loss = -torch.min(surr1, surr2).mean()`
   and `loss = policy_loss + self.value_loss_coeff * value_loss`.
   That's where the new aux-loss term lands.
3. The rollout collection path — where the trainer calls the
   policy network and stores actions / log-probs / values in a
   rollout buffer. You'll add a per-step-per-runner
   `fill_prob` capture here.
4. `env/bet_manager.py::Bet` — dataclass. New optional field
   `fill_prob_at_placement: float | None = None`. Must default
   to `None` so unrelated tests keep passing.
5. The bet-attachment path: when a paired passive fills, the
   resulting `Bet` object is created inside
   `PassiveOrderBook.on_tick` (and for the initial aggressive
   fill, via `BetManager.place_back` / `place_lay`). The
   prediction was computed by the policy network at the tick
   that PLACED the aggressive leg; you need to carry that number
   through to the eventual paired `Bet`s with the same
   `pair_id`. See the section below on "Capture → attach flow".
6. `registry/model_store.py::EvaluationBetRecord` + the parquet
   write/read paths. Mirror `fill_prob_at_placement` there; make
   reads tolerant of files saved before this column existed.

Write what you find into your scratchpad before editing. In
particular: does `PolicyOutput` get extended, or do you add a
new output field to `BasePolicy.forward`? Do the ppo_trainer
buffers already have a spot for per-runner auxiliary scalars, or
do you need to add one?

---

## Session 02 — Fill-probability auxiliary head

### Context

Session 01 gave the agent the mechanical ability to re-quote.
But the agent still picks `arb_spread` blindly — the only
signal flowing into that head is the long credit chain
`passive fills → locked_pnl → reward`. The fill-probability
head is the direct supervised signal: on every pair, we observe
whether the passive actually filled before race-off. That's a
binary outcome we can train a head on with BCE, and the
gradient flows back through the shared backbone to the
`arb_spread` head (and to `requote_signal`).

The head is **plumbed off** at the end of this session — the
config key `fill_prob_loss_weight` defaults to `0.0`, so the
total loss is numerically identical to Session 01 unless
someone opts in. A subsequent session turns the weight up and
verifies the head interferes gracefully.

### What to do

1. **Extend every architecture with a fill-probability head.**
   - In `agents/policy_network.py`, add a small linear head to
     `PPOLSTMPolicy`, `PPOTimeLSTMPolicy`, and
     `PPOTransformerPolicy`.
   - Input: whatever each architecture uses as its shared
     backbone tensor (the `lstm_last` / pooled transformer
     output). The head must share the backbone, NOT the actor
     head (hard_constraints §8).
   - Output: `(batch, max_runners)` logits. Apply `sigmoid`
     inside `PolicyOutput` (or expose logits; the trainer
     applies sigmoid + BCE-with-logits). Pick one and document
     why in the code comment.
   - Initialise with orthogonal gain=0.01 (matches the actor-
     head init — produces ≈0.5 output at init, i.e. "unsure").

2. **Extend `PolicyOutput`** with a new field
   `fill_prob_per_runner: torch.Tensor` shape
   `(batch, max_runners)`. Default value when a policy doesn't
   produce the head yet is a same-shape `0.5`-tensor, so any
   code that reads the field on a legacy path (e.g. tests that
   wrap `BasePolicy` directly) still works.

3. **Capture at placement time.** In `agents/ppo_trainer.py`'s
   rollout collection loop, when the policy is evaluated for a
   tick, store the per-runner `fill_prob_per_runner` values
   alongside the action / log-prob / value. Keyed by
   `(step_idx, slot_idx)`.

4. **Attach to paired bets.** When `BetfairEnv.step()` returns,
   inspect `info["action_debug"]` for `aggressive_placed=True`
   entries. For each such entry, the corresponding `Bet` is the
   newest entry in `env.bet_manager.bets` with that `sid` and
   `market_id`. Pull the decision-time `fill_prob` from the
   captured per-step store and write it to
   `bet.fill_prob_at_placement`. The paired passive `Bet`
   created later in `PassiveOrderBook.on_tick` inherits the
   same value by copying from the existing aggressive leg with
   the matching `pair_id` (look it up in
   `env.bet_manager.bets` by `pair_id` before creating the
   passive `Bet` — the aggressive always exists first).

   This keeps the hard_constraint §10 guarantee: the captured
   prediction is the one from the tick that placed the pair,
   not a recomputed post-hoc estimate.

5. **Aux loss in the PPO update.** In the update block that
   currently computes
   ```
   loss = policy_loss + self.value_loss_coeff * value_loss
   ```
   add a third term:
   ```
   fill_prob_loss = self._compute_fill_prob_loss(mb)   # BCE
   loss = loss + self.fill_prob_loss_weight * fill_prob_loss
   ```
   The minibatch-sized BCE: for every stored prediction that
   has a realised outcome (the pair either completed or went
   naked by race-off), compare the stored prediction to the
   binary label `1.0 if completed else 0.0`. Predictions
   without a resolved outcome (episode still running, or bet
   voided by missing result data) are EXCLUDED from the loss —
   do not insert a fake label.
   - Config knob: `config["reward"]["fill_prob_loss_weight"]`,
     default `0.0`. Wire through the same reward-override
     whitelist used for other reward genes.
   - Also log `fill_prob_loss` into the trainer's per-update
     stats (similar to `policy_loss` / `value_loss`) so
     operators can see how it's behaving.

6. **`Bet` field.** In `env/bet_manager.py::Bet`, add
   `fill_prob_at_placement: float | None = None`. Default
   `None` so existing tests that construct `Bet` directly with
   positional args keep working.

7. **Registry + parquet.**
   - `registry/model_store.py::EvaluationBetRecord` gets
     `fill_prob_at_placement: float | None = None`.
   - The parquet writer writes the column as a nullable float.
   - The parquet reader populates `None` for rows read from a
     file saved before this column existed (either via
     pyarrow's schema-evolution handling or an explicit
     `if col not in df: df[col] = None`).

8. **Checkpoint back-compat.** A pre-Session-02 state dict
   doesn't carry the new head's weights. Loading it with
   `strict=True` will fail. Two options:
   - Extend `migrate_scalping_action_head` (or write a sibling
     helper, `migrate_fill_prob_head`) that fresh-initialises
     the missing head on load.
   - Or loosen the loader to `strict=False` ONLY for the
     fill-probability head's parameter names.

   Pick ONE and document why. Either is fine; the constraint
   is that a Session 01 checkpoint must load and run without
   manual editing.

### Tests (add to tests/test_forced_arbitrage.py)

Add a new `TestFillProbHead` class. Suggested coverage — one
test per item; adapt wording but keep the invariants:

1. **`test_fill_prob_output_shape_and_range`** — build each of
   the three architectures and a dummy obs batch; call
   `forward`. Assert the new field is shape
   `(batch, max_runners)` and every element is in `[0, 1]`.

2. **`test_fill_prob_recorded_on_bet_at_placement`** — run one
   step of the scalping env with an aggressive back trigger,
   wrap the trainer's capture logic (or call a pure helper),
   and assert the placed aggressive `Bet` has a non-`None`
   `fill_prob_at_placement` in `[0, 1]`.

3. **`test_fill_prob_inherited_by_paired_passive`** — force
   the paired passive to fill (same harness shortcut used in
   the Session-2 reward tests). Assert the resulting passive
   `Bet` has the **same** `fill_prob_at_placement` as the
   aggressive leg — not a recomputed value.

4. **`test_fill_prob_bce_zero_on_perfect_predictions`** —
   contrive a minibatch where every prediction equals its
   outcome (1.0 or 0.0 exactly, with a tiny epsilon to avoid
   `log(0)`). Loss must be ≈ 0 (< 1e-5).

5. **`test_fill_prob_bce_gradient_direction`** — one sample:
   prediction=0.1, outcome=1.0. Backprop and assert the
   logit's gradient is negative (i.e. the optimiser would
   move the logit up). Symmetric case: prediction=0.9,
   outcome=0.0 → gradient positive.

6. **`test_fill_prob_weight_zero_is_noop_on_total_loss`** —
   one PPO update with `fill_prob_loss_weight = 0.0` and
   random predictions. Assert that the reported `loss` equals
   `policy_loss + value_loss_coeff * value_loss` within 1e-7.

7. **`test_fill_prob_weight_positive_changes_gradient_norm`**
   — same setup but weight=1.0; total-param gradient norm is
   strictly larger than the weight=0.0 case (proves the loss
   actually propagates).

8. **`test_parquet_backcompat_missing_column`** — write an
   `EvaluationBetRecord` with no `fill_prob_at_placement`,
   drop the column from the parquet file manually, then read
   it back. The resulting record must have
   `fill_prob_at_placement == None` and no crash.

9. **`test_parquet_roundtrip_with_fill_prob`** — write a
   record with `fill_prob_at_placement=0.73`, read it back,
   assert the value is preserved (within float tolerance).

10. **`test_legacy_checkpoint_loads_with_fill_prob_head`** —
    build a Session-01-shaped state dict (no fill-prob head
    weights), load it via the new loader path, assert
    `load_state_dict` succeeds and the head parameters exist
    with fresh init (non-zero for the linear weight, zero for
    bias).

11. **`test_raw_plus_shaped_invariant_still_holds_with_aux_loss`**
    — Session 01 had this; re-run with
    `fill_prob_loss_weight = 0.5`. Aux loss must NOT modify
    `info["raw_pnl_reward"]` or `info["shaped_bonus"]` —
    reward accumulators are unchanged.

12. **`test_fill_prob_excluded_from_loss_when_outcome_unresolved`**
    — include a prediction whose pair's `Bet` outcome is
    `UNSETTLED`. Compute the BCE loss; the unresolved sample
    must not contribute (change its prediction arbitrarily;
    loss should be identical).

### Exit criteria

- All new tests pass.
- `python -m pytest tests/test_forced_arbitrage.py tests/test_bet_manager.py tests/test_betfair_env.py tests/test_policy_network.py tests/test_ppo_trainer.py -q` — all green.
- `python -m pytest tests/ -q` — 1943+ passed, no new failures.
- `progress.md` updated with a Session 02 entry summarising
  what landed, noting the capture→attach flow (how the per-
  tick prediction ends up on the paired passive's `Bet`), and
  the checkpoint back-compat strategy chosen.
- `lessons_learnt.md` appended with anything surprising
  (BCE-with-logits vs sigmoid+BCE numerical stability, rollout
  buffer shape wrangling, parquet schema-evolution gotchas).
- Commit with a message referencing `plans/scalping-active-
  management/` and session 02. Call out explicitly that
  `fill_prob_loss_weight` defaults to 0.0 so reward scale is
  unchanged.

---

## Cross-session rules for the whole plan

- Run `pytest tests/ -x` after each session. All tests must
  pass.
- Do NOT touch `env/exchange_matcher.py`. See CLAUDE.md
  "Order matching: single-price, no walking" — three
  independent regressions last time this was relaxed.
- Do NOT touch the existing `arb_spread` action dim or the
  `arb_spread_scale` gene. This plan adds new dims / heads; it
  does not re-tune existing ones.
- Do NOT "improve" unrelated code you happen to read. Scope
  is tight.
- Commit after each session. Call out any reward-scale
  changes in commit messages. (Session 02 should NOT change
  reward scale — the aux head is plumbing-off by default.)
- Knock-on work for `ai-betfair` — drop a note in
  `ai-betfair/incoming/` per the cross-repo postbox
  convention. For Session 02 the live-inference surface is
  small (live inference doesn't train, it just reads the
  head's output); note what fields need to flow through to the
  recommendations payload.
