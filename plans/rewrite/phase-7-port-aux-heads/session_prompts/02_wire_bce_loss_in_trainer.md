---
session: phase-7-port-aux-heads / S02
phase: rewrite/phase-7-port-aux-heads
parent_purpose: ../purpose.md
depends_on: S01
---

# S02 — wire BCE + Gaussian-NLL auxiliary losses into `DiscretePPOTrainer`

## Context

S01 added all three heads to `DiscreteLSTMPolicy`. With loss
weights at 0 they're benign (BCE heads' sigmoid output ~0.5,
near-constant column into actor; risk_head's NLL gradient zero
so backbone unchanged). This session adds three training signals
so the heads actually learn:

- BCE on `fill_prob_head` (label = pair has both legs matched).
- BCE on `mature_prob_head` (strict label — force-closes negative).
- Gaussian NLL on `risk_head` (locked-P&L regression on
  completed pairs; naked pairs NaN-masked out).

…and wires all three operator-facing loss-weight knobs end-to-end.

Read `plans/rewrite/phase-7-port-aux-heads/purpose.md` first.

## Pre-reqs

- S01 done and merged.
- Read [`agents/ppo_trainer.py:1080-1100`](../../../../agents/ppo_trainer.py)
  — v1 weight-precedence pattern. Read it for context, do NOT
  copy verbatim (see "STOP" section below).
- Read [`agents/ppo_trainer.py:2440-2460`](../../../../agents/ppo_trainer.py)
  — v1 surrogate-loss composition with BCE + NLL terms.
- Read [`agents/ppo_trainer.py:1696-1712`](../../../../agents/ppo_trainer.py)
  — v1 risk label computation (commission + win/lose pnl from
  BACK/LAY pairing on completed pairs). Port the arithmetic
  verbatim — it's the canonical source.
- Read [`training_v2/discrete_ppo/trainer.py`](../../../../training_v2/discrete_ppo/trainer.py)
  — current v2 `_ppo_update` to understand where to add the
  three loss terms.
- Read [`training_v2/cohort/worker.py:245-269`](../../../../training_v2/cohort/worker.py)
  — `_build_per_agent_reward_overrides` helper. The plumbing
  decision below interacts with this.
- Read [`env/bet_manager.py`](../../../../env/bet_manager.py) —
  `Bet.force_close` field, pair-id grouping, BACK/LAY side
  enum. The strict mature_prob label and the risk locked-P&L
  arithmetic both depend on this.

## ⚠️ STOP — read this before writing any code

**Do NOT copy v1's `hp.get(name, config_fallback)` pattern verbatim
into v2. It will silently re-ship the exact bug this whole plan
exists to fix.**

Why v1's pattern works in v1:
- v1's per-agent hp dict only contains keys for genes that are
  actually being varied per-agent. If the operator pins
  `mature_prob_loss_weight` cohort-wide via `--reward-overrides`,
  the key is ABSENT from hp, so `hp.get(name, fallback)` returns
  the fallback (the override value). Precedence works.

Why the same pattern fails in v2:
- v2's `CohortGenes` is a dataclass that always carries every
  gene field. When serialised to the hp dict via
  `genes.to_dict()`, every key is present with its default
  value (0.0 for both loss weights). `hp.get(name, fallback)`
  returns 0.0 — the fallback is never consulted. The override
  is silently swallowed.
- This is exactly what produced the byte-identical eval results
  on 2026-05-04 cohort
  `v2_phase5_oc1_mpw05_clean5day_1777849498` vs the prior
  cohort. Same seed, same effective trainer state because the
  override never reached a consumer.

This is load-bearing. The v1 head work was real and useful, and
copying its precedence pattern without understanding why it
worked in v1 is how that work got silently un-shipped in v2.
Don't repeat it.

## Plumbing decision (resolve early in the session)

Two paths to make `--reward-overrides mature_prob_loss_weight=0.5`
reach the trainer. Only Path A is acceptable — Path B is
documented for context, not as a real option.

**A) Worker pre-merges reward_overrides into the per-agent hp
dict before constructing the trainer.** The override value
overwrites the gene default in the hp dict at construction
time. Trainer's `hp[name]` (or `hp.get(name, 0.0)`) then
returns the override value because the key now carries it.
Single source of truth, no precedence ambiguity, no new
constructor surface. **This is what to ship.**

**B) Worker passes `reward_overrides` as a new kwarg to
`DiscretePPOTrainer`; trainer reads it via the v1 precedence
pattern (`hp.get(name, reward_overrides.get(name, 0.0))`).**
Mirrors v1's PPOTrainer signature. Rejected because: (i) it
adds a second source of truth (hp + reward_overrides both
inside the trainer), (ii) it preserves the v1 precedence
pattern that has the silent-swallow failure mode if anyone
ever populates `hp[name]` with the default, (iii) it
duplicates the worker's existing reward_overrides-merge
logic across the trainer boundary.

Document the choice and the v1-vs-v2 hp-dict difference in
`lessons_learnt.md` (create if absent). The lesson is the
load-bearing artefact for future maintainers: anyone refactoring
the trainer's hyperparameter reads needs to know v2's hp dict
is always-populated.

## Deliverables

1. `training_v2/discrete_ppo/trainer.py`:
   - `__init__` reads `self.fill_prob_loss_weight`,
     `self.mature_prob_loss_weight`, AND
     `self.risk_loss_weight` from the per-agent hp dict
     directly: `float(hp.get(name, 0.0) or 0.0)`. NO config
     fallback for any of them — the worker has already merged
     any reward_overrides into hp by this point (Path A). A
     nested fallback would re-introduce the v1 precedence trap.
   - `_ppo_update` (or wherever the surrogate loss is composed):
     - Compute `fill_prob_loss = bce(fill_logit, fill_label,
       mask=runner_mask)` per mini-batch.
     - Compute `mature_prob_loss = bce(mature_logit, mature_label,
       mask=runner_mask)` per mini-batch.
     - Compute `risk_loss = gaussian_nll(risk_mean, risk_log_var,
       risk_label, mask=risk_mask)` per mini-batch. The mask is
       distinct from `runner_mask` because naked pairs are
       runner-present but risk-NaN. Standard NLL form:
       `0.5 * ((label - mean)^2 / exp(log_var) + log_var)`,
       averaged over masked entries. Skip the term entirely if
       no runner in the mini-batch has a valid label (avoid
       NaN propagation from a zero-divisor mean).
     - Add to total loss:
       `total_loss += self.fill_prob_loss_weight * fill_prob_loss
                    + self.mature_prob_loss_weight *
                      mature_prob_loss
                    + self.risk_loss_weight * risk_loss`.
     - Per-update log line includes `fill_prob_bce_mean=...`,
       `mature_prob_bce_mean=...`, AND `risk_nll_mean=...`.
   - When all three weights are 0.0, total_loss is
     byte-identical to pre-plan. Verify with a regression test.

2. Label computation helper (new module,
   `training_v2/discrete_ppo/aux_labels.py`):
   - `compute_fill_label(bet_manager_snapshot) -> per-runner
     dict[runner_id, float]` — returns 1.0 for runners with any
     pair whose `matched_legs >= 2`, else 0.0.
   - `compute_mature_label(bet_manager_snapshot) -> per-runner
     dict[runner_id, float]` — returns 1.0 for runners with any
     pair where `matched_legs >= 2 AND no leg has
     force_close=True`, else 0.0.
   - `compute_risk_label(bet_manager_snapshot) -> per-runner
     dict[runner_id, float | None]` — returns the locked-P&L
     value for runners with a completed pair (per
     `agents/ppo_trainer.py:1696-1712`: highest-priced BACK +
     lowest-priced LAY, commission 0.05, `locked = max(0,
     min(win_pnl, lose_pnl))`). Returns `None` (or
     `float('nan')`, depending on tensor representation) for
     runners with only naked pairs — naked has no realised
     locked outcome to supervise against.
   - `runner_mask(bet_manager_snapshot) -> per-runner bool` —
     True for runners with at least one open or completed pair
     in the rollout window. Used by the BCE terms.
   - `risk_mask(bet_manager_snapshot) -> per-runner bool` —
     True ONLY for runners with a completed pair (subset of
     `runner_mask`). Used by the NLL term to skip naked-only
     runners.

3. Rollout buffer threading:
   - The existing rollout collection in `DiscretePPOTrainer`
     stores transitions; per-step BetManager state must be
     captured at rollout time so all three labels are available
     at update time. If the buffer doesn't already carry this,
     extend `Transition` with a small `aux_targets` field
     populated at each `step` call. Tolerate missing field on
     pre-plan transition objects (default-tolerance per Phase 3
     convention).
   - Risk labels can backfill: at rollout-time the pair may
     still be open (label is unknown). The label is only valid
     once the pair completes. v1's solution
     (`agents/ppo_trainer.py:1683+` "backfill" pass) writes
     risk_labels into earlier transitions when later transitions
     show the pair completed. Port this pattern — without it,
     all transitions before a pair's second leg fills would be
     NaN-masked and risk_head would never train. The backfill
     runs once at end-of-rollout, before the PPO update.

4. Worker plumbing (path A above):
   - `train_one_agent` (or wherever the trainer is constructed):
     pre-merge cohort-level reward_overrides into the per-agent
     hp dict for trainer-side keys
     (`fill_prob_loss_weight`, `mature_prob_loss_weight`,
     `risk_loss_weight`). Existing reward-overrides for env-side
     keys (open_cost etc.) are unchanged.
   - Symmetric to `_PHASE5_GENES_VIA_REWARD_OVERRIDES`, add
     `_PHASE5_GENES_VIA_TRAINER_KEYS = frozenset({
        "fill_prob_loss_weight", "mature_prob_loss_weight",
        "risk_loss_weight"})`.

5. Tests (extend `tests/test_v2_aux_heads.py`):
   - `test_all_aux_losses_zero_when_weights_zero` — total_loss
     byte-identical to a control run with no aux loss code
     path. All three weights at 0 → no contribution to total
     loss.
   - `test_bce_loss_nonzero_when_bce_weight_nonzero` — weight
     0.5 on either BCE head produces a measurable contribution
     to total_loss.
   - `test_risk_nll_nonzero_when_risk_weight_nonzero` — weight
     0.5 on risk produces a measurable NLL contribution to
     total_loss when at least one transition has a valid
     (non-NaN) risk label.
   - `test_risk_nll_zero_when_no_completed_pairs_in_minibatch` —
     all-naked rollout → risk_loss term is skipped (or
     contributes 0) without NaN propagation.
   - `test_risk_label_arithmetic_matches_v1` — fixture with one
     completed pair (BACK £10 @ 5.0, LAY £8 @ 4.5, commission
     0.05). Compute label by hand; assert
     `compute_risk_label(...)` returns the same value to fp32
     precision. Anchors the port to v1.
   - `test_risk_label_naked_pairs_are_nan_or_none` — fixture
     with one naked pair (only one leg matched) → label is
     None / NaN; risk_mask is False for that runner.
   - `test_strict_mature_label_excludes_force_closes` — fixture
     with three pairs (matured, agent-closed, force-closed) →
     labels [1.0, 1.0, 0.0]. Load-bearing semantic test.
   - `test_loss_weight_precedence_hp_value_wins` — hp dict
     value wins over default 0.0. Note that v2's hp dict is
     ALWAYS populated (from `CohortGenes.to_dict()`); a unit
     test that constructs an empty hp dict does NOT exercise
     the realistic failure mode.
   - `test_reward_overrides_reaches_trainer` — **load-bearing
     integration test, the regression guard for the bug this
     plan exists to fix.** Parametrised over all three weight
     keys (`fill_prob_loss_weight`, `mature_prob_loss_weight`,
     `risk_loss_weight`). Small cohort launch with
     `--reward-overrides <key>=0.5` produces a trainer with
     `self.<key> == 0.5` for every agent. Must use the real
     `CohortGenes` → `hp` flow (not a hand-constructed hp
     dict) so it catches the v2-specific always-populated-hp
     failure mode. Spy on the worker's hp-merge step or assert
     on the trainer's stored attribute after construction.
     Mirrors the lesson from
     `plans/naked-clip-and-stability/lessons_learnt.md` §
     "Session 03 reward centering: units mismatch bug" —
     caller-only integration tests catch what unit tests miss.
   - `test_naive_v1_precedence_pattern_is_NOT_used` — read
     the constructor source (or a fixture exercising it) and
     assert that the trainer does NOT consult a `config["reward"]`
     fallback when reading the loss weights. Forward-looking
     guard against a refactor that helpfully re-adds the
     "fallback to config" pattern and silently re-ships the
     bug.

## Out of scope

- Tuning what loss weight is best. S03 + follow-on probes.
- BC pretrain on oracle labels (v1 has this; v2 doesn't; defer
  to a separate plan).

## Stop conditions

- Stop and ask if the rollout buffer doesn't expose BetManager
  state per transition AND extending `Transition` with
  `aux_targets` requires a non-trivial refactor of the
  collector loop. Pull the refactor into its own session
  prompt.
- Stop if `test_all_aux_losses_zero_when_weights_zero` doesn't
  byte-identify with a control run — silent contributions are
  unacceptable.
- Stop if the risk-label backfill pass interacts badly with v2's
  rollout buffer (e.g. transitions are batched across episodes
  in a way that makes pair-id grouping non-trivial). The v1
  pattern assumes per-rollout pair-id stability; if v2 violates
  that, document the divergence and either patch the buffer to
  preserve the assumption or compute risk labels at
  episode-settle time in the env (more invasive).
- Stop if v1 precedence semantics differ subtly from what the
  v2 worker plumbs. Document the difference; pick the v2-only
  hp-dict-with-no-fallback semantics; lock in tests.

## Done when

- All tests in `tests/test_v2_aux_heads.py` pass (S01's plus
  S02's additions).
- Smoke cohort: `python -m training_v2.cohort.runner --n-agents
   2 --generations 1 --days 2 --device cuda --seed 42 --data-dir
   data/processed --reward-overrides mature_prob_loss_weight=0.5
   --reward-overrides risk_loss_weight=0.5
   --output-dir registry/_phase7_s02_smoke` completes;
  per-update log lines show `mature_prob_bce_mean > 0` AND
  `risk_nll_mean > 0`.
- Same smoke with both weights at 0.0 shows the corresponding
  log fields at 0 (or absent).
- Commit message: `feat(rewrite): phase-7 S02 - three aux loss
  terms (BCE x2 + Gaussian NLL) in DiscretePPOTrainer + cohort
  plumbing`.
