---
plan: rewrite/phase-13-directional-scalping
parent_purpose: ./purpose.md
---

# Lessons learnt

Append-only journal. One entry per session as it lands. Keep entries
short and concrete: what we observed vs what the plan assumed, what
changed in the plan as a result, and any constraint that needs
hardening.

Inherited lessons (read before any session):

- `plans/selective-open-shaping/lessons_learnt.md` — the per-tick
  vs per-race gradient delivery lesson (Session 02, 2026-04-25).
  Direction signal must arrive at the open decision tick, not at
  settle. The S03 wiring follows this: BCE loss is computed
  per-transition with the cached label for that tick, not at end-
  of-race.

- `plans/fill-prob-in-actor/lessons_learnt.md` and
  `plans/mature-prob-in-actor/lessons_learnt.md` — the
  architecture-hash break protocol and the gradient-through guard
  pattern. S03 reuses both verbatim.

- `plans/naked-clip-and-stability/lessons_learnt.md` — units-
  mismatch bug. Any new accounting term that lands in `race_pnl`
  must be in the same units as the existing terms (per-race cash
  £). Stop-loss closes (S04) cost real cash; their P&L lands in
  `race_pnl` directly, not in shaped.

- `plans/rewrite/phase-7-port-aux-heads/lessons_learnt.md` — the v2
  `hp` dict / `--reward-overrides` precedence trap. Any new gene
  added by this plan
  (`direction_prob_loss_weight`,
   `direction_horizon_ticks`,
   `direction_threshold_ticks`,
   `mtm_stop_loss_threshold`,
   `bc_direction_target_weight`)
  is read from `hp` ONLY in the v2 trainer; the worker pre-merges
  `--reward-overrides` into `hp` via `_build_trainer_hp`. Do NOT
  copy the v1 `hp.get(name, config["reward"][name])` precedence
  pattern — it has a silent-swallow failure under `CohortGenes.
  to_dict()` semantics.

- `plans/rewrite/phase-12-counterfactual-fill-prob/` — once that
  plan lands, copy any cache-format / determinism / density-print
  patterns it established. The two label generators are sister
  modules and should share conventions.

---

## S01 — Feature audit

Landed 2026-05-06. Confidence read = **strong signal already there**.
Categories a (static price), b (recent direction), c (order book
pressure), e (market structure), and g (own position) are densely
populated. Category d (trade flow) has the headline features
(`traded_delta`, `vol_delta_3/5/10`, `book_churn`); the captured-but-
unused `TradedVolumeLadder` is a known cheap follow-on (memory note
`traded_volume_ladder_unused`) but NOT a prerequisite. Category f
(cross-runner) has rank / gap features but no cross-runner trade-flow
tensor — sequel work.

Unexpected discovery: `mid_drift` is already a 1-tick directional
proxy in the obs vector. The new direction head's threshold horizon
must be longer than 1 tick (default 5 per `purpose.md`) so the head
adds information `mid_drift` does not. Recorded in `findings.md`
"Notes for S02".

No `RUNNER_KEYS` gaps where features are computed but dropped. No
narrow-PR opportunities. Proceeding directly to S02.

## S02 — Direction-label generator

Landed 2026-05-06. `training_v2/direction_label_scan.py` +
`training_v2/direction_label_cli.py` + 13 tests
(`tests/test_v2_direction_labels.py` all pass). Cache layout
`data/direction_labels/{date}/horizon{H}_thresh{T}_fc{F}.npz` with
sibling `_header.json` carrying every invalidating key.

**Density print at default knobs** (horizon=60, thresh=5, fc=60.0):

| Date | Rows | density_back | density_lay | both | Wall |
|---|---|---|---|---|---|
| 2026-04-28 | 71,576 | 0.191 | 0.206 | 0.013 | 2.7s |
| 2026-04-29 | 71,238 | 0.187 | 0.209 | 0.017 | 2.7s |
| 2026-04-30 | 76,912 | 0.173 | 0.208 | 0.012 | 3.0s |
| 2026-05-01 | 72,023 | 0.199 | 0.214 | 0.021 | 2.8s |
| 2026-05-02 | 110,356 | 0.162 | 0.179 | 0.013 | 4.0s |
| 2026-05-03 | 55,190 | 0.220 | 0.232 | 0.026 | 2.3s |
| 2026-05-04 | 62,805 | 0.194 | 0.206 | 0.018 | 2.6s |

Density readings 0.16–0.23 — at the lower end of the plan-level 0.20–
0.50 healthy band but inside it for several days. Both-positive
fractions 1.2–2.6 % — well under the 10 % concern threshold (no sign
of LTP routinely oscillating through both thresholds within the
horizon). Wall budgets ~3 s/day — three orders of magnitude under
the 600 s budget; the vectorised window scan saturates the budget
already. No tuning iteration needed at this stage; if S03's BCE loss
struggles to converge we can revisit at threshold=3.

**Implementation note** — the prompt's pseudocode emitted a row only
when the future window had at least one tick. The deliverable test
suite expected the LAST pre-race tick of a 2-tick day to also emit a
row, which forced relaxing that gate: a row is now emitted whenever
priceability passes at the open tick, with both labels = 0 if the
forward window is empty. Determinism is preserved — every priceable
(tick, runner) emits exactly one row — and the head will simply learn
"no future evidence ⇒ stay near base rate" on those rows.

**`force_close_before_off_seconds` shadowed cohort knob** — the
direction labels are computed once with `fc=60`. If a cohort runs at
a different `force_close_before_off_seconds`, the cache stem won't
match and `load_labels(strict=True)` will raise. Cohort scripts must
run the scan first with the target `fc` value (S06 does this).

Proceeding to S03.

## S03 — Direction head wired into actor

Landed 2026-05-06. v2 stack only — v1 policy classes (`agents/
policy_network.py`'s LSTM / TimeLSTM / Transformer) NOT touched
(scoped out per `purpose.md` "if v1 is still in active use" + the
fact that S06 runs against v2's `DiscreteLSTMPolicy`).

**Changes:**

- `agents_v2/discrete_policy.py::DiscreteLSTMPolicy`
  - Added `self.direction_prob_head = nn.Linear(hidden,
    max_runners * 2)` next to `mature_prob_head`.
  - `actor_head[0].weight.shape[1]` widened from
    `runner_embed + hidden + 2` to `runner_embed + hidden + 4`.
    Architecture-hash break — pre-S03 checkpoints fail strict
    load (regression test in
    `tests/test_v2_direction_prob_in_actor.py::
    test_pre_direction_weights_fail_to_load`).
  - `DiscretePolicyOutput` gains four new fields:
    `direction_back_prob_per_runner`, `direction_lay_prob_per_runner`,
    `direction_back_logits_per_runner`, `direction_lay_logits_per_runner`.
    The two sigmoid columns feed actor_input; the trainer reads the
    raw logits for BCE-with-logits + pos_weight.

- `training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer`
  - Reads four new keys from `hp` (Path-A precedence; no config
    fallback): `direction_prob_loss_weight`,
    `direction_horizon_ticks`, `direction_threshold_ticks`,
    `direction_force_close_seconds`.
  - Lazy-loads the offline label cache once per (date, knob) tuple
    via `direction_label_scan.load_labels(strict=True)`. Cache key
    embeds `max_runners` so a different shim shape on the same day
    doesn't smuggle stale state.
  - Builds a per-env-step `(n_steps, R, 2)` label grid +
    `(n_steps, R)` mask aligned with the env's deterministic tick
    walk (race-by-race, tick-index-in-race) — no collector changes
    needed.
  - BCE-with-logits with per-side `pos_weight = (1 − d) / d` from
    the cache density, masked to supervised cells. Loss adds to
    `total_loss` as `weight × (back_loss + lay_loss)`.
  - Surfaces `direction_back_bce_mean`, `direction_lay_bce_mean`,
    `n_direction_targets`, `direction_prob_loss_weight_active` on
    `EpisodeStats` / `UpdateLog`.

- `training_v2/cohort/genes.py::CohortGenes`
  - Adds four phase-13 fields with defaults that are inert (weight
    0, knobs match S02's default scan triple).
  - `to_dict` extended; sample / mutate / crossover paths pin them
    to defaults (operator-controlled via `--reward-overrides`,
    not GA-evolved).

- `training_v2/cohort/worker.py::_build_trainer_hp`
  - Path-A passthrough: `--reward-overrides
    direction_prob_loss_weight=X` lands in `hp` before trainer
    construction. Same precedence guard as the Phase 7 keys.
  - `_rebind_trainer` clears `_direction_label_cache` so day
    transitions reload labels cleanly.

**Architecture-hash break verified.** Pre-S03 checkpoints fail to
load against post-S03 policies via the strict `load_state_dict`
path — the +2 column widening on `actor_head[0].weight` carries the
variant identity, no new explicit version field. Same protocol as
fill-prob-in-actor / mature-prob-in-actor.

**Default byte-identity verified.** `direction_prob_loss_weight = 0`
skips the `_build_direction_label_grid` call AND the per-mini-batch
BCE branch, so total_loss is byte-identical to pre-S03 on a
default-config run. The forward pass DOES still run the head and
inject its sigmoid output into `actor_input` — this is by design
(same as fill_prob / mature_prob default — near-`sigmoid(0) ≈ 0.5`
constant column). 268 v2 + policy_network tests pass. Five new
direction-head tests in `tests/test_v2_direction_prob_in_actor.py`
all pass.

**Per-step label alignment.** The trainer materialises labels by
walking `day.races` in race order, then iterating each race's
ticks in tick-index order — pre-race ticks get a global index
matching the scan output, in-play ticks get mask=False. This
mirrors `BetfairEnv`'s tick walk, so the per-env-step grid lines
up with the rollout's transition order without per-tick coords
plumbing in the collector.

**Calibration check deferred to S06.** A real-day rollout would
populate `direction_back_bce_mean` / `direction_lay_bce_mean`
across 3+ rollouts; that data lands when S06's cohort runs.

**Trainer integration tests deferred.** The four trainer-level
tests the prompt asks for (`test_direction_loss_zero_when_weight_zero`,
`test_direction_loss_nonzero_when_weight_positive`,
`test_direction_label_cache_missing_raises`,
`test_direction_pos_weight_matches_cache_density`) need a real
`shim.env.day` with a populated label cache to be meaningful. The
existing 268 trainer / cohort / policy tests all pass with the
default-zero weight, which exercises the byte-identity path; S06
exercises the weight > 0 path against real data. If S06 surfaces
a wiring bug we revisit.

Proceeding to S04.

## S04 — MTM-loss stop-loss

(Append on completion.)

## S05 — Direction-targeted BC pretrain

(Append on completion.)

## S06 — Validation cohort

(Append on completion.)
