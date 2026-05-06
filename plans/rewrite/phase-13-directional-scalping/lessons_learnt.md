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

(Append on completion.)

## S04 — MTM-loss stop-loss

(Append on completion.)

## S05 — Direction-targeted BC pretrain

(Append on completion.)

## S06 — Validation cohort

(Append on completion.)
