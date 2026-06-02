# BC getting-it-right — hard constraints (locked decisions)

Settled. A fresh session inherits these; don't relitigate.

## §1 — THE METRIC (north star). Measure BC by held-out maturation AUC, not rollout mat%.
- **PRIMARY:** the policy's *held-out per-(tick, runner) maturation AUC*
  — how well `mature_prob_head`'s output ranks "this open will mature"
  on the 7 reserved holdout days. Reference ceiling = LightGBM **0.76**
  on the same task/features (`plans/imitation-first/_step1/
  predictability_results.json`). This isolates "did the policy learn the
  signal" from every rollout confound.
- **SECONDARY:** a precision/recall curve — sweep the open threshold,
  plot mat% (precision) vs opens-count, on holdout.
- **TERTIARY:** one fully-hedged rollout number (mat% / locked / day_pnl,
  fc=120 + close_walk=10) at the chosen threshold. Report it, but do NOT
  optimize on it alone — that was the imitation-first BC mistake (rollout
  mat% blends selection + timing + fill model + budget + the ctx=1/
  sequence recurrence mismatch).

## §2 — Data split (inherited from imitation-first; no leak).
- Train = 42 days Apr 6 → May 19; Holdout = the reserved 7 (May 20, 21,
  22, 25, 27, 28, 29). NEVER train/select/threshold-tune on the holdout.
- Hold ~2 train days out for BC early-stop (BC-val). Explicit day lists.
- Maturation-conditioned full-obs + predictor caches already scanned
  (`data/oracle_cache_v2/`).

## §3 — Hard negatives (the core fix). 
- The negative class for the MATURATION discrimination is the
  **force-closing spread-placeable opens** — NOT random non-opportunities.
  Use `scan_day(..., maturation_label_out=labels)` to label every
  spread-placeable candidate matured/not; positives = matured,
  hard-negatives = not-matured-but-placeable.
- Random non-opportunity `NegativeOracleSample`s MAY be retained as a
  SEPARATE, lightly-weighted NOOP-teaching signal for the actor, but the
  maturation head trains on the hard split. Keep the two roles distinct.

## §4 — Selectivity via mature_prob threshold, not the actor knob.
- Selection = open iff `mature_prob > threshold` (the existing
  `mature_prob_open_threshold` gate). Produces a clean precision/recall
  curve. Do NOT reintroduce the `neg_weight` knife-edge as the
  selectivity mechanism — §"how we got here" in purpose.md.

## §5 — Input normalization ON.
- Full obs (143-d/runner) needs the opt-in `input_norm=True` per-dim
  standardizer (landed + tested, imitation-first Step 1b /
  `feedback_full_obs_needs_input_norm`). Compute stats from train obs.

## §6 — Architecture.
- LSTM h256 default (campaign-consistent). The ctx=1-vs-sequence BC
  question is a LEVER to test (Step D), not a locked default. Transformer
  is out of scope here.

## §7 — Reuse, don't reinvent.
- `mature_prob_head` + `mature_prob_loss_weight` + `mature_prob_open_
  threshold` (CLAUDE.md "mature_prob_head feeds actor_head"),
  `fill_prob_head`, `DiscreteBCPretrainer`, `RolloutCollector` all exist.
  Prefer wiring these over new training code. The strict mature_prob
  label (force-closed → 0) already matches our hard-negative definition.
- Iterate STANDALONE (the `plans/imitation-first/_step1/` scripts give
  full control + fast iteration) rather than through the cohort runner
  (which still needs `input_norm` wired through `worker.py` — a separate,
  foot-gun-prone integration per `feedback_audit_launch_wiring`). Wire the
  cohort path only once the recipe is proven.

## §8 — Success bar (the gate to PPO).
BC "works" iff BOTH:
1. **Signal:** held-out maturation AUC ≥ ~0.70 (approaching LightGBM's
   0.76). If it plateaus well below (≈0.60), the pooled-LSTM actor
   bottleneck is losing signal → that's an architecture finding, not a
   tuning problem.
2. **Action:** at some threshold, fully-hedged holdout rollout shows mat%
   materially above the ~1% random baseline (target ≥ ~15–20%, toward the
   oracle's selectivity) with locked positive — a clearly better
   warm-start than the imitation-first BC (4% mat%, −£1513/7d).
Only then is the reward-aware BC→PPO step warranted.

## §9 — Honesty / discipline.
- Every result reported against the AUC north-star + the holdout split.
- Negative results are first-class: if the lead lever (Step B) doesn't
  reach the bar, that's a real finding (points at architecture or data),
  not a reason to keep tuning knobs.
- No PPO until §8 passes (operator directive 2026-05-31).
