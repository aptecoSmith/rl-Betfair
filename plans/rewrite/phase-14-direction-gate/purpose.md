---
plan: rewrite/phase-14-direction-gate
status: DRAFT
opened: 2026-05-07
parent: plans/rewrite/phase-13-directional-scalping (NULL)
depends_on: phase-13 infrastructure (offline label generator,
            direction_prob_head, direction-label cache, BC layering)
---

# Phase 14 — Direction-gate selectivity

## Why this plan exists

Phase 13 added a direction signal to the policy and tested whether
the policy would learn to act on it. **Phase 13's S06 cohort
returned NULL: force-close rate stayed at 72-73% across all four
generations on both arms; direction BCE was flat at ~1.04, meaning
the head couldn't learn the label.** Detailed write-up in
`plans/rewrite/phase-13-directional-scalping/findings.md`.

After phase 13 closed we ran a series of supervised probes that
diagnosed the failure precisely. The findings — captured in
`lessons_learnt.md` of THIS plan — turn the NULL into an actionable
plan with quantified upside.

### What the probes established

1. **`tools/direction_head_supervised_probe.py`** — strip PPO, train
   `direction_prob_head` directly on cached labels. Showed the head
   gets only ~2× top-quintile lift even with 400 supervised steps.
   *Diagnosis: the head's architecture (single
   `Linear(hidden, max_runners*2)` off `lstm_last`) cannot extract
   per-runner alpha from the features.*

2. **`tools/direction_features_probe.py`** — replace the policy's
   head with a small per-runner MLP `Linear(in→64) → ReLU →
   Linear(64→2)`. Same data, same labels, same horizon. Result on
   2026-05-03: top-quintile lift jumps to 24-94×. Adding 8 augmented
   features (longer-window velocity, vol delta over 30/60 ticks,
   TradedVolumeLadder summaries) lifts that further to 38-94×.
   *Diagnosis: the features carry strong directional alpha; the
   bottleneck was the head architecture, not the data.*

3. **`tools/direction_threshold_sweep.py`** + `..._oos.py` +
   `..._xval.py` — sweep confidence thresholds, report mature rate
   and per-open P&L. In-sample numbers were partly overfit; honest
   out-of-sample on a held-out day showed only 3-7× lift. **Pooling
   4 training days lifted OOS calibration to 8-13× back-side, with
   monotonic 10-decile calibration tables.**

4. **`tools/cohort_per_pair_pnl_summary.py`** — extract real per-pair
   P&L from the phase-13 10h cohort scoreboards (n=92 eval rows):
   - Matured pair (completed + closed): **+£3.37** mean (σ=£0.38).
   - Force-closed pair: **−£1.80** mean (σ=£0.13).
   - Naked: **−£7.97** mean (rare, high-variance).
   - **Empirical break-even mature rate: 34.8%** — much lower than
     my eyeball £2.50/£3.00 ratio's 54.5% bar.

5. **Cross-validation sweep at empirical cost ratio.** Train MLP on
   4 days (2026-04-30 → 2026-05-03), eval on 3 held-out days
   (2026-04-28, -29, -05-04). With strict gate threshold (T ∈
   [0.90, 0.95]):

   | Eval day | Best T | Mature rate | £/open | Opens |
   |---|---|---|---|---|
   | 2026-04-28 | 0.95 | 63.1% | **+£1.46** | 325 |
   | 2026-04-29 | 0.95 | 53.2% | **+£0.95** | 233 |
   | 2026-05-04 | 0.90 | 45.1% | **+£0.53** | 1554 |

   **Three of three OOS days profitable.** Per-open P&L +£0.53 to
   +£1.46. At cohort scale (~410 opens/day) that's plausibly
   £200-£500/day per agent — versus current cohort ~−£300 to −£700.

   ⚠ **Caveat (per `sense_check.md`):** the probe counts every
   priceable row above the threshold as an "open". The cohort's
   actor opens at most ONCE per tick (categorical action sampled
   from masked logits). So the cohort's realised opens at T=0.95
   will be FEWER than the probe's row-count — perhaps 50-100
   per day instead of 233-1554. Translating: realistic per-agent
   per-day P&L is more like +£25-£200, not +£200-£500. The plan's
   success criterion is "positive day_pnl", not "+£X" — sign
   flip is the bar, magnitude is informative but not load-bearing.

### The strategic thesis

> The current cohort opens ~410 pairs/day; ~78% force-close at a
> loss. Use the direction predictor to **gate aggressively**: open
> only when `max(P_back, P_lay) ≥ T` for `T ∈ [0.85, 0.95]`. Volume
> drops 50-200×; mature rate climbs from 22% → 45-63%; per-pair
> economics flip positive.

Phase 14 implements the three compounding fixes the probes
identified as the path to that result:

(a) **Per-runner direction head.** Restructure
    `direction_prob_head` to mirror `actor_head`'s per-runner
    pattern. Without this the head cannot extract the calibration
    the probes demonstrated; phase 13's NULL would repeat.

(b) **Add 8 augmented features to `RUNNER_KEYS`.** Longer-window
    velocity (30 / 60 ticks) + per-price TradedVolumeLadder
    summaries. The probes show these add 50-70% incremental lift on
    top of the base features.

(c) **Hard-mask gate as a per-agent gene.** A
    `direction_gate_threshold` ∈ [0.5, 0.95] gene that the env
    consumes at action-mask-build time: `OPEN_BACK_i` and
    `OPEN_LAY_i` are masked out when
    `max(P_back_i, P_lay_i) < threshold`. The gate is the
    belt-and-braces — even if PPO doesn't learn to act on the
    head's output (the phase-13 failure mode), the env enforces
    selectivity mechanically.

## What this plan delivers

Four sessions, each landing as a separate commit. S01-S03
implement; S04 validates with a real cohort.

| Session | Deliverable | Depends on |
|---|---|---|
| S01 | Per-runner `direction_prob_head` MLP architecture | — |
| S02 | 8 augmented features in `RUNNER_KEYS`, OBS_SCHEMA_VERSION bump, cache regen | — |
| S03 | `direction_gate_threshold` per-agent gene + policy-side action mask | S01 (the gate reads the head's output) |
| S04 | Validation cohort + held-out-day re-eval | S01 + S02 + S03 |

S01 and S02 are independent; can run in either order. S03 reads the
head's output so depends on S01. S04 runs everything together.

## Hard constraints

See [hard_constraints.md](hard_constraints.md). Highlights:

- §1: All three fixes default to "off" — `direction_prob_loss_weight
  = 0.0` and `direction_gate_threshold = 0.5` (gate-disabled
  semantics: at 0.5 with positive-class density ~22%, the gate
  filters very few rows; effective behaviour is "no gate").
  Existing cohorts running without these levers stay byte-identical.

- §2: Architecture-hash break protocol same as phase 7 / 9 / 13 —
  pre-S01 checkpoints fail strict load by design.

- §3: OBS_SCHEMA_VERSION bump invalidates oracle and direction
  caches; cohort launch must re-scan before training.

- §4: `direction_gate_threshold` clamped at construction to
  [0.5, 0.95] — the upper cap prevents an agent from drawing 0.99+
  and never opening (which would starve PPO of signal).

- §5: Empirical cost-ratio assumption (£3.37 mat / £1.80 loss) is
  context for the gate threshold range, NOT a hard-coded assumption
  in the env. The env settles pairs at their actual realised P&L
  via the existing matcher; the probe-derived ratio is just what
  motivated the threshold range.

## Success bar

Plan-level gate is the **mature rate** climbing on the gate-on arm
of S04's validation cohort:

- **Primary gate (mature rate, gen 4):** mean across agents on the
  gate-on arm must reach **≥ 35%** (above the empirical break-even
  of 34.8%). Phase-13's baseline was 22.8%; we need a ≥ 12 pp lift.

- **Secondary gate (per-day P&L, eval days):** mean
  `eval_day_pnl` across held-out eval days on the gate-on arm must
  be **positive**. Phase-13 showed −£300 to −£700/day; we need to
  flip sign.

- **Non-degenerate:** `eval_pairs_opened` on the gate-on arm must
  stay above **50 per agent per day**. Lower means the agent
  starved itself with too-strict thresholds and the metric is
  dominated by per-pair noise rather than strategy quality.

- **OOS held-out re-eval:** for the top-3 surviving agents, re-run
  on 3 held-out days (not in training, not in eval). Mature rate
  should stay ≥ 35% on at least 2 of 3 held-out days for the
  intervention to count as robust (not an in-sample artefact).

## What this is NOT

- **Not a feature engineering plan from scratch.** The 8 augmented
  features are specifically chosen because the probes showed they
  add lift; we're not exploring a wider feature space here.
- **Not a label-spec change.** The direction labels stay at
  horizon=60, threshold=5 ticks (the probes confirmed shorter
  horizons learn LESS, not more).
- **Not a head-capacity sweep.** The probe established a small MLP
  head is sufficient. We're not running gene sweeps over hidden
  size, learning rate, etc — those keep the cohort defaults.
- **Not a re-run of phase-13's BC pretrain integration.** Phase 13
  S05 already wired direction-targeted BC. We're keeping it; not
  re-architecting it.
- **Not a stop-loss / force-close architecture change.** Those
  mechanisms already work (per phase-13 S04's discovery that the
  `stop_close` mechanism shipped earlier). The gate complements
  them, doesn't replace them.

## Open questions to resolve in S04

1. **What threshold does the GA converge on?** The probe says
   0.85-0.95 OOS. The cohort's data + cost ratio may push this
   slightly different. Read off the surviving population's gene
   distribution at gen 4.

2. **Does PPO compete with the hard mask?** The mask blocks
   `OPEN_*` actions but PPO still selects from the masked
   distribution. If the policy keeps picking NOOP because OPEN
   is blocked, that's fine. If the policy somehow degrades on
   OTHER actions (CLOSE, requote) because of the mask, we have
   a problem. Track action histograms.

3. **Does the augmented feature set (S02) help by itself?** Probe
   says +50-70% lift. S04's two-arm design (gate-off vs gate-on)
   doesn't directly test this — both arms have S02. A future
   ablation cohort could split S02 on/off.

4. **What's the realistic per-day P&L spread?** The probe used a
   single MLP; the cohort uses 12 different gene draws. Variance
   across agents at the same threshold tells us how much of the
   probe's signal generalises to varied PPO trajectories.

## Lessons inherited from phase 13

Read [lessons_learnt.md](lessons_learnt.md) before starting any
session. Highlights:

- **Diagnostic plumbing is load-bearing.** Phase 13 S06 NULL was
  qualified for two days because `direction_back_bce_mean` wasn't
  on the scoreboard. We didn't know if the head was training. Plan
  14 must surface every aux-head diagnostic by default —
  `train_mean_direction_back_bce` is in the scoreboard now (per
  commit 7fc3b73); this plan inherits that and extends it.

- **Always GPU.** Phase 13's 10h cohort ran on CPU because the
  launch shape inherited from prior smoke runs. Phase 14 sessions
  must default to `--device cuda`. Memory note
  `feedback_always_gpu.md` saved.

- **Probe before cohort.** Phase 13 ran a 10h cohort on a hypothesis
  that 2 minutes of supervised probing later disproved. Phase 14's
  S04 cohort is launched ONLY after S01-S03 have shown the head
  learns at the cohort scale (per the smoke probe in S04's
  prerequisites).

- **Empirical cost ratio matters more than feels.** Phase 13's
  break-even was eyeballed at 54%; real cohort data showed 34.8%.
  Future probes that compute "is this profitable?" must use real
  per-pair P&L from cohort scoreboards.

- **Multi-day training pooling raises true OOS lift.** Single-day
  OOS gave 3-7× lift; 4-day pooled gave 8-13×. Phase 14's S04
  cohort trains on 6 days as the norm; the BC pretrain caches all
  6 days' direction labels.

- **Day-by-day variance is large.** Per-day P&L varies ±£600 across
  agents at the same gene draw. The validation cohort's read-out
  must aggregate across MULTIPLE eval days, not a single one
  (phase-13 S06 hard_constraints §17–§18 inherited).

## What's NOT in scope (sequel work)

- Full feature engineering rebuild (deeper microstructure, external
  data sources). Phase 14 limits itself to the 8 augmented features
  the probes already validated.
- Magnitude-target direction labels (phase-13 hypothesis #3). The
  binary threshold-crossing label works at the empirical cost
  ratio; magnitude is a separate plan if Phase 14 succeeds.
- Cross-runner direction signal (predict the field's money flow).
  Sequel.
- New stop-loss / force-close architecture. Existing levers stay.
- A new policy class (e.g. transformer). Phase 14 only changes
  `DiscreteLSTMPolicy.direction_prob_head` shape.
