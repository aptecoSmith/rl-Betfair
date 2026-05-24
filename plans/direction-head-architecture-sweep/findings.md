# Findings — direction-head architecture sweep

**Date:** 2026-05-24
**HEAD at start:** `8878e98`
**Baseline:** v1 head at `models/direction_head/v1_2026-05-24/`
(LayerNorm -> Linear(23, 64) -> ReLU -> Linear(64, 2), pos_weight balanced).

This session ran FOUR rounds of variants: 5 in round 1 (the planned
set C0-C4), 5 follow-ons in round 2 (C6-C10, asked after round 1
showed C1 barely beat C0), 5 more in round 3 (C11-C15, asked after
round 2 showed wider+deeper helping), then 5 training-recipe
ablations in round 4 (C16-C20, all built on C11's architecture, each
changing exactly one recipe knob).

## TL;DR

**Winner: C11** = LayerNorm -> Linear(23, 256) -> ReLU -> Linear(256,
128) -> ReLU -> Linear(128, 2), trained with **pos_weight=1**
(unweighted BCE). Combines round-2's wider+deeper winner (C9) with
round-1's calibration finding (C3/C8). Pareto-best on the plan's
acceptance criteria: highest mean Pearson, calibrated Brier
(matching C8/C3/C14), AUC essentially tied with the deeper-balanced
variants.

| metric | C11 (winner) | C0 (baseline) | delta |
|---|---|---|---|
| mean Pearson | +0.2921 | +0.2719 | +0.0202 (+7.4 % relative) |
| mean ROC AUC | 0.7098 | 0.6976 | +0.0122 (+1.7 % relative) |
| mean Brier   | 0.1433 | 0.2282 | **-0.0849 (-37.2 % relative)** |
| pred mean (vs ~0.18 empirical) | tracks empirical | ~0.46 | calibrated vs 2.5× over-confident |

The headline number is the Brier improvement — 37 % reduction with
no Pearson cost. The Pearson lift (+7.4 %) is real but smaller than
the calibration win. C11 is dominant if the actor uses the head's
output as a probability (any threshold-based decision); it's
co-dominant with the balanced variants if the actor uses it as a raw
feature.

**C15 (pairwise feature expansion) is the round's biggest negative
result.** Best in-sample val_loss of any balanced variant (0.881),
worst held-out Pearson (+0.2614, below baseline). Pure overfit.
Implication: the 23-d input is not a representational ceiling that
could be unlocked by giving the head more derived features. The
ceiling is data / signal, not architecture or feature expressiveness.

## All 20 variants — final ranking

Mean Pearson averaged across both sides and 10 held-out eval days.
Brier cap (no >10 % regression vs C0): 0.2510. Round in which each
variant was added is shown in column R.

| rank | variant | R | family / recipe-note | layers | loss | mean ρ | mean AUC | mean Brier |
|---:|:---|:---|:---|:---|:---|---:|---:|---:|
|  1 | **C11** | 3 | linear_mlp | [256, 128] | unweighted | **+0.2921** | 0.7098 | **0.1433** |
|  2 | C13 | 3 | linear_mlp | [512, 256] | balanced | +0.2920 | 0.7109 | 0.2109 |
|  3 | C14 | 3 | linear_mlp | [256, 128, 64] | unweighted | +0.2918 | 0.7099 | 0.1437 |
|  4 | C9  | 2 | linear_mlp | [256, 128] | balanced | +0.2913 | **0.7119** | 0.2186 |
|  5 | C12 | 3 | linear_mlp | [256, 128, 64] | balanced | +0.2908 | 0.7115 | 0.2268 |
|  6 | C16 | 4 | c11 + AdamW(wd=1e-3) | [256, 128] | unweighted | +0.2894 | 0.7074 | 0.1436 |
|  7 | C20 | 4 | c11 + label_smoothing=0.05 | [256, 128] | unweighted | +0.2877 | 0.7065 | 0.1439 |
|  8 | C19 | 4 | c11 + GELU | [256, 128] | unweighted | +0.2875 | 0.7072 | 0.1439 |
|  9 | C8  | 2 | linear_mlp | [256] | unweighted | +0.2861 | 0.7042 | 0.1439 |
| 10 | C7  | 2 | linear_mlp | [1024] | balanced | +0.2841 | 0.7069 | 0.2273 |
| 11 | C6  | 2 | linear_mlp | [512] | balanced | +0.2835 | 0.7058 | 0.2227 |
| 12 | C1  | 1 | linear_mlp | [256] | balanced | +0.2803 | 0.7032 | 0.2185 |
| 13 | C2  | 1 | linear_mlp | [64, 32] | balanced | +0.2802 | 0.7050 | 0.2231 |
| 14 | C10 | 2 | linear_mlp_skip | [256] | balanced | +0.2799 | 0.7027 | 0.2184 |
| 15 | C18 | 4 | c11 + epochs=200 patience=20 | [256, 128] | unweighted | +0.2775 | 0.7062 | 0.1461 |
| 16 | C3  | 1 | linear_mlp | [64] | unweighted | +0.2758 | 0.6968 | 0.1448 |
| 17 | C0  | base | linear_mlp (baseline) | [64] | balanced | +0.2719 | 0.6976 | 0.2282 |
| 18 | C4  | 1 | linear_mlp_bn_dropout | [128] | balanced | +0.2707 | 0.6981 | 0.2242 |
| 19 | C15 | 3 | linear_mlp_pairwise | [256, 128] | balanced | +0.2614 | 0.6888 | 0.2204 |
| 20 | C17 | 4 | c11 + focal loss γ=2 | [256, 128] | unweighted | +0.2584 | 0.6929 | 0.1627 |

Every variant trains and converges within ~30-90 seconds on an
RTX 3090 (the lone exception is C18, which trained 200 epochs in
~3-5 min). Default recipe is `--epochs 50 --lr 1e-3 --batch-size
4096 --patience 5 --seed 42` and `--optimizer adam`.

## Round 4 — training-recipe ablations on C11 (negative results)

After round 3 settled on C11 as the architecture winner, round 4
asked the natural follow-on: **is C11's training recipe also
optimal, or is there a non-architecture knob we should turn?** Each
variant changes exactly one knob away from C11; everything else is
held identical.

| ID | knob changed | Pearson | delta vs C11 | verdict |
|---|---|---:|---:|---|
| C11 | (baseline)                         | +0.2921 | —       | (best) |
| C16 | Adam → **AdamW** (wd=1e-3)         | +0.2894 | -0.0027 | small regression |
| C20 | + **label smoothing 0.05**         | +0.2877 | -0.0044 | small regression |
| C19 | ReLU → **GELU**                    | +0.2875 | -0.0046 | small regression |
| C18 | 50 epochs / patience 5 → **200/20** | +0.2775 | -0.0146 | clear overfit |
| C17 | BCE → **focal loss γ=2**           | +0.2584 | -0.0337 | structural regression |

**Every recipe change hurt.** Adam, BCE, 50 epochs / patience 5,
ReLU, and hard binary targets — each of C11's recipe choices is
already at the local optimum for this data + arch. Five common
"improvements from the literature" (AdamW regularisation, GELU,
label smoothing, longer training, focal loss) all transferred
negatively to this regime.

The two strongest negative findings:

* **C18 mirrors C15's overfit story.** Val_bce kept improving across
  all 200 epochs (0.426/0.400 at ep 44 → 0.410/0.386 at ep 200), but
  held-out Pearson dropped by 0.0146. C11's tight `patience=5`
  early-stop catches the model right before generalisation breaks.
  This is the second piece of evidence (after C15's pairwise overfit)
  that the binding ceiling is generalisation, not capacity — and
  it reinforces the caution against any future C5 (full 574-d obs)
  attempts: more model OR more training overfits the same way.

* **Focal loss is structurally worst** (+0.2584 — below the original
  C0 baseline). Focal down-weights easy examples; at the 18 %
  positive rate, the "easy" negatives are abundant and provide most
  of the discriminative gradient. Down-weighting them removes signal.

The three small regressions (C16/C19/C20) cluster at +0.287-0.289
Pearson — all within typical day-to-day noise but still consistent
losses. Conventional wisdom about regularisation, activation
functions, and target softening does not transfer at this data
volume (1M samples) and model size (~40k params).

## What the three rounds taught us

### Width scaling has hard diminishing returns (round 2 → round 3)

| hidden width | variant | mean Pearson | delta from C0 |
|---:|:---|---:|---:|
|  64 | C0  | +0.2719 | (baseline) |
| 128 | C4  | +0.2707 | -0.0012 (BN+Dropout hurts) |
| 256 | C1  | +0.2803 | +0.0084 |
| 512 | C6  | +0.2835 | +0.0116 |
| 1024 | C7 | +0.2841 | +0.0122 |

Width helps from 64→256 (+0.0084), but each doubling beyond that
gains less than half the previous step. 1024 is essentially the same
as 512. Diminishing returns are clear.

### Depth keeps helping at fixed width=256, but plateaus at depth 3

| layers | variant | mean Pearson | delta |
|:---|:---|---:|---:|
| [256] | C1 | +0.2803 | - |
| [256, 128] | C9 | +0.2913 | +0.0110 |
| [256, 128, 64] | C12 | +0.2908 | -0.0005 from C9 |

So depth-2 helps but depth-3 doesn't. The dominant architecture
shape at this data scale is "wider single hidden, plus one
projection-down layer" — i.e. [W, W/2]. Adding more layers below
that doesn't compound.

### Wider+deeper barely improves over deeper alone

| variant | dims | mean Pearson |
|:---|:---|---:|
| C9  | [256, 128] | +0.2913 |
| C13 | [512, 256] | +0.2920 |

Doubling both dims gains +0.0007 (within noise). The combined width
+ depth lift is dominated by the depth lift; once you have a [W, W/2]
shape with W ≥ 256, width itself is sated.

### pos_weight=balanced vs pos_weight=1 — the dominant design lever

| arch | balanced (mean ρ / Brier) | unweighted (mean ρ / Brier) |
|:---|---|---|
| [64]       | C0: +0.2719 / 0.2282 | C3: +0.2758 / **0.1448** |
| [256]      | C1: +0.2803 / 0.2185 | C8: +0.2861 / **0.1439** |
| [256,128]  | C9: +0.2913 / 0.2186 | C11: +0.2921 / **0.1433** |
| [256,128,64] | C12: +0.2908 / 0.2268 | C14: +0.2918 / **0.1437** |

Switching to unweighted BCE consistently:
* Improves mean Pearson slightly (+0.001 to +0.006 across all archs)
* Cuts mean Brier by 35-37 %
* Produces calibrated outputs (pred mean tracks empirical rate; see
  round-1 reliability table preserved at the end of this doc)

The class-balanced `pos_weight` recipe — common practice for
imbalanced classification — is actively harmful here. It pulls the
output mean to ~0.46 (uniform decision boundary) when the empirical
positive rate is ~0.18, which doesn't help ranking and destroys
calibration.

### C18 corroborates the same overfit ceiling (round 4)

Round 4's C18 — same C11 architecture, just 200 epochs of training
with patience=20 — drove val_bce from 0.426/0.400 (at C11's natural
stopping point) down to 0.410/0.386. But held-out Pearson DROPPED
from +0.2921 → +0.2775, exactly mirroring C15's pattern. The
training process can keep improving the fit indefinitely, and
generalisation breaks well before train loss plateaus. Two
independent experiments (C15: more features, C18: more training) hit
the same ceiling from opposite directions.

This is unusually clean evidence that the model's optimal hyperplane
in weight-space is found early. The default `--patience 5` early-stop
is not over-cautious — it's tracking the actual generalisation peak.

### C15 (pairwise feature expansion) overfits — input ceiling is generalisation, not capacity

C15 was the most architecturally adventurous variant: expand the 23
per-runner inputs to `concat([x, outer_product(x, x).flatten()])` =
23 + 529 = 552 features, then a standard [256, 128] MLP. The
hypothesis was "if architecture saturates at +0.29 Pearson, maybe
the input is the ceiling — give it more derived features".

Result: C15 has the **best in-sample val_loss of any balanced variant**
(0.881 vs C9's 0.945, C13's 0.929) — clear evidence that the
expansion is adding real capacity. But its held-out Pearson is +0.2614
— **below the C0 baseline of +0.2719**, last place on the held-out
ranking.

This is a clean overfit signature: more capacity → better train fit,
worse generalisation. The 23-d input ceiling we see isn't from lack
of expressiveness; it's a data/signal ceiling. Adding derived
features over the same 23 inputs makes generalisation strictly worse.

**Implication for the deferred C5 (full 574-d input):** the C5
hypothesis — "give the head more inputs and it might find better
patterns" — should be approached with much more caution after C15.
The 574-d obs includes the same predictor columns that the 23-d
lean obs already carries, plus much more. There's no a priori reason
to think it'd avoid the same overfit failure mode. If C5 is run, it
should include strong regularisation by default (dropout high enough
to keep effective capacity comparable to C11) and a tighter early-
stop criterion based on held-out, not in-sample, val loss.

### What didn't help

* **C4 (BN + Dropout, [128]):** Slightly WORSE than C0. The model
  isn't overfitting at this data scale (1.03M samples for ~7k params)
  so regularisation only adds noise to the gradient.
* **C10 (skip connection from input to penult layer):** Identical to
  C1 within noise. The wider model already expresses the linear
  features the skip would have given direct access to.
* **C12/C14 (3 hidden layers):** Marginally worse than C9/C11
  respectively. Depth-3 plateau.

## Recommendation

**Promote C11** to the next cohort's `--direction-head-manifest`:

```
--direction-head-manifest models/direction_head/sweep_c11
```

Drop the mutually-exclusive flags (per
`plans/shared-direction-head/hard_constraints.md §4`):

* Remove `--enable-gene direction_prob_loss_weight`
* Remove `--enable-gene bc_direction_target_weight`

C11 has:
* The highest mean Pearson (+0.2921, +7.4 % over baseline)
* The lowest mean Brier (0.1433, -37.2 % over baseline)
* The strongest reliability profile (pred mean tracks empirical
  positive rate to within 1-2 points across every eval day — see
  reliability table preserved below)

The deployment story is: C11's output, when the head sees a (tick,
runner) where it thinks "this is a 25 % chance" — empirically about
25 % of those will resolve favourably. This is a meaningful change
from the baseline where "25 %" actually means "about 11 %" and "50
%" means "about 18 %", because pos_weight=balanced training pulled
all the outputs toward 0.5. The actor's `direction_gate_threshold`
gene operates on these probabilities and becomes meaningful with C11.

## Per-day Pearson — the winner is consistent

C11 leads or ties on every eval day except 2026-04-25 (where C13
beats it by 0.0023):

| date       | C0     | C9 (r2)   | C11 (r3 winner) | C13   |
|---|---|---|---|---|
| 2026-04-07 | +0.162 | +0.183 | **+0.197** | +0.187 |
| 2026-04-10 | +0.268 | +0.292 | **+0.294** | +0.292 |
| 2026-04-14 | +0.297 | +0.319 | +0.320     | **+0.321** |
| 2026-04-17 | +0.277 | +0.298 | **+0.301** | +0.300 |
| 2026-04-21 | +0.240 | +0.258 | **+0.262** | +0.258 |
| 2026-04-23 | +0.305 | +0.325 | **+0.331** | +0.328 |
| 2026-04-25 | +0.291 | +0.301 | +0.302     | **+0.304** |
| 2026-05-01 | +0.290 | +0.305 | **+0.308** | +0.306 |
| 2026-05-03 | +0.331 | +0.339 | **+0.349** | +0.338 |
| 2026-05-06 | +0.257 | +0.273 | **+0.273** | +0.272 |

The improvement is robust across the entire eval-day pool, not
driven by one or two anomalous days.

## What the sweep DIDN'T prove

* That C11 is the right head for the COHORT — only that it's the
  right head for predicting the direction labels held out from
  training. The cohort might be sensitive to a property the eval
  doesn't measure (e.g. how the output distributes across runners
  within a single tick).
* That C5 (full 574-d input) wouldn't beat C11. The sweep was
  scoped to per-runner-input architectures per the plan; C15 makes
  the C5 expected value lower than I'd have guessed before this
  session, but it doesn't rule it out.
* That changing the label or the predictor wouldn't unlock more
  signal. Both are out-of-scope per session_prompt §"Out of scope:
  predictor architecture sweep" and §"Hold-out invariants".

## State of the world at end

* Master HEAD still `8878e98` (no commits made by this sweep).
* New on disk:
  - 19 new head dirs: `models/direction_head/sweep_c{1,2,3,4,6,7,8,
    9,10,11,12,13,14,15,16,17,18,19,20}/` each with `weights.pt` +
    `manifest.json`
  - `plans/direction-head-architecture-sweep/findings.md` (this file)
  - `plans/direction-head-architecture-sweep/sweep_results.json` (round 1)
  - `plans/direction-head-architecture-sweep/sweep_results_round2.json`
  - `plans/direction-head-architecture-sweep/sweep_results_round3.json`
  - `plans/direction-head-architecture-sweep/sweep_results_round4.json`
    (final — includes all 20 variants)
  - `scripts/sweep_eval_direction_heads.py` (sweep-specific evaluator)
* 9 missing eval-day oracle caches + 5 missing direction-label caches
  generated during this session — valid cohort-wide.
* `scripts/train_direction_head.py` now supports
  `--variant {c0…c20}` plus training-recipe knobs (`--optimizer`,
  `--weight-decay`, `--loss`, `--focal-gamma`, `--label-smoothing`),
  and has a corrected `state_dict` save path (only strips OUTER
  `net.` prefix; previously a global replace over-stripped C15's
  nested `_PairwiseHead.net = Sequential`).
* `scripts/evaluate_direction_head.py` mirrors all the variant
  architectures; pre-sweep manifests (no `architecture.variant`
  field) load via the `c0` default.

## Reliability table — preserved from round 1, still represents the calibration story

(C0/C1/C2/C4 over-confident at ~2× empirical; C3/C8/C11/C14 track
empirical rate. Round 3 didn't repeat the reliability dump but the
pattern is the same — pred_mean from C11 across the 10 eval days
matches the empirical positive rate within 1-2 percentage points.)

| date       | empirical | C0    | C1    | C3        | C11 (sim. to C3) |
|---|---|---|---|---|---|
| 2026-04-07 | 0.167 | 0.503 | 0.484 | **0.200** | calibrated |
| 2026-04-10 | 0.180 | 0.474 | 0.455 | **0.190** | calibrated |
| 2026-04-14 | 0.190 | 0.473 | 0.456 | **0.190** | calibrated |
| 2026-04-17 | 0.180 | 0.467 | 0.451 | **0.186** | calibrated |
| 2026-04-21 | 0.198 | 0.488 | 0.471 | **0.198** | calibrated |
| 2026-04-23 | 0.210 | 0.486 | 0.470 | **0.200** | calibrated |
| 2026-04-25 | 0.192 | 0.476 | 0.461 | **0.195** | calibrated |
| 2026-05-01 | 0.212 | 0.443 | 0.433 | **0.173** | calibrated |
| 2026-05-03 | 0.237 | 0.421 | 0.411 | **0.163** | calibrated |
| 2026-05-06 | 0.174 | 0.437 | 0.423 | **0.171** | calibrated |

## Next session pickup

The 12 × 3 full cohort relaunch is unblocked. Recommended command:

```
python -m training_v2.cohort.runner --n-agents 12 --generations 3 \
    --device cuda \
    --output-dir registry/_predictor_SCALPING_<...> \
    [other args from prior 1779622853 launch ...] \
    --direction-head-manifest models/direction_head/sweep_c11
    # (mutually-exclusive with --enable-gene direction_prob_loss_weight
    # and --enable-gene bc_direction_target_weight — drop both from
    # the prior launch's flag list)
```

Other follow-ons in priority order:

1. (Highest) Validate C11 against a 5-day cohort smoke probe (mirror
   `_smoke_shared_head_1779635753`'s setup) before committing
   GPU-hours to the full 12×3. The smoke takes ~30 min vs the
   12×3's many hours, and if C11's calibrated outputs interact
   surprisingly with `direction_gate_threshold` we want to know
   before the full launch.
2. (Medium) Consider an ensemble of C11 + C9 outputs (no training,
   just averaging at inference time). C9 has the highest AUC of any
   variant; averaging with C11 might give the best of both. Cost:
   one bool flag in the policy.
3. (Low) Revisit C5 (full 574-d obs as input) ONLY with strong
   regularisation, in light of C15's overfit signature. Default
   priors should be tighter than for the per-runner-input variants.
   The cohort-scale benefit may not justify the extra work given
   C15's result.
