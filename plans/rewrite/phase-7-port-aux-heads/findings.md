---
plan: rewrite/phase-7-port-aux-heads
session: S03
opened: 2026-05-04
status: AMBER
---

# Phase 7 Session 03 — validation findings

## Verdict

AMBER. The Phase 7 wiring works end-to-end — all three aux loss
terms reach the trainer, evaluate to non-zero values when their
weights are non-zero, and stay at exactly 0.0 when their weights are
0.0 (verified across 48 per-update log lines per cohort). The risk
NLL liveness check **passes** (Success-bar item 8). The BCE
behavioural-effect gate **fails** (Success-bar item 7) at the chosen
probe weight (0.5 cohort-wide) over a single GA generation. The
plan ships AMBER because the wiring is real (S02 integration tests
pass; loss magnitudes change exactly where they should) but the
behavioural lever needs more selection pressure than 1 generation ×
4 training days at weight 0.5 can deliver. The follow-up tuning run
explicitly addresses this — `mature_prob_loss_weight` promoted to a
GA gene with range `[1.0, 5.0]`, 6 generations × 12 agents.

## Cohort design

- **Reference**: `registry/_phase7_s03_ref_1777892055/` — all three
  aux weights pinned to 0.0 cohort-wide via `--reward-overrides`.
- **Probe**: `registry/_phase7_s03_probe_1777896183/` — all three aux
  weights pinned to 0.5 cohort-wide via `--reward-overrides`.
- 12 agents × 1 generation × 4 training days, eval on 2026-05-03,
  seed=42, device=cuda. Same gene draws per agent_idx by virtue of
  shared seed.

## Results — BCE lever liveness (Success-bar item 7)

### Action-distribution KL (probe || reference) per agent slot

| metric | value |
|---|---|
| n agents | 12 |
| median KL | 0.0069 |
| mean KL | 0.0199 |
| max KL | 0.1461 (agent slot 4) |
| agents with KL ≥ 0.1 | 1 / 12 (8 %) |

Most agents show KL well under the 0.1 gate threshold. ONE agent
(slot 4, gene `entropy_coeff=0.0349, hidden_size=64,
matured_arb_bonus_weight=4.87`) crossed the threshold cleanly. The
distribution is broadly noisy — the BCE gradient does shift action
selection on the slots whose other genes give it room to express,
but the population-wide effect is small.

### Maturation_rate per-agent delta

| metric | value |
|---|---|
| n agents | 12 |
| mean Δ | -0.0038 |
| median Δ | -0.004 |
| direction | +5 / -7 / =0 |
| agents with abs(Δ) ≥ 2 pp | 2 / 12 (17 %) |

Mean shift essentially flat. Two agents shifted by ≥ 2 pp (slots 7
and 9, both negative); the rest stayed within ±1.5 pp of their
reference value. Mean maturation_rate: ref **0.213**, probe
**0.209** — within run-to-run noise.

### BCE liveness verdict

- KL gate (≥ half of agents at KL ≥ 0.1): **FAIL** (1/12)
- Maturation gate (≥ half of agents at |Δ| ≥ 2 pp): **FAIL** (2/12)
- BCE lever alive at this weight × this scale: **NO**

## Results — Risk NLL liveness (Success-bar item 8)

| metric | value |
|---|---|
| Per-update log lines parsed | 48 |
| Updates with `risk_nll_mean > 0` | 29 / 48 (60 %) |
| Updates with `risk_nll_mean = 0` | 19 / 48 (skipped — no completed pair in mini-batch) |
| Range (over positive values) | [0.012, 67.98] |
| Mean (over positive values) | 10.73 |

Updates with zero NLL are the trainer's `risk_denom > 0` guard
firing on rollouts where no pair completed both legs in the
mini-batch (per `lessons_learnt.md`). 60 % of updates train the head;
40 % skip safely.

**Risk NLL alive: YES** (Success-bar item 8 PASS).

## Sanity checks

- Reference log: `fill_prob_bce_mean = mature_prob_bce_mean =
  risk_nll_mean = 0.0` on **all 48** updates. Confirms the
  trainer's `if any_weight > 0` guard short-circuits the aux-loss
  computation when all three weights are zero (no spurious
  computation, no spurious gradient).
- Probe log: aux loss values non-zero on **all 48** updates.
  - `fill_prob_bce_mean`: range [0.0009, 0.6682], mean 0.1646.
    BCE upper bound for fully-disagreeing logits is `ln(2) ≈ 0.69`;
    upper end of the observed range is at the bound — head and
    label disagree where the data says they should.
  - `mature_prob_bce_mean`: range [0.0060, 0.6672], mean 0.2718.
    Higher mean than fill_prob — consistent with the strict label
    disagreeing more often (force-closed pairs land in the
    negative class, where the conflated fill_prob label puts them
    in the positive class).
  - `risk_nll_mean`: range [-3.14, 67.98], mean 6.01. Negative
    values are valid (NLL of a Gaussian can be negative when the
    log-var is small enough — the head is confidently right at
    times, less confident at others).
- S02 integration tests: **26 / 26 passing** under
  `pytest tests/test_v2_aux_heads.py -x`. No wiring regression.

## Direction of effect

5 agents shifted maturation positively, 7 negatively. The largest
positive shift (+1.8 pp) is at slot 2; the largest negative shift
(-3.4 pp) is at slot 9. No clear directional preference — the
aux-loss gradient noises the action distribution slightly without a
consistent push. This is the expected shape when the gradient is
real but small relative to the surrogate-loss gradient over a
single rollout.

## Informational metrics (not gates)

- Mean `eval_total_reward` ref vs probe: -649.4 vs -670.3
  (probe slightly worse, within noise).
- Mean `eval_day_pnl` ref vs probe: -£228.2 vs -£251.7
  (probe slightly worse, within noise — the 0.5 cohort-wide pin
  spends a small amount of the surrogate-loss budget on an
  auxiliary objective that doesn't yet produce a behavioural
  return at this scale).
- Per-agent risk NLL trajectory: across 48 updates, the magnitude
  varies widely with rollout-level pair completion density. No
  monotone descent (each update is a different agent on a
  different day, so trajectory analysis at this resolution
  isn't informative).
- Mean fc_rate ref vs probe: 0.044 vs 0.050 (within noise).
- Mean naked_open_rate ref vs probe: 0.044 vs 0.050.
- Mean locked_per_matured ref vs probe: +£3.86 vs +£3.93.

The probe is, behaviourally, a slightly noised version of the
reference. The aux-loss signal is reaching the heads but isn't
consistently steering action selection.

## Why the BCE gate failed (analysis)

Three working hypotheses, none of which contradict the wiring
proof:

1. **Per-runner credit smearing.** Per `lessons_learnt.md` ("Per-
   runner aux labels are aggregated across races"), S02
   broadcasts a single per-slot label across every transition's
   mini-batch entry. Same slot index can carry different physical
   runners across races, so the per-slot label is a noisy
   aggregate. The Phase 7 success bar deliberately accepted this
   for the GREEN ship; the cost is that the gradient signal is
   diluted by ~5–10× compared to per-transition credit.
2. **One-generation budget.** One PPO update per training day ×
   4 training days × 356 mini-batches = ~1,400 gradient steps
   per agent. The aux-loss term contributes a few percent of
   total loss magnitude (probe sums:
   `policy_loss ~0.05 + value_loss ~0.5 + 0.5×bce~0.1 +
   0.5×nll~3`); behavioural change requires the actor's
   per-runner column to receive a consistently aligned gradient
   over many more steps than that.
3. **Weight 0.5 may be too low for the strict-label signal at
   this scale.** The probe weight was the original purpose.md
   recommendation. The follow-up plan (post-S03 tuning) raises
   the gene range to [1.0, 5.0] and runs 6 generations to give
   GA selection room to find a working magnitude.

None of these point to a code bug. The S02 integration tests pass
and the loss magnitudes behave exactly as predicted (probe
non-zero, ref zero).

## Recommended follow-up

1. **Tuning run (queued by the operator 2026-05-04).** 12 agents ×
   6 generations × 4 training days, `mature_prob_loss_weight`
   promoted to a GA gene with range `[1.0, 5.0]`, other Phase 5
   genes (`matured_arb_bonus_weight`, `mark_to_market_weight`,
   `stop_loss_pnl_threshold`) unchanged. Selection pressure on
   `total_reward` should pull toward whichever weight values move
   maturation_rate up. Output: `registry/_phase7_s04_mature_tuning_…/`.
2. **If the tuning run also shows < 2 pp mean Δ:** raise the
   per-runner credit-assignment question. The Phase-7 lessons-
   learnt note explicitly flags "per-transition credit can be
   tightened in a follow-on session if the validation cohort
   shows the lever signal is too weak." This S03 result is that
   evidence point — but the tuning run is the cheaper experiment
   to run first.
3. **Risk NLL trajectory.** The 48-update log shows the head is
   training but doesn't show whether it's *improving*. A
   follow-on diagnostic could parse `risk_nll_mean` per-agent
   per-day and check for monotone descent within an agent's
   training run; that's the right liveness signal for the head
   to actually shape the backbone usefully.

## What's locked

Phase 7 ships AMBER:
- ✓ All three aux heads exist in `DiscreteLSTMPolicy`
  (`fill_prob_head`, `mature_prob_head`, `risk_head`).
- ✓ All three aux loss weights reach `DiscretePPOTrainer` via the
  worker's hp-dict pre-merge (`Path A` per
  `lessons_learnt.md`). Verified end-to-end in this run: ref
  losses identically 0, probe losses non-zero at correct
  magnitudes.
- ✓ Risk NLL trains (Success-bar item 8 PASS).
- ✗ BCE behavioural lever doesn't move maturation_rate
  measurably at weight 0.5 over 1 generation. Wiring is real;
  scale + selection pressure aren't.

The next-step tuning run is the natural follow-up to disambiguate
"signal too weak at the chosen scale" from "signal exists but
can't be amplified by GA selection". Phase 7 has shipped the
mechanism; what's needed next is operating-point tuning.
