# fc-cost-probes findings — 7 probes, 35 trained agents, ZERO lever-signal movement

**Status:** complete. **Conclusion: at 5-agent × 7-day cohort scale,
no reward-side or architectural intervention moves the per-agent
fc / cl / bets behaviour. The bottleneck is cohort scale / signal-
to-noise, not the levers themselves.**

The probes ran 2026-05-17 16:48 → 23:24 BST (~6.5h wall) on the
same gate config tnv3 used (raceconf, fc=120 in training).
Baseline = tnv3 gen-0 (n=12, 10-day eval): pnl=−£46/d, fc_n=54,
fc_£=−£86/d, cl_n=9, bets=178, locked=+£88/d, naked_span=£227/d.

## Probe roster

| # | Probe | Lever | Train | Outcome |
|---|---|---|---:|---|
| 1 | **A** | `close_signal_bonus=10` | 3d | NO BITE |
| 2 | **B** | `open_cost=0.5` | 3d | NO BITE |
| 3 | **C** | A + B + `mature_prob_loss_weight=3.0` | 3d | NO BITE |
| 4 | **O** | `close_signal_bonus=50` (5× A) | 3d | NO BITE (magnitude ruled out) |
| 5 | **H** | `force_close_seconds=180` (was 120) | 3d | NO BITE (timing ruled out) |
| 6 | **A2** | A repeated with 7 train days | 7d | NO BITE (sample size ruled out) |
| 7 | **D** | new `fc_prob_head` aux head + 7 train days | 7d | NO BITE (representation ruled out) |

## Aggregate cohort-mean numbers across all 7 probes

Mean cohort-mean (i.e. mean across the 7 probe-means) for each lever-signal metric:

| Metric | Baseline | Probe-mean | Δ |
|---|---:|---:|---:|
| pnl | −£46/d | −£18.5/d | +£27.5 (driven by naked tailwinds) |
| **fc_n** | **54** | **51.5** | **−2.5 (within noise)** |
| **cl_n** | **9** | **8.7** | **−0.3 (UNCHANGED OR LOWER)** |
| **bets** | **178** | **172** | **−6 (barely moved)** |
| fc_£ | −£86/d | −£96/d | −£10 (slightly worse) |
| locked | +£88/d | +£93/d | +£5 (slightly higher) |
| naked_span | £227/d | £93/d | −£134 (smaller eval window, not lever effect) |

**The lever signals (fc_n, cl_n, bets) sit within ±3 of baseline
across all 35 trained agents under all 7 distinct levers.**

## The "+£30-40/d wobble" ghost

Every probe reports a "mean pnl improvement" of +£17 to +£44/d
versus baseline. Reading the per-agent rows reveals these are
**consistently traceable to 1–2 agents per probe catching naked
tailwinds** on 1–2 days inside the 3-day eval window. Examples:

| Probe | Tailwind agent | Naked/d | Pnl/d |
|---|---|---:|---:|
| B | 8c7bdabc | +£48 | +£52 |
| B | 1c59ffd2 | +£34 | −£7 (but day max +£278) |
| C | ed51f840 | +£45 | +£32 |
| O | 22518d17 | +£59 | +£69 |
| O | 554ac85d | +£28 | +£18 |
| A2 | c321fb90 | +£43 | +£76 |

These are the same naked-tailwind agents we documented chasing
across tnv2 (4c217d70 +£19 in-sample → −£49 held-out). At the
5-agent × 3-day-eval scale, a single agent catching one
+£500/d naked day shifts the cohort mean by +£33/d. None of these
agents would survive held-out reeval — same pattern as tnv2's
4c217d70.

## Meta-findings (what each probe ruled out)

| Hypothesis | Probe(s) | Refuted? |
|---|---|---|
| Close bonus too small (£1 default) | A (£10), O (£50) | YES — £50 gave cl_n 8.4 LOWER than baseline 9 |
| Per-tick gradient missing (open_cost flat) | B (0.5 pinned) | YES — bets 174 vs baseline 178 |
| Lever combo unlocks each | C (A+B+mature_prob=3.0) | YES — same flat lever signals |
| Force-close timing | H (T-180 vs T-120) | YES — fc_n 52, fc_£ −£96 (worse per event) |
| 3 train days too short | A2 (7 train days) | YES — cl_n 7.7 LOWER than 3-day-A's 9.7 |
| Missing representational pathway | D (fc_prob_head) | YES — cl_n 7.7, fc_n 51.5, bets 165 |

The remaining hypothesis after all 7 probes is **cohort-scale signal-to-noise**:

- Each agent runs 3 train days × ~70 races/day = ~210 race-episodes
- Each race has ±£500/d naked variance from a few specific runners
- The reward-side gradient against fc cost averages to roughly
  −£86/d of fc cost — comparable in size, but spread across 600+
  opens per race so each open's gradient is tiny (~−£0.14)
- PPO's value-function noise at this sample size cannot distinguish
  −£0.14 per open from the ±£10-50 naked variance per pair
- The fc_prob_head ADDS a representational pathway but the gradient
  to make use of it lives in the same swamped signal

The architectural intervention (D) DID very mildly nudge the
policy toward fewer opens (bets 165 vs baseline 178, −7%) and
slightly lower locked floor (+£82 vs +£88). The new column is
having SOME effect — just not enough at this scale to clear the
naked-variance noise.

## Implementation summary

Three pieces of permanent infrastructure landed during the probe work
that survive beyond this finding:

1. **`close_signal_bonus` as env-kwarg** (`env/betfair_env.py`).
   Reward-overrides whitelist entry; default 1.0 = byte-identical.
   Allows future probes to tune without code changes.

2. **`tools/show_probe_status.py`** — chronological per-agent panel
   + bite-verdict thresholds per metric. Lifted the data-extraction
   from `show_cohort_status.py` and added the bite-verdict logic.
   Reusable for any future small-cohort probe.

3. **`fc_prob_head` aux head** (`agents_v2/discrete_policy.py`).
   Gated behind `enable_fc_prob_head=False` default — byte-
   identical to pre-probe-D when off. New strict-fc label function
   `assign_per_transition_fc_labels` in
   `training_v2/discrete_ppo/aux_labels.py`. BCE wired in
   `training_v2/discrete_ppo/trainer.py` via
   `fc_prob_loss_weight` (gene + reward-overrides passthrough).
   53 existing v2 policy tests pass with the default off.

The fc_prob_head infrastructure is the cleanest contribution —
even though D didn't bite at 5×7 scale, the head is correctly
trained against the right label and feeds actor_input as designed.
At a larger cohort scale (recommendation 1 below) it should
provide measurable lift.

## Recommended next steps

Two paths, ranked by expected info-per-hour:

### Path 1 — Larger cohort with the existing levers (PREFERRED)

Re-run the most promising combination at full cohort scale: 12–20
agents × 5–8 generations. The cross-agent GA selection compounds
the per-agent lever signal across generations — even a +£10/d
gradient that's invisible at 5 agents becomes selectable when 12
agents compete and the best 6 breed. The infrastructure to test
this is already in place from tnv3: same runner, same gates, just
swap `--composite-score-mode day_pnl_per_std` to use the day_pnl
selector that includes fc cost in numerator.

Recipe:
- `--n-agents 12 --generations 8 --days 13 --n-eval-days 10`
- `--reward-overrides force_close_before_off_seconds=120 enable_fc_prob_head=true fc_prob_loss_weight=3.0 open_cost=0.5`
- `--composite-score-mode day_pnl_per_std`
- Raceconf gate (current best)
- Wall: ~28h on the GPU we just freed

This is the "if anything bites, this will show it" run. It combines
the architectural intervention (D), the per-tick open_cost gradient
(B), and the day_pnl_per_std selector that tnv3 ran but couldn't
exploit at gen 1 because the underlying levers weren't there.

### Path 2 — Reduce naked variance at the source

If we want to keep cohort scale small for iteration speed, the only
remaining angle is to reduce the ±£500/d naked variance the policy
has to learn against. Three options:

- **Tighter `lay_price_max`** (e.g. 10 instead of 20). The
  lay-quality-gate already capped at 20; tightening to 10 removes
  more outsider lays where most naked-£500 days originate.
- **Tighter `predictor_p_win_back_threshold`** (e.g. 0.30 instead
  of 0.20). Fewer races qualify, but each is a higher-confidence
  signal — naked variance per opened pair drops.
- **Back-only mode** (no lay opens). Eliminates a whole class of
  naked variance; tests whether the lay side is structurally
  problematic.

Each is one CLI flag change, no code. Could fit 3-4 of these
into the next 8h.

### Combination

The strongest play is probably **Path 1 + the tighter lay_price_max
from Path 2 baked into the gate**. Run the 12-agent × 8-generation
cohort with the full reward-side stack AND lay_price_max=10.
That's one cohort, ~28h GPU, and it tests the entire fc-cost-
intervention hypothesis at the scale we have evidence is needed.

If that cohort doesn't bite either, the conclusion is harder
still: scalping at this gate / data window combination has a
structural fc-cost ceiling, and the productive next move is to
revisit the gate (different predictor thresholds, race-confidence
floor, different exclude-days windows) rather than to keep
hunting reward-side levers.
