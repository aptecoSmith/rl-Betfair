# BC + Exit Recovery — Round 3 experiments

## Why

The 2026-05-25 close-penalty + BC-pretrain probe
(`plans/oracle-alignment-investigation/findings.md`) cleanly
identified the policy's failure mode:

- **BC pretrain (1000 steps) fixes the SELECTION failure** — mat% 5×
  (1.3% → 6.2%), locked +£14 (+70%). Confirms the oracle-alignment
  hypothesis from the price-band investigation.
- **BC pretrain BREAKS the EXIT strategy** — close_signal collapses
  (cls% 31.7% → 0.8%), force-close rate rockets (63% → 89%), naked
  losses triple (-£28 → -£86). Net day_pnl WORSENS by £148/day.
- **CLOSE_SIGNAL_BONUS is a dead lever** — 5× swing in penalty
  magnitude moved cls% by ≤ 1.4pp.

The mechanism is now structurally clear: BC's training labels only
include OPEN_BACK / OPEN_LAY targets (see
`training_v2/discrete_ppo/bc_pretrain.py:191-198`). All BC gradient
pulls these two action probabilities UP; close_signal's prior decays
as a side-effect of the softmax. Post-BC the policy has been
relocated to the oracle's opening distribution but with its exit
action effectively un-trained.

## What we want to learn

**Can we land BC's +£14 locked-pnl benefit WITHOUT the £153 force-close
disaster?**

The question divides into three sub-questions, one per group below:

1. Is there a BC dose (between 0 and 1000 steps) that captures most of
   the selection benefit while preserving enough close_signal prior?
2. Can a per-step shaped or env-side mechanism re-establish exit
   behaviour on top of an aggressively-BC'd policy?
3. Do the two confirmed env-side levers — pwin_back gate (C2) and
   BC pretrain — compose to a net win, or do they interfere?

## Experiments

8 cells × 4 agents × 1 gen × 3 train × 5 eval days, ~25 min wall each.
Total wall ~3.5h. Same train/eval days as rounds 1 & 2 for direct
comparison.

### Group A — BC dose-response (3 cells)

Establish the BC selection-vs-exit Pareto curve before partnering BC
with other mechanisms.

| cell    | bc_pretrain_steps | other knobs        |
|---------|------------------:|--------------------|
| **E1_bc200**  |  200 | (defaults)         |
| **E2_bc500**  |  500 | (defaults)         |
| **E3_bc2000** | 2000 | (defaults)         |

Reference points (already in registry):
- BC=0 → PC0 (-£167, mat 1.3%, cls 31.7%, fc 63.1%)
- BC=1000 → PC3 (-£315, mat 6.2%, cls 0.8%, fc 89.1%)

E2 (BC=500) is the partner-reference for groups B and C.

### Group B — BC=500 + exit mechanism (3 cells)

Pin BC at the mid-dose (assumed best-of-A) and test three orthogonal
exit-recovery mechanisms:

| cell | bc_pretrain_steps | partner |
|------|------------------:|---------|
| **E4_bc500_fc60**  | 500 | `force_close_before_off_seconds=60` (cuts naked exposure window in half) |
| **E5_bc500_matbonus5** | 500 | `matured_arb_bonus_weight=5.0` (strong positive shaped gradient on natural maturation; default is 0) |
| **E6_bc500_opencost05** | 500 | `open_cost=0.5` (selective-open shaping; each open that doesn't mature or close costs £0.5, refund on resolution) |

Each partner targets the failure from a different angle:
- **E4** accepts that BC removes close_signal but bounds the
  resulting force-close exposure by halving the naked window.
- **E5** rewards the BEHAVIOUR we want to see more of (natural
  maturation), giving PPO a direct positive gradient on the
  outcome BC delivers more of. The +£14 locked benefit from BC
  becomes a stronger gradient via the matured_arb_bonus.
- **E6** penalises the behaviour we want to see less of (opens
  that force-close), giving PPO a direct negative gradient on
  the new failure mode BC creates. Refund mechanics already
  exclude force-close from the refund (`CLAUDE.md` "Selective-open
  shaping").

### Group C — Combine known winners (2 cells)

| cell | env levers | BC | partner |
|------|-----------|---:|---------|
| **E7_pwinback_bc500** | `--predictor-p-win-back-threshold 0.20` | 500 | none |
| **E8_pwinback_bc500_matbonus5** | `--predictor-p-win-back-threshold 0.20` | 500 | `matured_arb_bonus_weight=5.0` |

E7 tests whether the two confirmed env-side levers compose.
C2 (pwin_back alone) delivered +£64/day vs C0; if pwin_back is
filtering structurally-bad opens at the source, BC's downstream
re-alignment should compose cleanly with it. E8 layers in the
most likely Group-B winner (matured_arb_bonus) as a triple-stack
candidate for the deploy recipe.

### What's NOT in this round (and why)

- **Gradient knobs** (`learning_rate`, `gae_lambda`) cannot be
  pinned cohort-wide via existing CLI; pinning would require a
  new flag. Deferred to a follow-up round. The cohort runner's
  per-agent random gene draws produce too much variance at n=4
  for a single-cell pin-test anyway — needs n=12+ for that signal.
- **Direction gate at threshold 0.20 or lower** — D3 already
  tested this; the threshold-response curve is flat-and-harmful
  across all calibrated thresholds. Dropped permanently.
- **arb_spread_target_lock_pct sweep** — orthogonal to BC and
  exit recovery; queue as a separate round if BC doesn't deliver.
- **Modified BC formulation that includes close labels** —
  engineering effort (oracle close-side data, BC loss restructure).
  Deferred until we know BC + exit-partner can work at all.

## Acceptance criteria

The operator's stated success shape (2026-05-25): **reasonable number
of bets, high maturation rate, eventually profit**. Translated to
measurable targets:

### Per-cell deploy-candidate gate

A cell qualifies as a deploy candidate if ALL four are true:

| metric          | target                  | rationale |
|-----------------|-------------------------|-----------|
| **opens/day**   | 100 – 180               | "reasonable number of bets". Below 100 = policy collapsing (cf. C7 all_on at 88). Above 180 = paradoxical over-opening (cf. D2, PC3 at 200+). |
| **mat%**        | ≥ 5%                    | "high maturation rate". PC3 hit 6.2% so 5% is reachable. Baseline 1.3% is the floor we must beat clearly. |
| **fc%**         | ≤ 50%                   | Inversely correlated with mat%; bounds the force-close P&L drag. Baseline 63%, PC3 89%, both unacceptable. |
| **day_pnl**     | > -£100 (better than C2)| "eventually profit" — this round's bar is **trending toward** profit, not profitable yet. C2 (-£102) is the best result from rounds 1–2; we need a clear improvement on that. |

PLUS the **anti-lottery guard:** locked_pnl > 2 × σ(naked_leg).
Prevents naked-variance lottery winners (like C7) from sneaking past
on day_pnl alone. From `naked_variance_primary_metric` memo.

### Group-level reads (information gain even if no cell qualifies)

**Group A (BC dose-response):**
- Maps the BC selection-vs-exit Pareto curve. Useful even if every
  cell misses the bar — tells us the right partner-dose for
  follow-up rounds.
- E1 (BC=200) is the lowest-risk: if it lifts mat% above 1.3% while
  preserving cls% > 15%, light-touch BC is a viable cohort default.

**Group B (BC=500 + exit mechanism):**
- The diagnostic question: of {tight fc-window, mat-bonus,
  open_cost}, which one rescues BC's exit failure?
- Headline win = mat% ≥ 5% AND opens ≤ 180 AND fc% ≤ 50% on the
  same cell. If exactly one of E4/E5/E6 lands, that's our
  exit-recovery mechanism going forward.
- If none land but all three improve over PC3 (-£315): BC's
  selection benefit is real but no single partner is enough; next
  round explores compounding two partners.
- If none land AND none improve: BC's exit collapse is structurally
  uncombinable with shaped/env mechanisms; need to modify BC itself
  (close-action labels).

**Group C (combine known winners):**
- The deploy-candidate question. E7 / E8 are the most plausible
  qualifying cells.
- If E7 qualifies but E5 (its Group B sibling) doesn't, pwin_back's
  bite is what closes the gap — confirms "remove-decisions"
  primacy from the 2026-05-20 memo.
- If E8 qualifies AND beats E7 by > £30/day, the triple-stack
  composes — and the production recipe is pwin_back + BC=500 +
  matured_arb_bonus.

### Failure-mode signals

A cell that hits day_pnl > -£50 but **mat% = 0% or opens < 90** is a
naked-variance lottery winner (C7 phenotype). Do not chase. Report
it but exclude from deploy-candidate ranking.

A cell that hits mat% ≥ 5% but **fc% > 70%** is the PC3 phenotype
(selection won, exit broken). Report the partial win but exclude
from deploy-candidate ranking — the force-close P&L drag makes it
unshippable.

## Hard constraints (same as rounds 1 & 2)

- Same train days (`2026-04-06, 2026-04-08, 2026-04-09`) and eval
  days (`2026-04-10, 2026-04-17, 2026-04-21, 2026-05-03, 2026-05-06`).
- All 3 predictors loaded; lean obs.
- Frozen C11 direction head loaded.
- Policy-side direction gate OFF (D-cells decided this).
- `close_feasibility_max_spread_pct=0.05`,
  `matured_arb_expected_random=0.0` cohort-wide.

## Estimated wall time

8 cells × ~25 min = 3.3h. Queue as a chained wrapper after the
current pipeline completes (already done at 15:40 BST 2026-05-25).
Can launch immediately.

## Out of scope (queued follow-ups)

- **BC label augmentation** — add CLOSE / NOOP targets to the
  BC sample set so close_signal isn't trained-out as a side-effect.
  Implementation needs to invent oracle close-side labels (the
  current oracle scan only emits OPEN labels).
- **D4-equivalent probe** (`--direction-signal-gain 0`) — confirm
  the C11 obs columns add value vs zero. Owed from
  `direction-predictor-mechanism/findings.md`.
- **Env-side gate refusal counters** — wire the placeholder
  counters in `env/betfair_env.py:1186` so we can observe gate
  activity directly. ~30 min effort.
- **Gradient-knob pin CLI flags** — add `--learning-rate-pin`
  and `--gae-lambda-pin` for cohort-wide pinning. ~1h. Enables
  the deferred Group D from this round's initial draft.
