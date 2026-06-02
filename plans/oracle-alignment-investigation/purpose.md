# Oracle alignment: the missing-99% problem

## What

Pair-level analysis on the recipe-sensitivity-sweep cohort
(2026-04-10) revealed that ~99% of oracle-identified scalp
opportunities go unrealised by agents in the cohort. The breakdown:

| metric | value |
|---|---|
| Oracle-positive (tick, runner) pairs on 2026-04-10 | 73,946 |
| Agent opens placed (43 agents, full day) | 3,140 |
| Agent opens / oracle opportunities | 4.2% |
| Cohort matured pairs | 7 |
| Cohort mat / oracle volume | **0.009%** |
| Cohort mat / agent opens | 0.22% |

Two independent failures compound:

### Failure 1: Selection (where to open)

From the price-band + oracle comparison:
- Oracle says 82% of opportunities are at price ≥ 5.
- Cohort places 90% of opens at price < 5.

The cohort is fishing in the wrong band by ~5:1.

### Failure 2: Holding (don't kill the trade)

From the per-pair outcome distribution:

| outcome | % of opens | mean P&L |
|---|---|---|
| agent_closed | **68.8%** | -£2.01 |
| force_closed | 20.9% | -£0.46 |
| stop_closed | 9.7% | +£0.25 (median -£2.33) |
| naked | 0.4% | -£1.41 |
| **matured** | **0.2%** | **+£0.31** |

69% of pair opens are killed by `close_signal` before they can
mature. Of those, 82% close at a loss (mean -£5.19). The agent
learned to use close_signal as a stop-loss, not as a profit-take.
Total burn from close_signal across the cohort on 2026-04-10:
**-£4,333**.

Matured pairs lock +£0.31 each. The cash signal from "close now
at -£5" is structurally larger and immediate; the cash signal
from "wait and mature at +£0.31" is delayed and small. PPO
optimises for the immediate signal, so the policy learns
close_signal-as-stop-loss.

## Why both matter

If we only fixed selection (agent opens in the right band) but
not holding, mat% can't exceed 0.22% because the agent would
still close them. The cap is set by the close_signal usage.

If we only fixed holding (agent doesn't fire close_signal early)
but not selection, the agent would now hold-to-maturity but
on bad opens — high force-close cost would crush P&L.

Both need to be addressed. Selection is the upstream problem;
holding is the downstream amplifier.

## Hypotheses to test

### H1: close_signal-as-stop-loss is the bottleneck

Remove the agent's ability to fire close_signal (action-mask it
to always-NOOP) and run a probe. Predicted outcomes:
- mat% jumps significantly (5-10×) because pairs now have to
  mature, stop-close, force-close, or go naked.
- Per-pair P&L goes DOWN initially (no more loss-cutting) BUT
  matured rewards compound differently in PPO's value estimate.
- After enough PPO training, policy SHOULD learn to be more
  selective at OPEN time, since it can no longer escape bad
  opens.

### H2: BC pretrain alignment fixes selection

The oracle's distribution is at price 5-30, low champion_p_win.
BC pretrain with the oracle target should pull the policy's
opening distribution toward that band. The current cohort
disabled BC for sensitivity-sweep cleanliness, but production
needs it on.

Predicted outcomes (with BC=1000 vs BC=0):
- Open-band distribution shifts toward price 5-30.
- mat% increases because more opens land at oracle-positive
  moments.

### H3: Reward shape for "ride to maturation"

A shaped reward that pays the agent for HOLDING a pair past N
ticks (or for getting close to the passive's price) might
counter the close_signal cash gravity. This is more invasive
(new reward term) but plausible.

## Proposed probes

### P1: Mask close_signal cohort-wide (low risk, fast signal)

- 4 agents, 3 train days, 5 eval days, BC pretrain off.
- Add a cohort flag `--mask-close-signal` that forces NOOP at
  every CLOSE_i action slot. Requires small code change.
- Compare to the recipe-sensitivity-sweep baseline (same train/eval).

Expected wall: ~20 min cohort + ~5 min code change.

### P2: BC pretrain on vs off (this isolates the selection
issue)

- 2 cells × 4 agents × 3 train days × 5 eval days.
- Cell A: BC pretrain = 0 (current cohort, baseline).
- Cell B: BC pretrain = 1000 (production default).
- Compare open-band distribution, mat%, day_pnl.

Expected wall: ~40 min for two cells.

### P3: H1 + H2 combined (probe the joint effect)

- 1 cell × 4 agents: close_signal masked AND BC=1000.
- Should produce a meaningfully different phenotype:
  selective openings + forced holding.

Expected wall: ~25 min.

Total: ~85 min for the full set. Can run after the env-side
sweep finishes.

## Hard constraints

- Same training & eval days as recipe-sensitivity-sweep.
- Frozen C11 head loaded.
- No new training data sources.
- The close_signal mask must be a cohort-wide bool flag, not a
  gene (we want to ISOLATE its effect, not have the GA tune it).

## Acceptance

After P1+P2+P3:
- If P1 alone moves mat% from 0.2% → 5%+: holding is the
  bottleneck.
- If P2 alone moves open-band distribution toward price 5-30:
  BC alignment works.
- If P3 produces an agent with mat% > 10% and positive day_pnl:
  we have a viable production phenotype.

## Out of scope

- Direction-predictor gate experiments (see
  `plans/direction-predictor-mechanism/`).
- Architecture changes to the C11 head.
- New auxiliary heads.
