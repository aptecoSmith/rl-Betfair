# Oracle-alignment investigation — close-penalty + BC-pretrain findings

Wrapper: `run_after_env_sweep.sh`. Ran 2026-05-25 11:43 → 13:56 BST.
5 cells × 4 agents × 1 gen × 3 train days × 5 eval days. ~25 min/cell.

All cells share: frozen C11 direction head loaded, predictors active,
lean obs, force_close=120s, `close_feasibility_max_spread_pct=0.05`,
`matured_arb_expected_random=0.0`. NO policy-side direction gate.

## Top-line table

| cell                          | BC steps | close_signal_bonus | day_pnl | locked | naked  | forced  | closed | bets | opens | mat% | cls%  | fc%   |
|-------------------------------|---------:|-------------------:|--------:|-------:|-------:|--------:|-------:|-----:|------:|-----:|------:|------:|
| **PC0_baseline_bc0**          |        0 |   default (0.0)    | -166.6  | +19.1  | -28.2  | -106.5  | -51.1  |  263 |   134 | 1.3% | 31.7% | 63.1% |
| **PC1_close_penalty_-2_bc0**  |        0 |               -2.0 | -168.2  | +19.6  | -26.6  | -110.2  | -51.1  |  266 |   136 | 2.0% | 30.3% | 63.6% |
| **PC2_close_penalty_-5_bc0**  |        0 |               -5.0 | -169.2  | +17.8  | -20.8  | -116.3  | -49.8  |  266 |   135 | 1.5% | 31.3% | 63.7% |
| **PC3_bc1000_no_penalty**     |     1000 |   default (0.0)    | **-315.4** | +32.6 | -86.4 | **-259.6** | -1.8 | 454 |   231 | **6.2%** | 0.8% | **89.1%** |
| **PC4_bc1000_close_penalty_-2** |  1000 |               -2.0 | -303.4  | +34.4  | -72.0  | -263.8  | -2.1   |  466 |   238 | 5.9% | 0.6%  | 89.3% |

PC0 replicates C0 from the env-side sweep within ±0% (sanity ✓).

## Headline findings

### 1. CLOSE_SIGNAL_BONUS is a structurally weak lever

PC1 (`close_signal_bonus = -2`) and PC2 (`close_signal_bonus = -5`)
are statistically indistinguishable from PC0:

```
                day_pnl    cls%
  PC0 (0.0)     -166.6    31.7%
  PC1 (-2)      -168.2    30.3%
  PC2 (-5)      -169.2    31.3%
```

A 1.4-percentage-point movement in cls% (≈ 4% relative reduction)
across a 5× swing in the penalty magnitude. The shaped reward channel
on the `close_signal` action does not have learning authority at the
current probe scale (4 agents × 1 gen).

**Possible mechanisms** (not separated by this probe):

- GAE smoothing: at the typical 8-tick gap between `close_signal`
  and the close-leg settle, the penalty signal accumulates into the
  same return as the close-leg's realised cash P&L, and the cash
  signal dominates the gradient.
- close_signal is consumed by the policy as a stop-loss reflex
  trained at PPO step 0 on a strong negative cash signal
  (`plans/recipe-sensitivity-sweep/behavioural_findings.md` —
  close_signal kills 69 % of opens at adverse drift, 84 % of those
  are at -£ drift). The shaped penalty competes with a much larger
  cash gradient on the same action.

**Implication:** if we want to change close behaviour, we cannot do
it through `close_signal_bonus` as a shaped knob. The lever must
either alter the action structure (mask, remove the action) or alter
the cash dynamics (force-close window, equal-profit close sizing).

### 2. BC pretrain is the strongest behavioural lever found

PC3 (BC=1000 steps, no other change) produces the largest single
behaviour shift observed in any cell across this entire sweep:

| metric            | PC0     | PC3     | Δ      |
|-------------------|--------:|--------:|-------:|
| day_pnl           | -£166.6 | -£315.4 | -£148.8 |
| locked            |  +£19.1 |  +£32.6 |  +£13.5 |
| naked             |  -£28.2 |  -£86.4 |  -£58.2 |
| force_closed      | -£106.5 | -£259.6 | -£153.1 |
| **closed (cash)** |  -£51.1 |   -£1.8 |  +£49.3 |
| bets              |     263 |     454 |    +191 |
| opens             |     134 |     231 |     +97 |
| **mat%**          |    1.3% |    6.2% |   +4.9pp |
| **cls%**          |   31.7% |    0.8% |  -30.9pp |
| **fc%**           |   63.1% |   89.1% |  +26.0pp |

The shape is consistent across all 4 agents within PC3 (not a single
outlier). BC pretrain successfully shifted the agent's opening
distribution toward the oracle's preferred (price 5–30, low p_win)
region — mat% rose **5×** and locked_pnl rose **70%**. **Selection
worked.**

But the agent **lost its exit strategy entirely**:

- `close_signal` action collapsed from 51 closes/day → 1 close/day.
- Force-close rate rocketed from 63% → 89%.
- Opens nearly doubled (134 → 231).
- The naked term tripled in magnitude (-£28 → -£86).

The net result is **catastrophically worse** day_pnl despite the
selection win, because the force-close losses dominate the
locked-pnl gain by a factor of ~10×.

**Mechanism:** BC trains `actor_head` on oracle-target samples
(open at oracle-positive runners) but does NOT see close_signal in
its label set. The post-BC policy has been shifted in
representation-space toward "open more, at the oracle's distribution"
but its close_signal output prior has decayed in the process.

### 3. PC4 confirms close penalty cannot rescue BC

PC4 = PC3 + `close_signal_bonus = -2`. The combination is
**indistinguishable from PC3**:

```
              day_pnl   cls%   fc%
  PC3         -315.4    0.8%  89.1%
  PC4 (+-£2)  -303.4    0.6%  89.3%
```

Confirms that once BC has trained close_signal out of the policy,
adding a shaped penalty on close_signal does nothing — the policy
isn't using the action anyway, so the gradient never lands.

This is the cleanest possible refutation that close_signal_bonus is
the right lever for the "lost exit strategy" failure of BC pretrain.

## What this means for the bigger picture

### The agent failure mode is now **structurally identified**

Before this probe we knew:
- 99% of oracle-positive opportunities were being missed (selection
  failure).
- 69% of agent opens were killed via close_signal as stop-loss
  before maturation (holding failure).

After this probe we know:
- BC pretrain **fixes the selection failure** (5× mat% lift confirms
  the oracle-alignment hypothesis is correct).
- BC pretrain by itself **creates a worse holding failure** because
  it un-trains close_signal in the process.
- close_signal_bonus is **not the right lever** to recover the
  holding behaviour.

The next plan needs to find a way to land BC's selection benefit
WITHOUT the force-close catastrophe.

## Recommendation

- ❌ **Do not deploy BC=1000 as a standalone gene knob.** The cash
  catastrophe (-£148/day) outweighs any selection benefit.
- ❌ **Drop close_signal_bonus as a Phase-5 gene.** PC1/PC2/PC4
  collectively show it has no behavioural authority.
- 🔬 **BC pretrain is the most promising single lever found.**
  Recover it via the round-3 experiments (see
  `plans/bc-exit-recovery/purpose.md`) — combine BC with mechanisms
  that re-establish exit behaviour (tighter force-close window,
  matured_arb_bonus, open_cost shaping, BC dose-response).
- 🔬 **Investigate the BC implementation:** does BC actually emit
  close_signal target=0 on its samples? If yes, the close_signal
  decay is by-design and we need a different BC formulation. If no,
  the decay is a side-effect of un-grounding the prior and can be
  bounded.
- ✅ **Carry forward `pwin_back 0.20`** from C2 as the only confirmed
  env-side lever from this sweep.
