# Findings — EW force-close + arb_spread design investigation

Investigation triggered by [plans/force_close_and_arb_spread/session_prompt.md](session_prompt.md).
Cohort under analysis: `registry/_predictor_SCALPING_postfix_e3_cohort_1779530050`
(running, 1/30 agents evaluated as of writing). Sample is **one agent
(`557334ae`), 7 eval days, 405 pairs (788 bet legs)**. Caveats around
single-agent sample size below.

---

## The answer (one paragraph)

The operator's framing collapses on three points after looking at the
data. First: the empirical force-close cost is **not concentrated in
EW** — WIN markets carry the larger absolute drag (-£189 vs -£111
across 7 days, single agent) and a much higher force-close *rate*
(70.5 % WIN vs 41.4 % EW). Per-pair force-close cost is essentially
identical across market types (≈ -£1.30 each). Second: in the v2
discrete-action stack the agent **cannot pick arb_spread per-runner —
it cannot pick it at all**. The `DiscreteActionShim` hard-codes
`arb_ticks=20` for every OPEN_BACK / OPEN_LAY action ([agents_v2/env_shim.py:105](../../agents_v2/env_shim.py#L105),
[agents_v2/env_shim.py:313-319](../../agents_v2/env_shim.py#L313)),
and the only knob the GA evolves is a per-agent global multiplier
`arb_spread_scale ∈ [0.5, 2.0]` ([training_v2/cohort/genes.py:26](../../training_v2/cohort/genes.py#L26)).
So Q3 is structurally answered "the agent doesn't currently get the
choice"; the LEAN obs already carries per-runner microstructure
(`spread_pct`, `ltp_velocity_30/60`) but there's no action channel to
condition on it. Third: gen-0 agents on a freshly-reset registry are
random-init noise; the 76 % aggregate force-close rate is mostly
"agent fires aggressives anywhere and the passive at +20 ticks rarely
fills" — not a structural EW-vs-WIN asymmetry. The cleanest
single-change next step is to either pin `arb_spread_scale=0.5`
cohort-wide (halving the target to ~10 ticks → passive much more
likely to fill → force-close rate drops directly) OR promote the
arb_spread choice into the discrete action head per-runner so the GA
can actually learn to differentiate; both options laid out in §"Next
step" below.

---

## Q1: EW vs WIN force-close cost — empirical

### Method

`C:/tmp/analyze_bet_logs.py` iterates all `bet_logs/<uuid>/*.parquet`
files in the running cohort, groups bet rows by `pair_id`, and
computes per-pair `market_type` (from `is_each_way`), `final_outcome`,
and `pair_pnl` (sum of the pair's per-leg P&L).

**Single-agent caveat.** Only one agent (`557334ae`) has bet logs
written so far — the cohort is at 1/30 evaluated. All numbers below
are for that single random-init gen-0 policy. The pattern is unlikely
to flip qualitatively across the cohort (the structural causes — book
liquidity into the off, EW commission asymmetry — are agent-agnostic)
but the magnitudes will move.

### Cross-tab: market_type × final_outcome

**Count of pairs (405 total)**:

| market_type | agent_closed | force_closed | matured | naked | row total |
|---|---:|---:|---:|---:|---:|
| EW   | 98  | 82  | 0 | 18 | 198 |
| WIN  | 56  | 146 | 1 | 4  | 207 |

**Mean pair P&L (£)**:

| market_type | agent_closed | force_closed | naked  |
|---|---:|---:|---:|
| EW   | -0.362 | -1.358 | -0.253 |
| WIN  | -1.108 | -1.295 | +2.678 |

**Sum pair P&L (£)**:

| market_type | agent_closed | force_closed | naked   | row total |
|---|---:|---:|---:|---:|
| EW   | -35.46  | -111.36 | -4.55  | -151.38 |
| WIN  | -62.06  | -189.12 | +10.71 | -238.67 |

**Force-close RATE**: EW = 41.4 %, WIN = 70.5 %.

### Per-leg decomposition of force-closed pairs

For each force-closed pair we have one agg leg + one close leg
(force_close=True). Mean per-leg pnl:

| market_type | agg leg mean | close leg mean | spread cost ≈ |
|---|---:|---:|---:|
| WIN | +£4.49 | -£5.78 | ~£1.30 / pair |
| EW  | +£1.67 | -£3.02 | ~£1.36 / pair |

**Spread cost per force-closed pair is essentially identical** across
market types. The operator's anecdote (Warwick Each Way, Two To Tango,
2026-04-23 16:01) — net -£0.08 / pair — is below the empirical median;
EW force-closes are not consistently worse than WIN ones. Median pair
P&L on a force-closed pair is -£0.86 (EW) vs -£0.85 (WIN) — same.

What dominates the EW vs WIN cost difference is the **rate**, not the
per-pair magnitude. The agent force-closes ~1.8× as many WIN pairs as
EW pairs over the 7 days. Likely root cause (not directly testable
from this data): WIN markets have steeper pre-off price discovery,
the passive at +20 ticks is less likely to be caught up to, more
pairs survive to T-120s.

### Ranked options (do not pick — present trade-offs)

**Option A — disable force-close on EW markets entirely.**
Empirical reach: roughly **-£111 reclaimed** if all 82 force-closed
EW pairs were instead left to settle naked. Risk: 18 of the 22 naked
pairs in the data were EW, contributing -£4.55 (mean -£0.25 ±£11.2
std). Distribution-tail risk dominates — the worst naked EW pair was
-£23.26. Disabling force-close on EW means accepting that fat tail.
Across 7 days the mean is net-beneficial **but the variance jumps by
1 OOM**. For deployment safety this is the worst option (variance is
the deployment-critical metric per `memory/feedback_naked_variance_primary_metric.md`).

**Option B — earlier force-close threshold for EW (e.g. T-240s).**
Plausible — at T-240s the book is generally thicker than T-120s. No
empirical reach in this data (we'd need replay simulation with the
shifted threshold). The cost: the agent's passive close legs in the
[120s, 240s] window that *would* have filled naturally now get
front-run by the early force-close, paying the spread we'd otherwise
have avoided. Reach is a guess without re-running the day.

**Option C — different stake-sizing on the EW force-close (close
for equal-place-PNL).**
The operator's hypothesis was that the EW asymmetry concentrates loss
on the place-half. The data partially supports the framing: of 82 EW
force-closed pairs, 47 had a winning agg leg (lay won) and 35 had a
losing agg leg, with the close-leg flipping it the other way. But the
per-pair *mean* P&L (-£1.36) and *median* (-£0.86) are so close to
WIN that the asymmetry, if present, is second-order. Option C would
require re-deriving the equal-profit sizing formula for the EW
two-leg-payoff structure and porting it into `_attempt_close` —
non-trivial.

**Option D — filter EW open-side entirely.**
Empirical reach: removes -£151.38 of EW pair cost, but also removes
the upside (matured pairs, profitable agent_closed pairs). Net EW
total in the data is -£151 — that's the upper-bound saving. But the
*current* WIN total is -£238, so removing EW doesn't make the cohort
profitable; it just makes it less bad. The earlier probe the operator
referenced ("agents barely traded with EW hidden") suggests something
deeper: EW may be where the predictor's edge actually resides
(extreme outsiders with mispriced place probabilities), and the
agent's training signal goes flat without it.

**Honest "rank" given the evidence**: A and D both shrink the loss
but at structural cost (variance / edge); B and C need replay data
to evaluate. None of them addresses the root mechanism — passives
not filling because they're targeted 20 ticks away regardless of
microstructure.

---

## Q2: How arb_spread flows from policy → resting price

### v2 stack: the policy has no arb_spread action

In the v2 stack ([agents_v2/](../../agents_v2/) + [training_v2/discrete_ppo/](../../training_v2/discrete_ppo/))
the policy's discrete head emits one of these per step:

- `NOOP` (idx 0)
- `OPEN_BACK_i` for runner i ∈ [0, max_runners)
- `OPEN_LAY_i`
- `CLOSE_i`

Layout locked at [agents_v2/action_space.py:7-22](../../agents_v2/action_space.py#L7).
Total size = `1 + 3 * max_runners`. **Nowhere in this surface does
the policy emit a per-runner arb_spread value.**

The env's `_process_action` ([env/betfair_env.py:3287-3306](../../env/betfair_env.py#L3287))
still reads `action[4 * max_runners + slot]` as `arb_raw` and maps it
through `frac = (arb_raw + 1) / 2; ticks = MIN + frac * (MAX - MIN)`
with `MIN_ARB_TICKS=1, MAX_ARB_TICKS=25`. So the env-side decode
exists — but the v2 shim populates this slot with a **constant**.

### What populates the env's `arb_spread` slot in v2

[agents_v2/env_shim.py:105](../../agents_v2/env_shim.py#L105)::

```python
_DEFAULT_ARB_TICKS = 20
```

This is the constructor default; the cohort runner never overrides
it. The shim pre-computes the inverse of the env's mapping ONCE at
construction ([env_shim.py:209-213](../../agents_v2/env_shim.py#L209)):

```python
arb_frac = (self._arb_ticks - MIN_ARB_TICKS) / (MAX_ARB_TICKS - MIN_ARB_TICKS)
self._arb_raw = float(np.clip(2.0 * arb_frac - 1.0, -1.0, 1.0))
# arb_ticks=20, MIN=1, MAX=25 ⇒ raw ≈ 0.5833
```

`encode_action` then plants this same `_arb_raw` into the action
vector for EVERY OPEN_* action ([env_shim.py:313-319](../../agents_v2/env_shim.py#L313)):

```python
if self._scalping:
    arb_raw = (
        self._arb_raw
        if arb_spread is None
        else self._encode_arb_spread(int(arb_spread))
    )
    action[4 * self._N + slot] = arb_raw  # arb_spread
```

`arb_spread` is never passed from any caller — both
[`training_v2/discrete_ppo/rollout.py:606`](../../training_v2/discrete_ppo/rollout.py#L606)
and [`training_v2/discrete_ppo/batched_rollout.py:440`](../../training_v2/discrete_ppo/batched_rollout.py#L440)
call `shim.step(action, stake=..., arb_spread=None)`. So during
training and eval the arb_spread is **always 20 ticks**, regardless
of runner, regardless of obs, regardless of agent.

### What the GA evolves: `arb_spread_scale` (per-agent global)

The only per-agent variation is the gene `arb_spread_scale ∈ [0.5, 2.0]`
([training_v2/cohort/genes.py:26,118,158,214,300](../../training_v2/cohort/genes.py#L26))
which the env reads as a global multiplier in
[env/betfair_env.py:1577](../../env/betfair_env.py#L1577) and applies
in [env/betfair_env.py:3296-3303](../../env/betfair_env.py#L3296)::

```python
raw_ticks = (
    MIN_ARB_TICKS
    + arb_frac * (MAX_ARB_TICKS - MIN_ARB_TICKS)
) * self._arb_spread_scale
arb_ticks = int(round(max(MIN_ARB_TICKS, min(MAX_ARB_TICKS, raw_ticks))))
```

With `arb_frac = 0.79167` (the locked shim value) and `scale ∈
[0.5, 2.0]`: agents get arb_ticks ∈ [10, 25]. The current cohort's
sole gen-0 agent ([scoreboard.jsonl](../../registry/_predictor_SCALPING_postfix_e3_cohort_1779530050/scoreboard.jsonl)
shows `arb_spread_scale: 1.0`) gets exactly 20 ticks.

There IS also a commission-feasibility floor
([env/betfair_env.py:3404-3420](../../env/betfair_env.py#L3404)) that
can BUMP `arb_ticks` UP if 20 ticks at the agg price wouldn't clear
commission — for typical prices (3.0–10.0) the floor is well below
20, so 20 sticks.

### Where the passive_price is computed

`_maybe_place_paired` ([env/betfair_env.py:3712](../../env/betfair_env.py#L3712))
takes `arb_ticks` and computes `passive_price = tick_offset(agg_price, arb_ticks, ±1)`
([env/betfair_env.py:3791-3797](../../env/betfair_env.py#L3791)).
Equal-profit sizing (per [CLAUDE.md "Equal-profit pair sizing"](../../CLAUDE.md))
then runs to compute the passive stake.

### Bottom line — Q2

For the running cohort, **every aggressive open posts its passive
counter-leg exactly 20 Betfair-ladder ticks away from the agg fill**,
modulated only by the per-agent global `arb_spread_scale` (which
isn't even varying in gen 0 — the lone evaluated agent has it at
1.0). The 76 % force-close aggregate rate is a direct consequence:
20 ticks is far at prices like 4.40 (Two To Tango's example: 1 tick
≈ 0.05 → 20 ticks ≈ price 5.40 lay target, while the LTP trended
down to ~4.70). The passive can't fill, T-120s arrives, env
force-closes.

The `target_pnl_pair_sizing_enabled` path is not active in this
cohort (no `--target-pnl-pair-sizing-enabled` flag on the runner;
default False at [env/betfair_env.py:1499](../../env/betfair_env.py#L1499)).

---

## Q3: Per-horse arb_spread + obs schema

### Reframed: the policy structurally CAN'T pick per-horse

Q3 as posed assumes the agent has a per-runner arb_spread action and
asks whether the obs is rich enough to differentiate. Q2 establishes
that the agent doesn't have that action at all — so the obs question
is moot for arb_spread, though it remains relevant for whether the
agent can differentiate WHICH runner to open / when to close.

### LEAN obs already carries per-runner microstructure

[env/betfair_env.py:524-544](../../env/betfair_env.py#L524) defines
`LEAN_RUNNER_KEYS` (23 keys per runner) under
`predictor_lean_obs=True`. Per-runner microstructure entries:

- `back_price_1` — best back price (also gives implied p_win = 1/this)
- `lay_price_1` — best lay price
- `spread_pct` — visible top-of-book spread (PRESENT — answers the
  operator's "current visible spread in ticks" question)
- `ltp_velocity_30` — ~8 min EMA of price movement (PRESENT —
  proxies near-term volatility)
- `ltp_velocity_60` — ~16 min EMA (PRESENT — longer-horizon
  volatility)

Plus the per-runner predictor outputs (champion_p_win, ranker_top1,
direction_q10/50/90 at 1m/3m/7m horizons). That's a lot of per-runner
signal already.

**Missing from LEAN obs that would help if arb_spread became an
action**:
- per-runner **traded volume in last N seconds** (TVL captured at
  [memory `project_traded_volume_ladder_unused`](../../C:/Users/jsmit/.claude/projects/C--Users-jsmit-source-repos-rl-betfair/memory/project_traded_volume_ladder_unused.md)
  in parquets from 2026-04-26, no v2 feature reads it) — direct
  proxy for "is anyone trading this runner near LTP, will my +20-tick
  passive realistically catch a fill?"
- per-runner **back/lay ladder depth at top 3 levels** (Betfair feed
  caps at 3 — see [memory `project_book_depth_n3_widen_later`](../../C:/Users/jsmit/.claude/projects/C--Users-jsmit-source-repos-rl-betfair/memory/project_book_depth_n3_widen_later.md))
  — "how thick is the wall I'd need to walk through".
- per-runner **time since last trade** — long quiescent periods near
  the off mean the passive is unlikely to fill.

But adding obs features without an action to condition them on is
pointless. **The architecturally meaningful question is whether to
extend the v2 action space**, not whether to add obs features.

### Did the agent actually vary observed spread by pwin bucket?

Caveat: in agent-closed and force-closed pairs the close leg's price
is the *market* at close time, not the agent's original 20-tick
target — so this metric measures execution distance, not chosen
arb_spread. With only 1 matured pair in the data, true chosen-spread
distribution is uncomputable. Reported below for completeness:

Median |close_price - agg_price| (in ticks) by pwin bucket:

| pwin_bucket | n | median | mean | std |
|---|---:|---:|---:|---:|
| <0.05 | 37 | 4.0 | 4.03 | 3.42 |
| 0.05-0.10 | 44 | 2.0 | 3.77 | 4.51 |
| 0.10-0.20 | 95 | 3.0 | 4.27 | 4.08 |
| 0.20-0.40 | 80 | 3.0 | 3.70 | 3.64 |
| 0.40-0.70 | 98 | 3.0 | 4.54 | 3.91 |
| >0.70 | 28 | 3.5 | 4.25 | 3.41 |

There's **no monotonic dependence on pwin** — consistent with "this
is dominated by execution price at close time, not by the agent's
choice". The agent isn't differentiating because (a) it has no
mechanism to and (b) the visible signal is execution noise, not its
own choices.

---

## Three questions, one root cause

The three questions collapse to one structural issue: **arb_spread is
not a per-runner action and the cohort is using the default 20-tick
target on every open**. The EW force-close cost framing in Q1 turns
out to be smaller than the operator suspected (Q1 mean per-pair cost
is symmetric across WIN/EW); Q2 reveals that the agent has no
arb_spread channel at all; Q3's "can the agent learn per-horse
differentiation" answers no for the structural reason from Q2.

This is gen-0 random-init data — the cohort will train and the
GA-evolved `arb_spread_scale` may pull toward 0.5 or 2.0 across
generations as the population finds whatever scale produces the best
composite. But no individual agent will EVER pick a different
arb_spread for a 1.20 favourite vs a 12.0 long-shot — the
architecture forbids it.

---

## Recommended next step (one change, low risk)

**Pin `arb_spread_scale=0.5` cohort-wide and re-launch a probe.**

Why this one:

- Halves the target spread to 10 ticks at the locked shim default.
  10 ticks at price 4.40 ≈ 0.5 price units → a much more reachable
  passive close target than 20 ticks ≈ 1.0 price unit.
- Direct mechanical effect on the 76 % aggregate force-close rate —
  the passive is far more likely to fill within the [open, T-120s]
  window because it's closer to where the market actually trades.
- Single flag change: `--arb-spread-scale 0.5` on
  `training_v2.cohort.runner`. The plumbing already exists
  ([training_v2/cohort/runner.py:1759-1767](../../training_v2/cohort/runner.py#L1759)).
- The cohort-wide pin overrides the per-agent gene
  ([runner.py:1962-1965](../../training_v2/cohort/runner.py#L1962))
  so we're confident we're testing the mechanism, not GA drift.
- Reversible — if it tanks the locked_pnl share (because narrow
  spreads also lock less profit per pair under the equal-profit
  sizing), revert.
- Compares cleanly against the current cohort's scoreboard
  (same code, same days, same predictor — only the spread changes).

What this won't fix: gen-0 random-init agents will still fire
aggressively at noise. The probe is informative ONLY if we compare
the **same metric** (force-close rate, per-pair P&L, locked / naked
split) at matched generation count.

**If the probe still shows high force-close rates with scale=0.5**,
the follow-up is architectural: promote `arb_spread` to a per-runner
discrete action dim (extend `DiscreteActionSpace` from
`1 + 3 * max_runners` to e.g. `1 + 4 * max_runners` with a "spread
band" dimension at OPEN time, or fold it into the OPEN_*_NARROW vs
OPEN_*_WIDE pair). That's a multi-session piece of work and should
NOT be the first attempt — pin the scale first, see how much of the
problem is mechanical vs how much is policy-structural.

---

## Sample size / honesty notes

- **One agent only** has bet logs (cohort 1/30). All Q1 numbers will
  shift as the other 29 agents finish. The ordinal claim "WIN
  force-close rate > EW force-close rate" is likely robust (it's a
  structural book-liquidity claim, not an agent-specific one) but
  the magnitudes will move.
- The 405-pair sample IS large enough to support the per-pair mean
  comparisons (std/√n on the force-close mean is ≈ £0.30 — the
  WIN-vs-EW means differ by £0.06, well within noise).
- Anything quoted with "across all agents in gen 0" in the session
  prompt cannot be computed yet. Once gen 0 finishes (~12.5 h),
  rerun `C:/tmp/analyze_bet_logs.py` for the cohort-wide picture.
