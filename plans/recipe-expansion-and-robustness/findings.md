# Recipe Expansion and Robustness — findings

Campaign window: 2026-05-25 → 2026-05-30 (ongoing). Multi-day
autonomous experiment push targeting a deployable true-scalping
recipe (agent opens pairs that mature). Probe scale: 4 agents × 1
gen × 3 train days × held-out eval.

> Companion docs:
> - `purpose.md` — original plan
> - `autonomous_loop_v2.md` — operating instructions for the loop
> - `BREAKTHROUGH.md` — **SUPERSEDED**, kept as a warning (the +£287
>   it celebrated was eval-window overfitting)
> - `monitoring_notes.md` — chronological log of decisions
> - `plans/EXPERIMENTS.md` — campaign-wide chronological digest
> - `plans/EXPLORATIONS.md` — strategic-analysis essays (incl. the
>   economics framework + the held-out correction)

## TL;DR

**No deployable recipe yet.** Every recipe tested is held-out
negative. Best so far: **N4 (full-aug + pwin BAND 0.20–0.50) at
-£78/day** on 7 unseen days (vs baseline -£232). True scalping
mechanic IS +EV per matured pair (+£3–6 locked after commission),
but mat% is stuck at 4–7% — too low for locked total to cover
force-close cost of misses.

**The fundamental problem (now written down explicitly):**

```
per-open net = mat% × locked/pair − fc% × fc_cost/pair − close_cost
            ≈ 0.05 × £2.5    − 0.80 × £1.5         − ~£0.45
            ≈ +£0.13         − £1.20               − £0.45
            ≈ −£1.07/open ALWAYS NEGATIVE at current op-point
```

Selectivity (opening fewer pairs) reduces the *count* of negative
opens → "less bad" day_pnl. It does **not** flip the per-open sign.
To reach net-positive, either mat% must rise ~6× (5% → ~30%) or
fc-cost/pair must drop dramatically.

## Held-out leaderboard (7 unseen days 2026-05-07..05-19, fc=120)

Ranked by **held-out day_pnl** (the deployable number).

| rank | recipe | day_pnl | opens | mat% | lkd/mat | round |
|---|---|---:|---:|---:|---:|---|
| 1 | **full-aug + pwin BAND 0.20–0.50** | **-£78** | 52 | 1.9% | +£4.80 | N4 |
| 2 | full-aug + pwin 0.35 | -£98 | 82 | 4.2% | +£2.43 | N2 |
| 2 | full-aug + pwin 0.25 (seed 42) | -£98 | 90 | 5.0% | +£2.57 | M6 / P_seed42 |
| 4 | full-aug + pwin 0.25 + tight0030 + pwin025 | -£128 | 81 | 6.5% | +£1.50 | M7 |
| 5 | full-aug + pwin 0.25 (seed 44) | -£110 | 100 | 3.3% | +£2.65 | P_seed44 |
| 6 | full-aug + fc=60 | -£114 | 95 | 5.3% | +£2.56 | O1 |
| 7 | full-aug + pwin 0.25 mean across 6 seeds | -£130 | ~100 | ~4% | ~£2.4 | Round P |
| ref | E7 (pwin_back + BC=500) | -£227 | 162 | 3.7% | +£2.59 | H1 HB2 |
| ref | baseline (no gates) | -£232 | 157 | 1.6% | +£6.63 | H1 HB0 |

**Reference for what NOT to do:** the in-sample +£287 from
fc=0+lay_max=100 → **-£175 on held-out** (HV cells). Naked = zero-EV
directional variance.

## What works

1. **fc=120 safety rail** stays ON. Disabling force-close *exposes*
   the agent to naked directional variance, which is zero-EV.
2. **pwin_back floor at 0.20** (env-side action mask) filters out
   structurally-bad back picks — confirmed positive lever.
3. **Close+hold augmentation** (L2 NOOP-at-oracle-neg + L3a/L4
   close+hold labels) restores close behaviour post-BC and reduces
   opens (selectivity). Best non-BC variant has opens ~100 vs E7's
   162.
4. **pwin BAND (0.20–0.50)** caps super-favourites. The best lever
   found — drops opens to 52 with lkd/mat +£4.80 (highest seen).
5. **Shorter fc window (fc=60)** modestly cuts naked exposure
   near the off; minor but real held-out benefit (-£114 vs
   -£130 ish).

## What doesn't work

1. **fc=0 (disable force-close)** — looked best in-sample (+£287),
   loses heavily on held-out (-£175). The naked P&L it harvests is
   zero-EV directional luck on the in-sample window.
2. **Shaped reward levers at probe scale** — close_signal_bonus,
   matured_arb_bonus, open_cost: all moved cls%/mat% by <2pp despite
   5× swings. PPO at 4 agents × 1 gen doesn't extract enough gradient
   from shaped rewards to change behaviour.
3. **Direction-gate (any threshold)** — flat-and-harmful 0.20-0.45.
   Direction predictor's signal in OBS is useful; using it as an
   action mask is not.
4. **BC alone (no augmentation)** — un-trains close_signal,
   produces 89% fc rate, day_pnl -£300. Fixed by L2/L3a/L4
   augmentation but still net-negative.
5. **Stacking "best" levers blindly** — combining lay_max=100 +
   pwin band 0.20-0.50 (Round 9.8) gave -£163 vs each alone
   (+£287 / +£262 in-sample). Levers don't always compose
   positively.

## KEY FINDING #2 (2026-05-30) — the no-walk rule on the CLOSE path is the biggest single leak

Follow-up to KEY FINDING #1 below, prompted by the operator asking
why a back@2.92 / lay@2.94 pair lost £27. Root cause is NOT open
selection — it's a matcher artifact on the close/force-close path.

**Mechanism.** `exchange_matcher._match` fills a bet at ONE price
level only (`matched = min(stake, one_level_size)`; remainder
cancelled, never walked). `_attempt_close` sizes the close leg at
full equal-profit but routes it through this single-level matcher,
so when the best opposite-side level is smaller than the hedge
needs, the close PARTIALLY fills and leaves the aggressive leg
under-hedged → directional exposure. This applies to BOTH
`close_signal` (agent) and force-close (env) — both cross via the
same matcher.

**The book was NOT thin — no-walk discarded accessible liquidity.**
Worked example (pair 785f84170f, Runman, 212s to off, LTP 2.94):
needed ~£42.60 lay to hedge a £43.59 back. AvailableToLay book:

    2.94 × £17.55   ← only level no-walk took
    2.96 × £146.21  ← left on the table (ONE tick away!)
    2.98 × £66.15   ← left on the table

£146 sat one tick away. No-walk took £17.55, threw away the £25
hedge, ate −£27 directional loss to avoid laying £25 one tick worse.

**Quantified prize (1 agent, 7 held-out days, 191 close/fc pairs):**
- Realised P&L on these pairs: **−£338.86** (mean −£1.77, 174/191
  negative, range −£30..+£50 = directional-variance signature).
- **98% of loss-making pairs (181/184) had ≥10-tick book depth to
  ~complete the hedge** — £814 of £880 worst-case exposure was
  mechanically reachable by a bounded walk.

So ~£339/agent/7d of avoidable directional loss — dwarfs the
open-selection problem and could alone move this agent from −£68/7d
toward breakeven.

**Spec is on our side.** `docs/betfair_market_model.md` §2: real
Betfair fills "at the named price OR BETTER, leaving any unfillable
remainder resting … you cannot submit a market-order that sweeps the
book. You always name a price." A bounded limit order DOES consume
intervening better levels up to its limit. The sim's single-level
rule is MORE conservative than reality; a 10-tick-bounded close walk
is a named limit order, not the unbounded sweep the no-walk ban
targets (that ban is about OPENING through £1000 junk).

**Fix (operator-directed 2026-05-30):** allow the close/force-close
path to walk up to N=10 ticks from the best price, filling across
levels until the equal-profit hedge completes or the limit/cap is
hit. OPEN path keeps strict single-level no-walk (unchanged). Hard
cap (`max_back_price`/`max_lay_price`) still enforced on every walked
level; junk filter unchanged. Back-exposure closes (lay walk) are the
priority; lay-exposure (back walk) applied symmetrically for now,
flagged for separate experiment. Implemented as a knob defaulting OFF
(byte-identical) + a held-out A/B round to prove the lift.

## KEY FINDING #1 (2026-05-30) — per-pair forensics: the edge is REAL, three leaks bury it

Per-pair bet-log analysis of a representative Q1 agent (05ccc6af,
7 held-out days, 217 filled-leg pairs) splits the population cleanly
and rewrites the campaign's framing. Scoreboard counters are PER-DAY;
bet logs are 7-day totals (217 ÷ 7 ≈ 31 opens/day = `eval_pairs_opened`).

| category | count (7d) | % | median lock %/agg-stake | sign |
|---|--:|--:|--:|---|
| force-closed | **175** | **81%** | — | the safety-rail toll (−£18..−84) |
| naturally matured | 19 | 9% | **+0.68%** (range +0.35..+3.50) | **all 19 POSITIVE** |
| agent-closed (`close_signal`) | 16 | 7% | **−15.8%** (range −0.94..−61.75) | **all 16 NEGATIVE** |
| naked (1 leg only) | 7 | 3% | — | — |

**Three independent conclusions, each load-bearing:**

1. **The 2%-scalping edge is real and works.** Every naturally-matured
   pair locked POSITIVE, win ≈ lose (balanced equal-profit), clustering
   +0.5–1% with several at +2–3.5%. This is the strategy operating
   exactly as the `arb_spread_target_lock_pct=0.02` design intends.
   Absolute £/pair is small ONLY because stakes are small — scaling
   stake scales the £ linearly without touching the edge. Scalping is
   a 2% game; the matured pairs ARE playing it correctly.

2. **Force-close is the throughput killer, not a bug — 81% of opens
   never fill their passive.** The agent opens ~31 pairs/day but only
   ~3 mature naturally; the other ~25 sit unfilled until the T−120
   safety rail crosses the spread at market (the −£18..−84 fc-cost).
   The rail is doing its job (bounding what would otherwise be naked
   ±£100s variance). The fault is UPSTREAM: the agent opens far too
   many pairs whose passive won't fill. **Open only fillable pairs →
   the 81% toll collapses → the 9% positive-locked population becomes
   the whole book.** This is the entire thesis of Path C (mature gate)
   and Path D (liquidity gate), now backed by direct evidence the
   upside exists.

3. **`close_signal` is a footgun — every agent-close lost money via
   UNDER-HEDGING.** All 16 agent-closed pairs were negative (median
   −15.8%, worst −61.75%). Mechanism: agent-closes use the STRICT
   matcher (single price, no walk, LTP + junk filter) and do NOT
   overdraft — so the equal-profit-sized close leg only PARTIALLY
   fills at one level (e.g. pair 785f84170f: back £43.6 vs close-lay
   only £17.6 matched), leaving the pair directionally exposed. Ironic
   contrast: the env's force-close uses the RELAXED matcher + overdraft
   and fully flattens, so the SAFETY rail closes cleaner than the
   agent's VOLUNTARY closes. Options: (a) give agent-closes the same
   relaxed+overdraft treatment so they fully hedge, or (b) disable
   `close_signal` entirely and rely on natural maturation + the
   force-close rail. Given 16/16 were value-destructive, (b) is the
   cheap immediate win; (a) preserves the tool if we want it.

**Campaign redirect:** the binding constraint is FILL RELIABILITY of
the passive leg (drives both the 81% force-close toll and the doomed
opens the agent later closes at a loss). Every lever should target
"open only pairs whose passive will fill": Path C (mature_prob gate,
Round T — already queued), Path D (liquidity/book-depth gate), and a
`close_signal` fix/disable as an independent quick win. Spread/band
tuning (Rounds Q/R/S) operates downstream of this and cannot fix it.

## Per-open economics — the central framework

For any recipe to be deployable, the per-open arithmetic must be
positive:

```
per_open = (mat% × locked/pair) − (fc% × fc_cost/pair) − (cls% × close_cost/pair)
```

Today's recipes sit at **-£0.5 to -£1.1 per open**. Multiplied by
50-160 opens/day = the -£78 to -£232 day_pnl we see.

To flip per-open positive, three orthogonal paths:

### Path A — selectivity (Round R direction)

Open fewer, better pairs. Doesn't change per-open sign, but reduces
the count of negative-EV opens. **Floor for this path** = number of
opens × per-open arithmetic; the absolute minimum is opens = 0,
day_pnl = £0. **This is N4's path.** Round R pushes it further
(band widths, N4 seed replicates, band+fc=60 stack).

### Path B — mat-lift (Round S direction)

Trade locked/pair for higher mat%. Make passives easy to fill (tight
spread), cut fc-cost/pair (short fc window). Targets the per-open
arithmetic by flipping the sign at the source. **Round S explicitly
tests this** — extreme tight_lock (0.0005-0.0001) + fc=30/15 +
stacked variants.

### Path C — architectural: mature_prob open-gate (BUILT — Round T)

Use the policy's own `mature_prob_head` to gate opens at the source:
only open pairs the model predicts will mature. Targets per-open
sign by filtering on PREDICTED maturation rather than indirect
proxies. **The highest-leverage untried mechanism.** Built
2026-05-30: policy-side mask layer (`--mature-prob-open-threshold`)
that zeros OPEN logits per-runner where `sigmoid(mature_prob_head)
< threshold`; mirrors the direction-gate's KL-consistency +
warmup-anneal contract. Tests: `tests/test_v2_mature_gate.py`.
Round T (chained S→T) sweeps 0.20/0.30/0.40/0.50 + a head-trained-
no-gate control (T5) on N4 base.

**Critical setup.** `mature_prob_head` is an IN-POLICY aux head,
not an external predictor. Every prior round ran
`mature_prob_loss_weight = 0`, so the head was UNTRAINED
(≈constant 0.5) — gating on it would be degenerate. Round T pins
`mature_prob_loss_weight = 2.0` cohort-wide so the head learns the
strict-maturation label inline during the 1-gen run.

**Contingency — promote the head to sweep+freeze+share if Round T
shows it's undertrained.** The direction head already gets the
"win-predictor" treatment: an architecture sweep
(`models/direction_head/sweep_c1..c18`), winner `sweep_c11` loaded
FROZEN and shared across all agents via `--direction-head-manifest`
(per-agent supervised loss forced to 0). `mature_prob_head` has NO
equivalent. IF Round T's diagnostics say the inline head didn't
learn — i.e. the gate masks ~everything or ~nothing, mat% stays
flat across the threshold sweep, T5 ≈ T-gated, or
`mature_prob_bce_mean` stays high through the run — THEN the limiter
is the 1-gen/3-day training budget, and the fix is to give the
mature head its own supervised sweep on oracle maturation labels,
pick the best architecture, freeze it, and load it into every agent
(mirror `plans/shared-direction-head/`, add `--mature-head-manifest`).
DO NOT pre-build this sweep — only if Round T proves inline training
is the bottleneck.

**Caveat (why the analogy to win/direction predictors is imperfect):**
the win and direction predictors forecast EXOGENOUS market facts
(race outcome, price direction) independent of the agent, so
freeze-and-share is clean. The mature label is partly ENDOGENOUS —
agent-closed pairs count as matured, force-close/naked as not — so a
head frozen from one policy's close behaviour may not transfer
perfectly to another's. Still usable as a prior, but expect some
distribution shift a frozen direction head doesn't suffer.

### Path D — liquidity gate (future)

Only open on runners with deep enough book that passive lays fill
reliably. Needs new env field (per-runner book depth) exposed to
the action mask. Not yet built.

## Methodology lessons (locked in, do not relitigate)

1. **Always eval on held-out, never train on held-out.** Maintain
   train (Apr 6/8/9) / iteration-eval (May odd days 7-19) / final-
   test (May even days 8-20) split.
2. **Select on LOCKED P&L + mat%, not total day_pnl.** Total is
   naked-noise on short windows.
3. **fc=120 safety rail ON** for any deployable recipe.
4. **Document each cell's mat%, locked/pair, fc-cost/pair** — not
   just day_pnl. Per-open economics is the right unit.
5. **Bash polling chains are the durable autonomy mechanism** on
   Windows + git-bash + Claude Code. "Smarter" daemons (ScheduleWakeup,
   supervisor) have all failed at this environment.
6. **Don't celebrate "less negative" as "almost there."** Quote the
   per-open arithmetic; recipe is deployable iff per_open ≥ 0.

## Open questions for follow-on work

- Does Round S (mat-lift path) materially shift the per-open sign?
- Does the mature_prob open-gate (Path C) work at probe scale?
- Is `mature_prob_head` trainable to useful discrimination INLINE
  (1-gen, weight=2.0, Round T), or does it need the direction-head
  treatment (dedicated supervised sweep → freeze → share)? Decide
  from Round T's BCE trace + threshold-sweep response.
- If neither: does a structural change to selection — e.g. a
  liquidity gate, a different open-decision signal — produce a
  recipe with per-open ≥ 0?
- If still no: is the fundamental claim (true scalping is profitable
  on these markets after commission) actually viable, or does
  Betfair's 5% commission + the available spread structure rule it
  out on the open-side selection alone?

## Recommended next steps (in order)

1. **Wait for Round S** (held-out, mat-lift path) — ~17:30 BST
   2026-05-30. Read the per-open numbers to see if either lever
   moved the sign.
2. **Build the mature_prob open-gate** (Path C) — the principled
   selector. ~1h of policy-side env code. Round T = held-out probe
   of it.
3. **If neither Path B nor Path C produces a held-out per-open ≥ 0
   recipe:** broader rethink. Possibilities include a structurally
   different open-decision signal, a liquidity-aware selector, or
   the honest finding that probe-scale single-gen training can't
   land a deployable recipe and we need a much larger cohort/multi-
   gen run on the architecturally-richer base.
4. **For any candidate that achieves held-out per-open ≥ 0:**
   confirm on the final-test set (May even days 8-20, never touched)
   before considering deployment.

## Book-depth investigation (2026-05-30) — 10 levels ARE in the feed, but not for our data

Operator asked whether we can get 10 ladder levels into the parquets
(impacts close-walk reach AND the direction predictor). Full cross-repo
trace (rl-betfair extractor + StreamRecorder1 + local MySQL hotDataRefactored):

1. **The raw Betfair stream carries 10 levels.** StreamRecorder1
   subscribes to `EX_ALL_OFFERS`; raw deltas (`rawmarketchanges.RawJson`,
   `e.Change`) contain `batb`/`batl` arrays with levels 0-9 = TEN levels
   `[level, price, size]`. Confirmed on real 2026-05-20 data.
2. **The resolved snapshot truncates to 3.** `ResolvedMarketSnaps.SnapJson`
   = `SerializeObject(e.Snap)` keeps only top-3 of batb/batl. Our extractor
   reads ResolvedMarketSnaps → 3-level parquets. `feature_engineer.py:542-543`
   then slices `[:3]` anyway, and the direction predictor + all RL book
   features see only those 3.
3. **Our campaign data CANNOT be deepened.** `rawmarketchanges` contains
   ONLY 2026-05-20 (2347 rows) — the raw pipeline (migration 2026-03-25)
   only ever ran one day. April training + May 7-19 eval predate any raw
   capture and survive only as already-extracted 3-level parquets; the hot
   DB no longer retains them. No cheap historical re-extraction.
4. **Forward fix is simple** (batb/batl is already a top-10 SNAPSHOT, not a
   delta — no replay needed): have the recorder persist 10 levels in the
   resolved snap (not 3), confirm the recorder is actually running (hot DB
   stops at 2026-05-20, ~10 days stale — likely OFF), accumulate new
   10-level data, widen `feature_engineer` [:3]→[:10] (backward-compatible)
   + add depth-imbalance features, then retrain predictors on new data.

**Bottom line:** deeper book is a FORWARD-DATA project (weeks: enable
capture → accumulate → re-engineer → retrain), independent of and not
blocking the close-walk work on current 3-level data. The direction-
predictor retrain the operator floated can't happen until enough
10-level data exists. Close-walk is unaffected (3 levels suffice to hedge).
