# Session prompt: investigate EW force-close, arb_spread action design

Self-contained brief for a fresh agent session. The user has three
design questions surfaced by reviewing live cohort bet logs. Investigate,
don't implement.

---

## Context

You're picking up a design investigation on **rl-betfair**, a v2 cohort
GA-PPO trainer for Betfair scalping policies. The agents post pair-trade
scalps: an aggressive lay/back leg matches at the visible top-of-book, and
a passive counter-leg is posted at an offset to lock a spread. When the
passive never fills, the env force-closes the pair at `T−force_close_before_off_seconds`
(currently 120s) using a relaxed matcher to flatten on whatever book is
visible.

## What triggered the investigation

A live bet on the post-matcher-fix cohort:

```
[2026-04-23 16:01:00] 557334ae 16:02 Warwick Each Way  lay Two To Tango @ 4.40 £10.37 ->  won  +1.40  [force_closed pwin=0.04]
[2026-04-23 16:01:17] 557334ae 16:02 Warwick Each Way back Two To Tango @ 4.70 £ 9.99 -> lost  -1.48  [force_closed close,fc pwin=0.04]
```

Both `force_closed`. The agent's aggressive lay at 4.40 fired at 60s pre-off,
the passive back target was somewhere far above 4.70, the env force-closed
at 4.70 17 seconds later, EW asymmetry consumed the small directional
profit (net pair P&L: −£0.08). For the full walkthrough see the operator
conversation log — but the punchline is:

- The agent's choice of arb_spread target was unrealistic for the time
  remaining + the visible spread.
- The force-close in EW pays spread cost on BOTH the win-half AND the
  place-half of the bet, where a win-market force-close only pays one.

Agent 557334ae's gen 0 pair-outcome breakdown over 7 eval days:
**1 naturally matured, 154 agent-closed, 228 force-closed, 22 naked.**
405 pairs resolved. The matcher fix means the agent has to actually
position close legs where the market trades, and gen 0 random-init agents
are nowhere near that.

## The three questions

### Q1. Do we need a different EW force-close mechanism?

EW markets seem to pay disproportionately when force-closed:

- The lay leg's win-half wins (horse didn't win) but the place-half loses
  (horse placed) — a partial win.
- The aggressive force-close back at a slightly drifted price has its win-
  half lose (horse didn't win) but the place-half partially recovers — a
  partial loss bigger than the lay's win.
- Net: forcibly crossing the spread on an EW market eats more spread than
  on a win market.

Possible design changes (don't pick one yet — investigate):
- **Option A:** disable force-close on EW markets entirely (let them ride
  naked to settle). Cheaper close on average but catastrophic when a
  long-shot wins.
- **Option B:** earlier force-close threshold for EW (e.g. `force_close_before_off_seconds=240` for EW only) — get a better book before the
  thin late-stage period.
- **Option C:** different stake-sizing on the EW force-close (e.g. close
  for equal-place-PNL instead of equal-win-PNL).
- **Option D:** stop the agent OPENING pairs on EW markets in the first
  place (a `--market-type-filter WIN` style env flag — we have the
  plumbing; an old probe with this filter showed agents barely traded at
  all when EW was hidden, which suggests EW is where the agent's "edge"
  signal mostly lives even though force-close eats it).

What you need to find out empirically (from the live cohort's bet logs):
- Per-market-type, what fraction of resolved pairs are force-closed?
- Per-market-type, what's the mean per-pair P&L of force-closed pairs vs
  agent-closed vs naturally-matured?
- Per-market-type, what's the cumulative force-close cost across all agents
  in gen 0?

### Q2. Is `arb_spread` something each agent picks itself?

It's an action dimension (per-tick, per-runner) — but does the agent's
policy actually drive it meaningfully? Or is it mostly fixed by gene
initialisation / mapping?

Specifically:
- Where in `env/betfair_env.py` is the arb_spread action consumed?
  (Search for `arb_spread` — there's both an action-dim path and a
  reward-side knob.)
- What range does arb_spread map to in ticks? (Look for `MAX_ARB_TICKS`,
  `tick_offset`, `target_pnl_pair_sizing_enabled`.)
- The `--target-pnl-pair-sizing-enabled` mode reinterprets arb_spread as a
  £-target ∈ [£0.20, £5.00]. Which mode is the current cohort using?
- Is there a gene `arb_spread_scale` that multiplies the mapped range?
  How does the GA evolve it?

If arb_spread is fixed by a gene, then the GA evolves a per-agent
"willingness to take a wide arb" globally, not per-runner. If it's purely
an action, the agent's POLICY decides per-runner.

### Q3. Does it make sense for agents to pick arb_spread per-horse?

Different runners have different microstructure:
- Favourites: tight book, narrow spread, lots of liquidity. Small arb_spread
  targets fill often.
- Long-shots (pwin=0.04 like Two To Tango): wide book, sparse liquidity,
  price moves more between trades. Different arb_spread strategy probably
  needed.

If arb_spread is already per-runner (as I suspect), then the question is
whether the agent's OBSERVATION has enough per-runner microstructure for
it to learn the differentiation. Specifically:
- Does the obs vector include per-runner spread / depth / volatility
  features?
- Or just the lean predictor outputs + LTP velocity?
- Could we add features like "current visible spread in ticks" or "recent
  per-runner trade volume" to help the agent condition arb_spread?

Note: the cohort runs with `predictor_lean_obs=True`, which restricts
the per-runner obs to 23 keys (see `env/betfair_env.py::LEAN_RUNNER_KEYS`).
Check whether spread / volatility are in there.

## Your task

Produce a write-up at
`plans/force_close_and_arb_spread/findings.md` with:

1. **Q1 Answer:** Empirical per-market-type force-close cost from the live
   cohort's bet logs (paths below). Then a ranked list of options (A-D)
   with measured-impact projections. DO NOT pick a winner — present
   evidence + trade-offs and let the operator decide.
2. **Q2 Answer:** Definitive trace of how arb_spread flows from the
   policy's action head through to the resting price. Cite line numbers
   in `env/betfair_env.py`, `agents_v2/action_space.py`, and the
   `_open_paired` / equal-profit-sizing path.
3. **Q3 Answer:** Inspection of the LEAN obs schema vs what the agent
   needs to differentiate runners. Specific recommendation: if obs is
   adequate, say so + explain how the agent SHOULD learn this with more
   GA generations. If obs is missing something critical, propose 2-4
   per-runner features to add (with where they'd live in the obs schema).

## Constraints

- **DO NOT modify code.** This is investigation only.
- **DO NOT stop the running cohort.** Verify it's alive at start
  (`Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*training_v2.cohort.runner*' }`).
- The running cohort is at
  `registry/_predictor_SCALPING_postfix_e3_cohort_1779530050` (verify name
  on arrival — there's a slight chance it's been renamed).
- Bet logs for gen-0 agents live at
  `registry/<cohort>/bet_logs/<agent_uuid>/<date>.parquet`. Use these for
  empirical analysis.
- An example complete-agent text dump lives at
  `C:\tmp\cohort_agent_557334ae_bets.txt` (820 lines). Read for the human
  format if useful.
- The `_fill_tvl_features` regression test is the load-bearing correctness
  guard for predictor input shape: don't suggest anything that would break
  it.
- The matcher fix (`PassiveOrder.crossed` gate in
  `env/bet_manager.py::PassiveOrderBook.on_tick`) is correctness-critical
  — don't suggest reverting it.

## Key code paths to read

- `env/betfair_env.py::_open_paired` (~line 3680) — passive target price
  computation. Look at the `tick_offset` path and the
  `target_pnl_pair_sizing_enabled` path.
- `env/betfair_env.py::_force_close_open_pairs` — force-close logic +
  matcher relaxation.
- `env/bet_manager.py::place_back` / `place_lay` — explicit-price passive
  placement path (used by `_open_paired` to post the close leg).
- `env/exchange_matcher.py::_match` — the matcher with the `force_close`
  flag (relaxed semantics).
- `env/betfair_env.py::LEAN_RUNNER_KEYS` (~line 524) — the 23-key lean obs
  schema.
- `agents_v2/action_space.py` + `agents_v2/env_shim.py::encode_action` —
  how the discrete action index + stake + arb_spread are decoded.
- `data/feature_engineer.py::runner_tick_features` and
  `market_tick_features` — what per-runner / per-market features exist
  today (whether or not they're in lean_obs).
- `CLAUDE.md` — "Equal-profit pair sizing", "Force-close at T−N",
  "Order matching: single-price, no walking". Critical context.

## Empirical analysis bits to actually run

1. Iterate ALL agents' bet logs under
   `registry/<cohort>/bet_logs/`. For each pair (group by `pair_id`),
   compute:
   - market_type (WIN / EACH_WAY — see `is_each_way` column)
   - final_outcome (matured / agent_closed / force_closed / naked)
   - pair P&L
2. Cross-tabulate market_type × final_outcome → mean pair P&L, count of
   pairs.
3. For force-closed pairs specifically, decompose: how much was the
   spread cost (back_price - lay_price for back-first pairs; reverse for
   lay-first) × stake?
4. For Q3: pick 10 runners across the cohort with diverse pwin values
   (favourites at pwin > 0.5 and long-shots at pwin < 0.1). For each,
   compute the agent's chosen passive-close target distance from LTP in
   ticks. Do agents pick different distances for different pwin? (If yes,
   they're already conditioning on runner identity. If no, the policy
   isn't using per-runner info.)

## Style + honesty notes

- Past investigations have over-projected savings from optimisations.
  Don't quote numbers without empirical backing.
- "More obs features" is not always the answer — sometimes the agent has
  enough info but the GA hasn't bred for using it yet. Be specific about
  which.
- If your investigation reveals that ALL THREE questions point to the
  same root cause (e.g. "the recipe is too wide on EW and the agent
  doesn't have per-runner microstructure" might collapse to "filter EW
  or add 3 lean-obs features"), say so.

## When you're done

Present at the top of the findings:

1. A one-paragraph "the answer" summary.
2. The three answers in order with evidence.
3. A single recommended next step the operator could try in the NEXT
   cohort (one change, low risk).

Then wait for the operator to decide.
