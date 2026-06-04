# Explorations

Analytical discussions, ad-hoc data questions, and strategic reasoning
that surface things we want to remember but aren't necessarily acting
on. Companion to `EXPERIMENTS.md` (which records launched cohorts and
their verdicts).

Per-entry structure:
- **Date + topic** as the H2
- **Question** that triggered the analysis
- **What the data showed** with concrete numbers
- **Interpretation** — what we now believe
- **Implications / queued thoughts** — what (if anything) this should
  change in future plans

Append-only. Don't edit historical entries; add new ones below.

---

## 2026-05-17 — Back-first vs lay-first phenotype in tnv2

**Question**: Two predecessor cohorts had a bimodal back-first / lay-first
distribution. Has tnv2 (the variance-penalty cohort under raceconf gate)
also split into two phenotypes? And is the apparent back-first edge in
day_pnl real or in-sample luck?

**What the data showed** (across 67 trained tnv2 agents, in-sample
training-eval window of 10 days):

- Bimodal phenotype has collapsed. Every agent is lay-leaning. Range
  30.9 % → 46.9 % back-first; median 42 %. Zero agents in the
  >50 % back-first bucket.
- Correlation back_pct ↔ day_pnl: **+0.435**.
- Decomposed: back_pct ↔ locked_pnl = **+0.864**; back_pct ↔ naked_pnl
  = **+0.152**; back_pct ↔ force_close_pnl = −0.355.
- Top 5 agents by back_pct have mean locked = £105/d vs bottom 5's £75
  — a structural £30/d advantage to back-leaning.

**Interpretation**: the bimodal split is gate-dependent. raceconf
(race_confidence_threshold = 0.50) + no lay_price_max admits a different
race population than the predecessor lay-quality-gate cohort. In this
gate, agents converge on a mixed phenotype with a lay-first bias.

The +0.435 back_pct ↔ day_pnl correlation is structural, not luck. It's
driven by the locked floor (back-first scalps mature more reliably in
this gate), not by naked variance. **The single positive-PnL agent
4c217d70 (+£19 in-sample) was an exception**: it had the cohort-typical
locked floor but caught a +£27 naked tailwind. Held-out reeval (2026-05-17
fc=0 oldwindow) reverted it to **−£49**, exactly the predicted
sample-noise collapse.

**Implications**:
- Treat single-agent in-sample positive-PnL with suspicion. The cohort
  pattern (back_pct → locked floor) is the trustworthy signal.
- The bimodal phenotype only emerges under gate configurations that
  force a directional choice (e.g. lay_price_max bounds the lay-side
  price range, pushing agents toward lay-favourite). With no such
  bound, agents do both sides at differential intensity.
- For tnv3's day_pnl_per_std selection: the GA will naturally select
  toward back-leaning (since locked drives the day_pnl numerator).

---

## 2026-05-17 — Price-bucket stratification of first legs

**Question**: When agents place their first leg, what price bucket are
they choosing, and is the price-action pattern doing something
strategic? (Follow-on: would lay-first outsiders have favorable EV on
the naked side?)

**What the data showed** (60,306 first legs, tnv2 cohort-wide):

| Price | back | lay | back % |
|---|---:|---:|---:|
| 1.5–3 (heavy favs) | 9,011 | 992 | **90 %** |
| 3–5 (favs) | 8,364 | 5,404 | 61 % |
| 5–10 (mid) | 6,480 | 22,144 | **23 %** |
| 10–15 | 1,132 | 6,578 | 15 % |
| 15–30 (outsiders) | 201 | 0 | 100 % |
| 30+ | 0 | 0 | — |

Naked outcomes (470 naked legs):
- Lay-first nakeds at 3–5: **91 % positive**, mean +£8.90 (n=23)
- Lay-first nakeds at 10–15: 75 % positive, mean +£3.38 (n=123)
- Back-first nakeds at 3–5: 24 % positive, mean −£5.86 (n=86)
- Back-first nakeds at 15–30: **0 % positive** (n=16)
- Cohort-wide: lay-first nakeds +£2.23/leg (60 % positive); back-first
  nakeds −£2.45/leg (30 % positive).

**Interpretation**: the agents have learned a clear price-action
strategy:
1. **Back heavy favourites** (price 1.5–5). Backs at low prices have
   positive matched-arb economics and decent fill rates because volume
   is high.
2. **Lay mid-prices** (5–15). Most runners in this band lose. Naked-lay
   outcomes are structurally profitable (favourable EV on the lay
   side: keep the back stake when the runner loses, which is most
   of the time).
3. **Never lay outsiders** (>15). The +EV math says you'd win the lay
   ~95 % of the time, but the spread cost at high prices is huge
   (bid-ask might be 25→50, a 100 % spread), liquidity is thin
   (force-close can pay a £20+ liability), and force-close at price
   30+ on an unmatched lay is catastrophic. Agents have learned this
   structurally — they almost never operate at >15 either side.

The 201 "back-first at 15–30" cases are likely the residue of failed
high-priced scalp attempts (passive lay-at-high never fills, leaving
the aggressive back as the only matched leg), not deliberate
"back-the-outsider" strategy.

**Why the 0.3 % outsider residue isn't driven to zero**:
- Sparse training signal (201 trades × £2–5 cost = trivial vs the
  £80/day fc-cost problem)
- PPO learns per-pair, not per-class — no explicit "don't outsider-
  scalp" abstraction
- Action space is continuous in tick-space; no price-bucket gate
- mature_prob_head sees outsiders ~0.3 % of training and so doesn't
  learn the class well
- No architectural penalty matches per-bucket EV (open_cost applies
  uniformly across the ladder)

The agent has self-limited outsider trading to 0.3 % without any
architectural help. Pushing it to 0 % would save ~£10/day per agent —
real but small vs the £80/day fc problem currently in focus.

**Implications**:
- The "tailwind" the user asked about (favourable naked EV) IS real —
  but it's on lay-first nakeds at mid prices, not on back-first
  nakeds.
- The back-leaning agents' day_pnl edge is from locked floor, not from
  catching the lay-naked tailwind.
- For a future plan: if we want to push outsider-trading to zero
  more aggressively, options are (a) `back_price_max` hard mask
  (mirrors existing `lay_price_max`), (b) per-bucket `open_cost`
  genes, (c) raise `mature_prob_loss_weight` floor so the head trains
  harder on rare classes. None worth doing now — small fish.

---

## 2026-05-17 — How GA selection metric shapes the agent's economics

**Question**: tnv2 used `locked_per_std = locked_pnl / (1 + naked_std)`
as the GA selection score. Every top agent had locked ~£100/d and fc
cost ~−£90/d, netting ~−£20/d on day_pnl. Why does the GA accept this
trade-off, and what does the corrected `day_pnl_per_std` do
differently?

**What the data showed**:

tnv2 top-10 agents had a consistent shape:
- locked: +£80 to +£110/d (high — what the metric rewards)
- fc cost: −£75 to −£95/d (high — invisible to the metric)
- closed: −£1 to −£15/d (small)
- naked: −£40 to +£30/d (mostly negative; one outlier +£27)
- **day_pnl: −£5 to −£25/d** (the only positive was 4c217d70 at +£19 via naked luck)

`locked_per_std` rewards two things:
1. High locked_pnl (numerator) → high-volume opens that mature
2. Low naked_std (denominator) → tight variance

Both are achievable via the SAME phenotype: open many pairs aggressively
(high locked AND high pair count). The env's fc=120 mechanic auto-
flattens the un-matured ones at T−120, which keeps naked_std tight
(force-close caps the naked tail). But the £75–95/d fc cost from
those auto-flattens lives in `day_pnl`, not in `locked` or `naked_std`.
**The metric is blind to the cost it incurs.**

**Interpretation**: tnv2's selection signal was structurally
misaligned with the actual deployment objective. The GA correctly
optimised it — and the result was agents that look great on the
metric and terrible on the deployment-relevant cash.

`day_pnl_per_std` (tnv3) uses the cash bottom line as the numerator:
- High day_pnl now requires either high locked OR low fc cost (or
  both)
- The GA can't game the locked-floor lever without paying the cost
- Trade-off: re-introduces naked-sign reading at selection time
  (which bit tnv1 at 3-day eval), but at 10-day eval the noise drops
  ~1.8× — manageable

**Implications**:
- The architecture lesson is broader than this plan: **any selection
  metric that ignores one cost component will select for agents that
  maximise that cost.** "Locked floor / std" rewards opening volume
  while ignoring fc; "naked-only" rewards exposure while ignoring
  locked; "matured-pair count" rewards over-opening for the bonus.
- The "right" metric for a scalping strategy is probably `day_pnl /
  σ(day_pnl)` — proper Sharpe on cash — but that requires enough
  in-sample days to estimate σ(day_pnl) reliably. With 10 eval days
  this might be feasible; tnv3 settles for `day_pnl / (1 +
  σ(naked))` as a 1.8×-tighter-than-3-day variant.
- If tnv3 also fails to clear bands, the next iteration should probably
  go all-in on full Sharpe (`day_pnl / σ(day_pnl)`) with even more
  eval days (e.g. 15+).


---

## 2026-05-25 — Retrieval / analogue-forecasting architecture viability

**Question**: Can a vector-encoded pre-off price trajectory + kNN
lookup on historical analogues predict the next 5 min of LTP
better than naive baselines? If yes, retrieval might be worth
investing in as a complement (or replacement) to the parametric
PPO+LSTM stack. Side-thread probe under
[plans/trajectory-retrieval-probe/](trajectory-retrieval-probe/).

**What the data showed** (18,033 index rows / 6,934 query rows on
query days 2026-05-05 → 2026-05-14, k=5 nearest-neighbour on 10
hand-engineered z-scored features):

| Method | MAE | vs B1 | dir_acc |
|---|---|---|---|
| B1 constant (`Δlog LTP = 0`) | **0.14168** | — | (degenerate) |
| B2 linear extrap | 0.17118 | −20.8 % | 0.483 |
| B3 per-fav-rank prior | 0.14292 | −0.9 % | 0.549 |
| kNN k=5 | 0.15915 | **−12.3 %** | 0.517 |

The constant-prediction baseline beats every alternative including
the kNN retrieval. Target |mean| log-return is ~14 % over 5 min
(prices DO move a lot) — but the moves are unpredictable from the
10-feature embedding tried.

**Interpretation**: short-horizon pre-off LTP direction looks like
a near-random-walk relative to cheap hand-engineered features. The
kNN does beat B2 on both MAE and directional accuracy (so it IS
extracting some signal), just not enough to clear the variance of
log-returns. Decision rule (locked in purpose.md before any data
was looked at) fires PARK.

**Implications / queued thoughts**:

- The negative DOES NOT condemn the current PPO+LSTM stack. The
  current stack's value-add is spread capture via scalping pair
  lifecycle (locked PnL) and selective open behaviour, NOT
  directional prediction. Different problem.
- Three retrieval questions remain open and might have better
  signal-to-noise: (1) fill-probability retrieval (binary target,
  not continuous log-return), (2) race-outcome retrieval at
  race-start (the existing `betfair-predictors` problem in
  non-parametric form), (3) mature-probability retrieval (same
  shape as #1). None blocked by this probe; each is its own
  viability question.
- Practical artifact preserved: `scratch/trajectory_retrieval/`
  contains a clean long-form tick history (3.55M rows) and a
  query-ready feature dataset (29.7k rows). Re-runnable from
  the script in <3 min. Reusable for future probes on different
  prediction targets.
- Two feature-engineering lessons promoted to memory
  (`feedback_feature_engineering_diagnostics.md`): value-domain
  checks beat shape-domain checks; ~90σ z-scores are bug signals
  not tail signals.

---

## 2026-05-27: Naked-EV is structural, not noise — and force_close eats it

**Trigger:** Across 7 different recipes in the recipe-expansion
campaign (Round 6 cells), the per-episode naked P&L term was
consistently POSITIVE — ranging +£44 to +£95/day. This wasn't
predicted by anyone's prior; in fact the team's earlier framing
(memo `naked_variance_primary_metric`) treated naked variance as
the deployment risk to suppress.

**Question:** is the consistency across 7 independent
hyperparameter recipes evidence that naked P&L is **structurally
EV-positive** (i.e., a real edge from back-leg selection) rather
than directional luck on a particular eval window?

**Analysis:**

- 4 agents × 5 eval days = 20 datapoints per recipe.
- Across recipes, mean naked stayed in +£44 to +£95 range despite
  varied selection levers (pwin thresholds, BC dose, augmentation
  on/off, env gates).
- A naked pair is one whose passive lay leg never filled before
  T-force_close. The agent placed the back leg at some price X.
  If at T-120s the lay hasn't filled, env CROSSES THE SPREAD —
  paying ~£3-5 of spread to flatten. That force-closed loss was
  -£125/day across cells.
- If we remove the force-close, the naked legs settle at race
  outcome. The +£77 naked-pnl observation suggested **back legs
  on these specific runners pay out positively at settlement**.

**Hypothesis:** the agent's back-leg selection (after BC's
oracle alignment + pwin_back's selection prior) is structurally
EV-positive. The env's force-close-at-T-120s is throwing away
the EV by crossing thin spreads when nakeds could just settle.

**Test:** Round 6.5 (6 cells) — disable env force-close
(`force_close_before_off_seconds=0`).

**Result:** Hypothesis confirmed. 5 of 6 cells positive
day_pnl. 20 of 20 agents positive. Mean +£214, range +£71 to
+£370. Naked term jumped +£77 → +£270 as more pairs settled
naked. Locked dropped to +£4-7 but naked dominates.

**Interpretation:**

- The "naked variance is risk" framing of the
  `naked_variance_primary_metric` memo was WRONG for this
  recipe. Naked is the SIGNAL, not the noise.
- The recipe's back-leg selection (BC trained on oracle samples
  at price 5-30, pwin_back gating to p_win ≥ 0.20) appears to
  pick runners that drift longer-than-fair but ultimately settle
  toward win — a real EV edge on the live ladder.
- Force-close was an unprincipled cost ceiling that didn't know
  about this edge. Disabling it captures the EV.

**Caveats / open questions for follow-up:**

1. The eval window (5 fixed days) might be a lucky drift slice.
   Need held-out eval probe.
2. Live trading carries the naked position INTO the in-play
   period. Settlement variance under live conditions may differ
   from sim. Slippage, adverse selection, partial fills,
   in-play market dynamics — none modelled here.
3. Locked P&L is only +£4-7, so the recipe is NOT a scalping
   recipe in the original framing — it's a "selective
   back-leg with no env exit" recipe. Should be re-labelled
   for clarity.
4. The L/σ_naked ratio across agents is ~0.04-0.10 — locked is
   tiny relative to naked variance. The recipe's profitability
   is entirely contingent on naked EV staying positive.
5. Why is naked EV positive in the first place? The agent
   selects back legs on horses that drift but win — could be
   a real market inefficiency the predictors are capturing, or
   could be eval-window artefact. Need to characterise the
   per-runner profile of winning nakeds.

**Methodological note:** this is an example of a metric being
"obviously bad" by prior framing turning out to be the actual
edge once interrogated. The `naked_variance_primary_metric` memo
should be amended — naked variance per leg is the WRONG
deployment metric for this recipe family.

---

## 2026-05-28: CORRECTION — the naked-EV "edge" was eval-window overfitting

**This entry corrects the 2026-05-27 entry above, which concluded
naked-EV was structural. It was NOT.**

The fc=0 recipe produced +£260-287/day in-sample (eval days
2026-04-10..05-06). Held-out validation on 14 never-seen days
(2026-05-07..05-20) collapsed it to **-£155 to -£195/day** — every
held-out agent-day negative.

**What happened:** across 60+ probe cells we repeatedly evaluated and
selected on the SAME 5 April eval days. That's classic eval-set
overfitting. The naked back-legs that "won more than fair" did so
only on those specific 5 days' race outcomes. The policy behaviour
was identical out-of-sample (same opens, mat%, cls%) but the naked
directional P&L flipped sign.

**Corrected conclusions:**
1. Naked P&L is ~zero-EV directional variance (negative after
   commission). It is the deployment RISK, not an edge. The original
   `naked_variance_primary_metric` memo was right.
2. force_close is a SAFETY RAIL that bounds naked variance into
   bounded spread cost — not a profit-eating bug. Keep it ON.
3. The only structural edge is LOCKED P&L (genuine scalping spread
   capture), and it's small (+£4-20/day in these probes).
4. **METHODOLOGY FIX (locked in): every experiment must eval on
   held-out days never used in training OR selection. Maintain a
   train / iteration-eval / final-test split. Never iterate against
   training-adjacent days.** This single discipline would have caught
   the mirage 2 days earlier.

**Cost of the lesson:** ~2 days of GPU chasing a recipe that loses
money live. Cheap relative to deploying it. This is the highest-value
negative result of the campaign.

---

## 2026-05-30: Per-open economics — why "selectivity alone" can't reach profitability

**Trigger:** Round N showed N4 (pwin band 0.20-0.50) hit -£78 on
held-out — the best ever. Tempting to read "selectivity is THE
lever" and just push it further. Operator pushed back: "low but
positive £ per scalp and high mat rate will lead to profitability"
— don't conflate "less bad" with "actually positive."

**The economics, written down to keep us honest:**

For any recipe, the daily P&L decomposes as:

```
day_pnl = mat_count × locked/pair
        − fc_count × fc_cost/pair
        − close_count × close_cost/pair
        + naked (≈ 0 in expectation, ignore)
```

Per-open contribution at our current operating point (e.g. H1 E7):

| component | mat | locked/pair | fc% | fc_cost/pair | per-open |
|---|---:|---:|---:|---:|---:|
| locked side | 5% | £2.5 | — | — | +£0.125 |
| fc side | — | — | ~80% | ~£1.50 | -£1.20 |
| close side | — | — | (~15% × ~£3) | — | -£0.45 |
| **net** | | | | | **-£1.07 / open** |

Selectivity (Round N4: 52 opens) reduces TOTAL day_pnl by reducing
the *number* of negative-EV opens. It does NOT change the per-open
sign. You can drive day_pnl from -£200 → -£78 by opening less, but
you cannot reach +£0 by this alone — the only zero-bet recipe is
day_pnl = £0.

**To flip the per-open sign positive, EITHER:**
1. **mat% × locked/pair must rise** (need ~6× from 0.125 to 0.75).
   Realistic path: mat% from 5% → 30%+ at the cost of locked/pair
   dropping (but the product still rises).
2. **fc% × fc_cost/pair must drop** dramatically. Shorter fc window
   crosses less spread when env force-closes — directly cuts
   fc_cost/pair.
3. **Both moderately** — mat 15% × £1 = £0.15 + fc 50% × £0.6 = £0.30
   negative → net -£0.15/open. Still negative; need more.

**Two distinct directions to explore in parallel — NOT a one-lever
hunt:**

- **Path A: Selectivity** (Round R now testing). Fewer-but-better
  opens. N4 is the leader. Reduces *count* of negative-EV opens but
  ceiling is determined by per-open sign.
- **Path B: mat%-lift** (Round S explicitly designed for this).
  Trade locked/pair for mat%. Extreme tight_lock + short fc window.
  Targets the per-open sign by making more passive legs fill AND
  cutting the cost when they don't.
- **Path C (architectural, queued):** **mature_prob open-gate** —
  use the policy's own mature_prob_head to gate opens at the source.
  Only open pairs the model predicts will mature. Targets per-open
  sign by filtering on PREDICTED fill probability.
- **Path D (future):** Liquidity-aware gating — only open on runners
  with deep enough book that the passive lay actually fills. Needs
  new env field exposed.

**The mistake to avoid:** mono-pursuit of any single lever. The
right campaign explores multiple paths and looks for the one (or
combination) that actually flips per-open economics positive.
"Less negative" is not the goal.

**Implications going forward:**

- Don't celebrate held-out -£78 as "almost there." It's
  fundamentally negative-EV per open, just opened fewer times.
- The mature_prob open-gate is the highest-leverage untried
  mechanism — it directly targets selection-by-predicted-fill
  rather than indirect levers (selectivity, spread).
- Per-cell analysis should always report mat%, locked/pair AND
  fc-cost/pair — not just day_pnl — so we can see which side of
  the per-open arithmetic moved.

---

## 2026-05-30 — Imitation-first: does the scalping opportunity exist, and is it learnable?

**Question**: The recipe-expansion campaign found no held-out-positive
recipe (best −£78/day) with maturation stuck at a ~5% base rate. Before
spending more GPU on online PPO, two cheap diagnostic questions: (1) does
the hindsight arb oracle itself make money out-of-sample (the ceiling)?
(2) is the maturing-open decision predictable from decision-time features
(learnable)? See `plans/imitation-first/findings.md`.

**What the data showed** (7 reserved holdout days, May 20-29, ~505k
candidates; env fc=120 + close_walk=10):

- **The oracle itself bleeds.** Running the spread-placeable oracle's own
  labels through the real env: **−£474/day, 88.7% force-close, 5.1%
  natural maturation.** The oracle labels "a profitable SPREAD is
  placeable", not "the passive will FILL" — it never forward-walks. This
  is the campaign's core flaw, at the labeler.
- **Maturation-conditioning fixes it.** New `scan_day(maturation_
  conditioned=True)` forward-walks each candidate under the ENV-FAITHFUL
  fill model (crossing on available-to-LAY + cumulative traded volume ≥
  queue_ahead — NOT the close/hold heuristic's ATB-touch, which
  over-counted fills 12x). Re-run on holdout: **+£559 locked, fc 89%→31%,
  maturation 5%→67%** (budget-unconstrained); ~breakeven under deployment
  budget.
- **Maturation is predictable.** LightGBM on `(full predictor-injected
  obs → matured?)`, 8 train days → 7 holdout: **holdout AUC 0.76,
  top-decile precision 0.30 = 2.45x base-rate lift.** (Scale-invariant
  trees, so immune to the unnormalized-full-obs problem below.)
- **Full obs is unnormalized.** Audit: 335/2254 dims abs-max > 50 (max
  190k); the v2 policy has no input-norm on the actor path (lean obs
  masked this). A BC-only learnability probe would be confounded — hence
  the LightGBM proxy.

**Interpretation**: the campaign was solving the wrong problem (online
PPO exploration) when the issue was the LABELER. With maturation-
conditioned labels the opportunity is real (~breakeven hindsight ceiling,
locked edge ≈ force-close toll) AND the maturation decision is learnable
(AUC 0.76). The lever to turn breakeven into profit is SELECTIVITY: a
policy that opens only high-mature-probability runners pushes its
opened-subset maturation above the breakeven point. The 2.45x top-decile
lift says the signal to do that is present.

**Implications / queued**: (1) train all future scalping work on
maturation-conditioned labels, not spread-placeable. (2) Step 2 =
reward-aware selective BC→PPO; first unblock the policy input-norm
(opt-in, needs sign-off), then `maturation_reward_mode` + open_cost toll,
fc=120 + close_walk=10 pinned, select on LOCKED, eval on the reserved 7
holdout days once. (3) The recipe + caches are staged
(`plans/imitation-first/`).

---

## 2026-06-04 — Where the training-throughput floor actually goes (two CPU levers we wrongly dismissed)

**Question**: `plans/pbt-gpu-forward/findings.md` concluded the LightGBM
predictor/scorer is **86–97%** of the agent-day (forward 3–14%). Is that
right, and what is the real throughput lever — before building a GPU lane?

**What the data showed** (full-agent profiles, LSTM-128, lean-obs, day
2026-04-10, in `plans/pbt-gpu-forward/_measure/`; cProfile inflates
Python-heavy code, so treat as buckets not decimals):

- **Live env** `train_episode` = 72.9 s. **Cached env** (static_obs cache
  injected — the real multiprocess path) = 62.2 s. Baking saved **~10.6 s**
  = the base `engineer_day` features + the per-RACE race-outcome predictor.
- **The per-TICK direction predictor + scorer do NOT bake out** — even on
  the cached path `feature_extractor` (~8 s), `lightgbm.predict` (~4 s) and
  `isotonic.predict` (~3 s) all still run: ~15 s, **~24%** of the agent-day.
- Forward + PPO update (lstm/linear/backward/adam) = the biggest single
  GPU-able bucket (~20–25% for LSTM-128; **dominant for a big transformer**,
  whose ctx²-attention forward is ~10× the LSTM's). Env sim
  (matching/settlement) ~16%, un-cacheable (depends on the agent's actions).

**Interpretation**: BOTH prior claims were wrong. The doc's "86–97%
LightGBM" took a no-cache scorer profile and pinned it on a cache-on
agent-day. The counter-claim that LightGBM is "baked away / negligible" was
also wrong — the *race-outcome* model bakes (per-race), but the *direction*
(price-mover) model + its scorer features run **per tick** and survive the
cache. Truth in between.

**Implications / queued thoughts**:
- The GPU policy lane (`--gpu-policy-lane`, built 2026-06-04) addresses
  forward+update for big-ctx transformers — the priority.
- **Two CPU levers left on the table — REVISIT when cohort throughput
  matters** (both real, both measured here, both wrongly dismissed earlier):
  1. **Native-compile the direction predictor** (LightGBM → Treelite /
     `lleaves`): ~6–10% of the agent-day, typically 2–10× faster tree
     inference, identical outputs. ALSO helps live deployment
     (`ai-betfair`), where it runs per tick uncached.
  2. **Optimise the scorer `FeatureExtractor`**
     (`training_v2/scorer/feature_extractor.py`): ~14%, Python/numpy
     per-tick velocity loops (`_extract_into`, `_delta_window`,
     `_best_back`, `_spread_in_ticks`). Vectorise / Numba, or bake the
     per-tick direction features into static_obs the way `engineer_day`
     already is.
- Both are dwarfed by the env-sim + per-tick Python-loop floor that only the
  tensor-env (R4) rewrite removes — but they are cheap, self-contained
  ~10–20% wins if a campaign ever needs the cohort faster.
