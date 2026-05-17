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
