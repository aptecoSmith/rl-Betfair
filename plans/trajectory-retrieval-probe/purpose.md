---
plan: trajectory-retrieval-probe
status: scaffolded
created: 2026-05-25
side_thread: true
touches_production: false
related:
  - plans/price-direction-predictor/ (parametric alternative to retrieval)
  - plans/direction-predictor-mechanism/
  - plans/predictor-integration/
---

# Purpose — viability test for retrieval / analogue-forecasting architecture

## The idea

Treat each (race, runner) as a **price trajectory vector** over the
final ~30 min before the off, plus surrounding context (other
runners' trajectories, form, venue, etc.). At decision time T,
encode the trajectory-so-far and look up the **k nearest historical
analogues**; their *continuations* form a distribution over what the
rest of the race is likely to look like. That distribution informs
the action.

This is the well-known **analogue forecasting** / **retrieval-based
prediction** family (used in weather forecasting from the 1960s, in
modern ML as the "retrieval-augmented" pattern, and is the
non-parametric cousin of what an LSTM/transformer policy does
internally). The architectural appeal vs. the current PPO+LSTM
stack:

| | Parametric (current) | Retrieval / analogue |
|---|---|---|
| Sample efficiency | Worse on rare combos | Better when DB is large |
| Interpretability | Black box | "Looks like these 5 historical races" — debuggable |
| Distribution shift | Retrain | Just append to index |
| Cold-start (rare horse/venue) | Interpolates OK | Bad — no neighbours |
| What it learns | A function | A *similarity metric* + a database |

The current `fill_prob_head` / `mature_prob_head` / `risk_head` aux
heads already nudge the architecture in this direction (predict
future quantities, feed predictions into the actor). A retrieval
system could either **replace** the LSTM backbone OR **augment** it
as an additional observation channel.

## Why we're probing first, not building

Two things have to be true for retrieval to beat parametric methods
here:

1. The encoder produces vectors where **closeness in feature space
   corresponds to closeness in outcome space**. (If two trajectories
   look the same on our 10 hand-engineered features but evolve
   completely differently in the next 5 min, retrieval gives noise.)
2. There are **enough historical neighbours for any given query**
   for the average-of-top-k to be a meaningful prediction. With ~8.6k
   races × ~9 runners = ~77k trajectories in the index, this is
   plausible but not guaranteed.

The probe answers both questions with a single experiment: **can
hand-engineered kNN predictions of the next 5 minutes of LTP beat
naive baselines (price-stays-constant, linear extrapolation) on
held-out days?** If yes, the architecture is alive and worth a
learned encoder. If no, the encoder problem is harder than the
architecture and we'd want to know that before investing weeks.

## Decision rule (locked before probe runs)

| Probe outcome | What it means | Next step |
|---|---|---|
| Beats B1 (constant) by >10% MAE **and** beats B2 (linear extrap) | The 10-feature embedding captures real signal. | Plan a learned-encoder follow-on; consider integrating retrieval as an aux channel in the v2 policy. |
| Beats B1 by 3-10%, doesn't beat B2 | Marginal — the embedding is helping but not beating cheap baselines. | Try richer features (cross-runner, form data) before deciding. |
| Matches B1 or loses to B2 | This encoder is too crude. | Architecture not ruled out, but the next step is "build a learned encoder", which is a much bigger project. Park unless we have spare cycles. |
| Loses to B1 across the board | Signal-to-noise so low that retrieval likely can't help. | Park the idea. The current parametric models are probably already squeezing what little signal exists. |

The decision rule is locked **before** the probe runs to avoid the
"interpret results to match desired outcome" failure mode.

## What this probe explicitly does NOT do

- **Touch the env, training stack, or any cohort runner.** This is a
  standalone script in [scripts/](scripts/) reading parquets
  directly. The production training path is unaffected.
- **Train a model.** Hand-engineered features only. The whole point
  is to test the architecture's premise before paying the cost of
  building a learned encoder.
- **Make trading decisions.** The probe predicts *price*. The
  prediction → action gap is left for the follow-on plan if the
  probe succeeds.
- **Run on GPU.** Whole experiment is <30 min on CPU.

## Cost & risk

- **Human time:** 1-2 focused days end-to-end (see master_todo for
  phase breakdown).
- **Compute:** trivial. No cohort, no overnight, no GPU.
- **Risk to production work:** zero. New script, new plan folder,
  new scratch outputs — no edits to env / training / agents / tests.
  See [hard_constraints.md](hard_constraints.md) §1.
- **Risk of false-positive:** moderate. A probe that beats B1 by 5 %
  on one date split might not generalise. The probe writes per-day
  + per-venue + per-favourite-rank breakdowns specifically to
  catch this.
- **Risk of false-negative:** moderate. v1 features are deliberately
  crude (10 hand-engineered dims, no cross-runner context, no form).
  A "marginal" result probably means "the encoder is the problem",
  not "the architecture is dead" — captured in the decision rule.
