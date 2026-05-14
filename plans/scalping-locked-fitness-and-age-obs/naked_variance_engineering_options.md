# Engineering options to tighten naked variance

Hand-off doc for the new session, 2026-05-14. The
`scalping-locked-fitness-and-age-obs` plan as scaffolded has Lever
1 = `composite_score = locked + 0.25 * naked` which rewards naked
mean but doesn't penalise variance. The 7-day forward reeval
showed this isn't enough: top-5 by composite_score were all
high-variance even after locked-floor-weighting.

This doc lays out 5 directions for the new session to consider.
Each is independent; some compose. Start with the cheapest.

## Inputs available (when the GPU sweep finishes — currently running)

- `registry/_predictor_SCALPING_layq_1778712871/all_bets.parquet` — every
  bet from every lay-quality-gate agent on in-sample days
  2026-05-04/05/06. 12,149 bets across 58 agents.
- `registry/_predictor_SCALPING_layq_1778712871/phenotypes.csv` — per-agent
  side mix, prices, pwins.
- `registry/_predictor_SCALPING_raceconf_1778661062/naked_pnl_per_leg.csv`
  — (in progress) per-naked-leg pnl for raceconf cohort. ~50min ETA on GPU.
- Per-leg variance distribution analysis script:
  `C:/tmp/naked_variance_analysis.py` (computes per-agent σ_leg etc).
  Should be promoted to `tools/build_naked_variance_report.py`.

## Engineering options

### Option A — selection metric penalises variance (Lever 1 revision)

**Change `scalping-locked-fitness-and-age-obs` plan's Lever 1.**

```python
# Current proposal:
score = locked_pnl + 0.25 * naked_pnl

# Better options:
# (1) Penalise variance directly (need σ_leg computed at eval time)
score = locked_pnl - lambda_var * sigma_naked_leg * sqrt(n_naked)

# (2) Sharpe-like
score = locked_pnl / (1 + alpha * sigma_naked_leg)

# (3) Hard filter
score = locked_pnl if sigma_naked_leg <= 30 else -infinity
```

**Effort:** Low — 50 lines of code in `worker.py`. The eval rollout
already collects per-bet pnl in scoreboard; just need to compute
σ across naked legs during eval and store it as a new column.

**Risk:** Picking the right λ/α/threshold is empirical. Should
probe on the lay-quality-gate cohort's bet logs first.

**Probe path:** before committing to a λ/α value, rank the
lay-quality-gate cohort's 58 agents by each candidate formula and
see if the *generalising* agents (3a91f162, 942240e3, 61cff936,
etc.) surface in the top-5.

### Option B — reward shaping to penalise per-leg variance

**Add a reward term that punishes large naked pnl outcomes
during training.**

Currently the env's `shaped_bonus` has:
- `-0.95 * sum(max(0, per_pair_naked_pnl))` (clip windfalls)
- `+£1 per close_signal success`
- naked-loss-scale × negative-naked

The new term:
```
shaped += -beta * abs(per_pair_naked_pnl)
```

Or equivalently the L2 form:
```
shaped += -gamma * per_pair_naked_pnl ** 2
```

These directly punish variance — both wins AND losses on big naked
outcomes cost reward. The agent learns to either avoid the leg or
close it before the outcome becomes large.

**Effort:** Medium — 1 new gene, env change, env tests, no
architecture-hash break (it's a reward-only change).

**Risk:** Could collapse the agent toward "never open anything"
if β is too high. Probe with small values first (β = 0.01-0.05).

### Option C — close-cost obs feature pushes proactive closing

Already proposed (Phase 2 of `scalping-locked-fitness-and-age-obs`)
as `seconds_since_aggressive_placed`. The hypothesis is that this
feature lets the agent learn to close stale pairs before they go
naked.

This is **complementary to A and B** — A/B change the optimisation
target, C gives the agent the obs-side signal to act on it.

**Effort:** Already scaffolded; just needs to land in code.

### Option D — force-close in training (deferred Lever 3)

Train with `force_close_before_off_seconds = 60` or `120`. Caps
every naked pair at "small loss" magnitude (the force-close
spread cost, typically £2-£3 per closed leg) — physically removes
the high-variance tail.

**Effort:** Low — change one training param. But the cohort needs
to be re-trained to get the policy adapted to the new env.

**Risk:** Changes the entire optimisation surface. May produce
different agent phenotypes than the current cohort. Lose
comparability to predecessor cohorts on raw P&L (force-close
costs are baked in).

This is the cleanest fix per
`project_force_close_train_vs_deploy.md`: train and deploy with
the same fc setting.

### Option E — gate-side hard cap on individual leg exposure

Add an env hard cap: any single bet whose worst-case naked loss
exceeds some threshold (e.g. £30) is refused at action-mask time.

```python
# In compute_mask, after the pwin / lay-price-max / race-confidence
# gates, add:
worst_case_loss = leverage_obs_features.worst_case_naked_pnl[runner]
if abs(worst_case_loss) > MAX_LEG_LOSS:
    mask[OPEN_BACK / OPEN_LAY] = False
```

This uses the Phase 2b leverage features already in the obs to
hard-refuse high-leverage opens.

**Effort:** Low — 1 new env kwarg + 5 lines in compute_mask.

**Risk:** Could starve the agent of opportunities. Probe the
admitted set EV before launching.

## Recommended sequence

1. **First** — Option A with formula (3) (hard filter). This is
   the cheapest test: rank the EXISTING lay-quality-gate cohort
   by `locked if σ_leg ≤ 30 else 0`, see if the top-5 from that
   ranking outperforms the composite_score top-5 on the 7-day
   forward reeval. Cost: 0 retraining, ~1h.

2. **If (1) confirms the variance metric is right** — Option B
   (reward shaping). Train a new cohort with the new shaping term
   active. The session-handoff for the next-plan can fold this
   in alongside the existing Lever 1+2.

3. **In parallel with (2)** — Option D (force-close in training).
   These compose: shaping makes the agent avoid high-variance
   opens, force-close bounds the realised tail. Together they
   should produce very tight per-leg pnl distributions.

4. **Option E (gate cap)** is the conservative "patch" if (B/D)
   take longer than expected — it works on the existing trained
   weights, costs nothing at train time.

5. **Option C** is already scaffolded; land it alongside any of
   the above so the agent has the obs-side signal to act on the
   reward signal.

## Open questions for the new session

- What's the right λ/α/threshold value? Probe on existing bet logs.
- Should the variance penalty be on σ_leg or max-loss-leg? Different
  agent phenotypes optimise for different bounded statistics.
- Is the variance reduction worth potential mean-pnl reduction? The
  realised forward-day fc=120 result was ~£0/day net; agents could
  end up with bounded variance AROUND ZERO not around a high mean.
  Need to quantify the tradeoff.

## Plan-level question

Is `scalping-locked-fitness-and-age-obs` (as scaffolded) still the
right next plan, OR should it be redesigned around variance
penalisation?

- **Keep as scaffolded, modify Lever 1:** swap the formula to one
  that includes variance, land everything else (exclude-days flag,
  age-obs feature). This is the smallest scope-change.
- **Redesign:** new plan `scalping-tight-variance` with Options A+B+C
  bundled. Drops the locked-weighted score formula in favour of
  explicit variance penalisation.

I'd recommend the former (modify Lever 1 in place). The plan name
becomes slightly misleading but the work is otherwise sound. Decide
when promoting the plan.
