# Imitation-first scalping — purpose

**Created 2026-05-30.** Entry point for a fresh session. Read this, then
`hard_constraints.md` (locked decisions), then `master_todo.md` (the
three executable steps). NOT yet started.

## One-paragraph thesis

We have a **fixed historical dataset (49 days)** and an **arb oracle that
labels, with hindsight, every profitable scalp** in it. That is the setup
for **imitation / offline learning**, NOT online exploration. The campaign
has spent weeks on **online PPO**, whose core pain here is reward
sparsity — the reward lands at settle, hundreds-to-thousands of ticks
after the open, so "PPO sees ~0 gradient on 99% of steps." We don't need
to *explore* to discover profitable scalps; the oracle already shows us
where they are. So: **learn directly from the oracle (dense, supervised,
sparsity-free), and only add reward-aware RL as polish.** Validate cheaply
before committing GPU-weeks.

## How we got here (so a fresh session doesn't relitigate)

Findings from the recipe-expansion campaign + this session
(`plans/recipe-expansion-and-robustness/findings.md`,
`monitoring_notes.md`):

- **No recipe is held-out positive.** Best was N4 at −£78/day (7 unseen
  days); honest fully-hedged number ~−£125 (close-walk, below).
- **Maturation is the bottleneck.** mat% sits ~5% across every recipe ≈ a
  base rate, i.e. little open-selection edge. Per matured pair the 2%
  scalp edge is real (+£0.5–2% locked, balanced) — there just aren't
  enough matured pairs to cover the force-close toll on the rest.
- **Round T (2026-05-30):** the `mature_prob` open-gate did NOT lift mat%
  (flat ~5% across thresholds) — BUT on lean obs (23-d/runner) + 3 train
  days. Handicapped; not a clean verdict on predictability.
- **Round W (2026-05-30):** close-walk (completing hedges) is a deploy
  **variance-reducer, not a profit lever** — it pays guaranteed spread to
  remove zero-EV directional variance. Keep it ON for deploy honesty.
- **direction-head sweep (2026-05-24):** price-direction (the maturation
  *driver*) is predictable at **held-out AUC ≈ 0.70** — real but weak.
  Conclusion "ceiling is data/signal" was about a **23-d reduced** head;
  it does NOT bound the full feature set.
- **Two starvation axes we never questioned (this session):** trained on
  **3 of 49 days**, and on **lean obs** (23-d/runner) — never the full
  **143-d/runner** obs. The oracle proves scalps exist in this data; real
  traders extract them from the same feed. The signal is present; we made
  it unreachable by starving + compressing the input.
- **Reward delivery (this session):** `per_pair_reward_at_resolution`
  exists — pays a matured pair's locked P&L at the **maturation tick**
  (it's outcome-independent once hedged, so fully known then), not at
  settle. The direct fix to the sparsity problem.
- **Architecture:** transformer + TimeLSTM backbones are *built* in v2 but
  were **never raced** on held-out P&L (whole campaign = LSTM h256). No
  verdict; second-order vs the data/signal/reward levers.

## The plan in one line

**Step 0** prove the opportunity is in the *unseen* data (oracle's own
holdout P&L = the ceiling) → **Step 1** prove it's *learnable from
observable features* (BC the oracle to convergence on full obs + 42 days,
eval on holdout, no PPO, sparsity-free) → **Step 2** only if Step 1
promises, add reward-aware polish (BC→PPO fine-tune / DAgger / offline RL)
to fix BC's distribution-shift gap.

Each step is a cheap, decisive gate before the next. Step 0+1 are
**hours/days and sparsity-free** — far cheaper and more directly
informative than the online-PPO GA the campaign kept reaching for. The
full-obs + cache work feeds every path, so nothing is wasted.

## What's already built (this session) vs TODO

**Built + tested:** close-walk matcher (`close_walk_ticks`), GA
resume/checkpoint (`--resume-from`), `maturation_only_reward` pure helper
(+ tests), the `mature_prob` open-gate. See
`plans/ga-recipe-search/master_todo.md` for those.

**TODO (this plan):** full-obs oracle-cache rebuild (42 days); Step 0
oracle-holdout-P&L harness; Step 1 BC-to-convergence runner + holdout
eval; Step 2 `maturation_reward_mode` settle-wiring +
`per_pair_reward_at_resolution` wiring + the BC→PPO/offline route.

`plans/ga-recipe-search/` holds the shared infrastructure constraints
(data split, full obs, reward shape, `locked_per_std` selection, pins,
resume) and is the **Step 2 PPO-fine-tune vehicle** — this plan leads,
that one is downstream.
