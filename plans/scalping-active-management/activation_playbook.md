# Activation Playbook — turning on the aux-head gradients

## Why this doc exists

Sessions 02 and 03 land the two auxiliary heads (fill-probability +
risk) as **plumbing-off**: `fill_prob_loss_weight = 0.0` and
`risk_loss_weight = 0.0`. Both heads run forward every tick and write
predictions onto each `Bet`, but no gradient flows into them — they
output whatever their orthogonal-init-gain=0.01 produces, which is
≈ 0.5 for the fill-prob sigmoid and noise for the risk head. Any UI
built on top of those predictions (Sessions 04–06) will show
effectively-random confidences until this activation happens.

This doc is the explicit protocol for:

1. Confirming the zero-weight baseline behaves exactly like the
   pre-head code.
2. Sweeping each weight up one head at a time, measuring interference.
3. Picking the best joint weights.
4. Promoting those values to `config/scalping_gen1.yaml` so subsequent
   training runs (including Session 07's validation run) use a trained
   head.

**When to run this:** after Session 03 lands, before Sessions 04–06
(UI) land — otherwise the UI shows badges driven by untrained heads
and operators mistrust the whole feature. It can overlap with the UI
work if those sessions add the surfaces with an "insufficient data"
fallback that hides the badge when the prediction is near its
init-default.

---

## Pre-requisites

- [ ] Sessions 02 + 03 both merged to `master`.
- [ ] `pytest tests/ -q` → full suite green on the head commit.
- [ ] A known-good scalping config exists and is committed. The Gen 1
      config used by commit `7a3968a` is the natural baseline.
- [ ] A held-out eval day range, untouched by training, reserved for
      calibration measurement. Calibration on the training days is
      trivially good (hard_constraints §13: training-day fills train
      the head; only eval-day fills are used to judge how calibrated
      the agent is).

---

## Protocol

### Step A — Zero-weight baseline (sanity)

Purpose: prove that with both weights at 0.0 the training dynamics are
byte-identical to Session 01. If they aren't, there's a plumbing bug
we need to catch before interpreting any non-zero-weight result.

```bash
python -m training.run_training \
  --config config/scalping_gen1.yaml \
  --days 2025-09-01:2025-09-30 \
  --seed 42 \
  --tag activation-A-baseline \
  --overrides reward.fill_prob_loss_weight=0.0 reward.risk_loss_weight=0.0
```

Capture:

- `logs/training/episodes.jsonl` — reward per episode.
- Final scoreboard row: `L/N ratio`, `composite`, `arb_rate`,
  `arbs_completed`, `arbs_naked`, `mean_pnl`.
- Run `scripts/scalping_active_comparison.py` (added in Session 07) to
  dump the fill-prob MACE on the eval-day bet log. Expect ≈ 0.5 at
  every bucket — a random-coin-flip head produces exactly this.

**Acceptance:** episode-level reward trajectory is within numerical
noise (< 1e-4) of a reference run on the same code commit with the
aux-head diff reverted. If it isn't, the plumbing leaked gradient
somewhere (e.g. a non-zero default slipped into a gene passthrough).

### Step B — Fill-prob weight sweep

Risk still off. Each row is a separate training run on the same seed
and day range as Step A, so the only changed variable is the weight.

| Run tag | `fill_prob_loss_weight` | `risk_loss_weight` | Purpose |
|---|---|---|---|
| `activation-B-001` | `0.01` | `0.0` | Low — is there any useful signal at all? |
| `activation-B-010` | `0.10` | `0.0` | Expected working point. |
| `activation-B-100` | `1.00` | `0.0` | Stress test — does the aux gradient swamp PPO? |

Metrics per run:

| Metric | Target | Source | Fails the run if… |
|---|---|---|---|
| `mean_reward` delta vs Step A | within ± 20 % | episodes.jsonl | > 20 % drop (hard_constraints §3) |
| Fill-prob MACE on held-out day | trend *down* vs Step A's ~0.5 | comparison script | MACE > Step A's |
| `arb_rate` on top model | trend *up* vs Step A | scoreboard | drops below Step A |
| Per-head entropy (`arb_spread`, `requote_signal`) | trajectory similar to Step A | training log `action_stats` | collapses to < 50 % of Step A's value (aux gradient is interfering with exploration) |
| Gradient-norm ratio `\|\|aux_grad\|\| / \|\|policy_grad\|\|` | < 1.0 averaged per update | `loss_info` diagnostic | > 1.0 sustained (aux is dominating the optimiser) |

The best of the three runs is the candidate. If all three fail, see
the Failure Modes table below before proceeding to Step C.

### Step C — Risk-head weight sweep

Fix `fill_prob_loss_weight` at the Step B winner. Sweep
`risk_loss_weight` independently:

| Run tag | `fill_prob_loss_weight` | `risk_loss_weight` |
|---|---|---|
| `activation-C-001` | _winner of B_ | `0.01` |
| `activation-C-010` | _winner of B_ | `0.10` |
| `activation-C-100` | _winner of B_ | `1.00` |

Same metrics table as Step B, plus:

| Metric | Target | Source |
|---|---|---|
| Risk-prediction MACE on locked-pnl | Spearman ρ > 0.3 between predicted and realised stddev | comparison script |

### Step D — Combined verification

Re-run at the joint winners `(B*, C*)` for a longer horizon (e.g. full
Gen 1 day range) to confirm the two heads don't multiplicatively
interfere. Expected: joint mean reward ≈ max of the two solo runs
(they should be orthogonal — one predicts fill-or-not, the other
predicts P&L distribution).

If joint reward is materially worse than either solo run, the heads
are competing for backbone capacity — bump backbone width (Session 03
`hidden_size` gene) or split the backbone into two branches. Document
either remediation in `lessons_learnt.md`.

### Step E — Promote to default

Once Step D passes:

1. Edit `config/scalping_gen1.yaml` (or whatever master config new
   runs pick up):

   ```yaml
   reward:
     fill_prob_loss_weight: <B*>
     risk_loss_weight: <C*>
   ```

2. Append an entry to `plans/scalping-active-management/progress.md`
   naming the chosen values, the Step A reward baseline they were
   compared against, and the MACE + Spearman ρ achieved on the
   held-out day.

3. Commit with a message that **explicitly names the reward-scale
   change**, per the cross-session rule in `hard_constraints.md §20`.
   Even though the aux losses live outside the reward accumulators,
   enabling them changes what the optimiser pulls on — operators
   comparing a post-promotion model's P&L against a pre-promotion
   baseline need to know the training signal changed.

At this point Sessions 04–06 UI surfaces are safe to trust: calibration
chips will show meaningful confidences, the reliability diagram will
have structure, the scoreboard MACE column will rank models by actual
self-awareness rather than noise.

---

## Failure modes + remediation

| Symptom | Likely cause | Remediation |
|---|---|---|
| Mean reward drops > 20 % at every Step-B weight | Head shares too much backbone, aux gradient is fighting PPO for the same parameters | Refactor head to condition on a `.detach()`-ed copy of `lstm_last` / `out_last` — aux head still trains, but its gradient doesn't propagate into the shared backbone. **Note:** this loses the "direct gradient to arb_spread" benefit from `purpose.md §2` (one of the three listed gains). Document the trade-off in `lessons_learnt.md` before choosing this. |
| MACE stays ≈ 0.5 at every weight | Labels aren't landing — the backfill loop in `_collect_rollout` is always writing NaN | Add an assert inside `_collect_rollout` that `pair_to_transition` has entries whenever `env.all_settled_bets` has settled pairs with `pair_id`. Fix the mismatch. Likely a `sid → slot_idx` resolution edge case. |
| `arb_spread` entropy collapses within the first 2 epochs | Aux gradient is correlated with policy gradient; PPO sees a reinforcing signal and locks in early | Drop the weight by 10×. If persistent even at `1e-3`, swap BCE for focal loss (γ = 2) so the head de-weights easy examples and the gradient magnitude is lower on average. |
| Fill-prob predictions cluster at 0.5 but MACE still improves | Head is learning the marginal fill rate, not the conditional. Useful-ish but not what we want for per-bet badges. | Move the head's input one layer earlier (before the LSTM output norm) OR bump `lstm_hidden_size` to give the backbone more capacity to encode "which runner is about to fill". |
| Risk-head predictions have wild log-var swings | Log-var not clipped tightly enough (Session 03 spec says clip to `[-8, 4]`) | Tighten to `[-6, 2]` — covers stddev of ~£0.05 to ~£7 at £100 stakes, which is the realistic operating range. |

---

## Acceptance — when are we done?

The activation is **complete** when all four of these hold at the
promoted weights:

1. Mean reward within ± 5 % of the Step A baseline (looser than the
   ± 20 % failure floor — at activation-promote time we want to see
   "essentially the same" not just "not catastrophic").
2. Fill-prob MACE ≤ 10 % on held-out eval days, with each of the four
   buckets in `purpose.md §What success looks like` within ± 10 %
   of the observed rate. (± 5 % per bucket is the long-term aspiration;
   ± 10 % is the first-cut activation bar.)
3. Arb_rate on the top scalper in a fresh Gen 0 population ≥
   baseline arb_rate. The whole plan exists to lift this; if activation
   doesn't move the needle, either the weights are too small or the
   head architecture is wrong. Either way don't promote.
4. The 303-test exit-criteria suite (forced_arbitrage + bet_manager +
   betfair_env + policy_network + ppo_trainer) stays green at the
   promoted config. Regression here means the plumbing interacts with
   some edge case the Session-02 / Session-03 tests didn't cover — go
   back and add the missing test before promoting.

If any of (1)-(3) fails after the full protocol, leave the defaults
at 0.0, document the failure in `lessons_learnt.md`, and open a
follow-up session to investigate. **Do not promote a weight that
fails these bars just to unblock the UI** — the UI is allowed to hide
the badge when the prediction equals the init-default.

---

## Relationship to Session 07

Session 07 in `master_todo.md` ("Training run + analysis") is the
**validation** of the whole plan — it produces the before/after
comparison against the Gen 1 baseline and writes the final story into
`lessons_learnt.md`. This activation playbook is the **prerequisite**
for Session 07: Session 07 needs non-zero weights to demonstrate the
plan's value. The activation runs listed above (A–E) feed their
numbers into Session 07's comparison CSVs directly.

If you run Session 07 with weights at 0.0, you're measuring the
re-quote mechanic (Session 01) in isolation — useful, but not the
plan's net effect.
