# Session prompt — PPO KL explosion + force-close P&L magnitude investigation

**Scope:** investigation only. No code changes to env / trainer / reward math.
Deliverable is a written diagnosis + a follow-on plan skeleton (new folder under
`plans/`) for the fix work. Implementation happens in a later session, not this
one.

**Runs in parallel with live training.** The `arb-signal-cleanup-probe` plan is
actively training (cohort W in progress, cohort A still to come). **Do not
stop, pause, or interfere with the running worker.** Read `logs/worker.log`,
`logs/training/episodes.jsonl`, and the registry in read-only mode. Do not
touch `config.yaml`, the running process, or the GPU.

---

## Why this session exists

The `arb-signal-cleanup-probe` gen-0 and gen-1 review (2026-04-23) surfaced two
independent problems that the probe plan itself cannot resolve, because they
are upstream of the mechanisms it's testing:

### Problem 1 — PPO KL explosion on every update

Across every cohort, every architecture, every agent, every gen, every day:

```
PPO KL early-stop after epoch 0: approx_kl=1667.5862 > threshold=0.0300
PPO KL early-stop after epoch 0: approx_kl=3652.1375 > threshold=0.0300
PPO KL early-stop after epoch 0: approx_kl=2804.4443 > threshold=0.0300
...
```

`approx_kl` sits in the 1,000–7,000 range against the 0.03 early-stop
threshold. **Every PPO update early-stops after exactly one epoch**, skipping
the remaining 3. The run is effectively doing one gradient pass per day per
agent through PPO — BC pretrain is doing the real teaching, PPO is starved.

Policy-loss is modest (0.19–1.47), so gradients aren't literally NaN-ing. But
the trust-region is not being respected; the policy is making huge log-prob
shifts in a single epoch and then locking. This is the same class of bug
fixed in `plans/policy-startup-stability/` (commit 8b8ca67, fresh-init
explosion → advantage normalisation added), so check whether advantage norm is
actually wired on the live code path for this probe's configs — not just
whether it exists in `agents/ppo_trainer.py`.

**This is a bug, not a signal.** Reproducible across every configuration.
More generations will not produce new evidence.

### Problem 2 — Force-close P&L magnitude swamps the reward signal

Gen-1 cohort W (27 agents, 18 eps each):

| Metric | Mean per race |
|---|---|
| `arbs_force_closed` | ~183 |
| `scalping_force_closed_pnl` | **−£183** |
| `arbs_completed` | 22 (good — gen-0 was ~0) |
| `arbs_closed` | 17 |
| `arbs_naked` | 22 |

The plan's rationale still holds per-pair (`CLAUDE.md` "Force-close at T−N
(2026-04-21)"): a ±£0.50–£3 spread cost is strictly better than ±£100s of
naked variance. **But at 183 closes/race the aggregate cost is the entire
reward budget.** The force-close term alone explains the gen-1 median
last-5-ep pnl of −£43.56 and mean of −£126.

The top-performing gen-1 agents are not the ones closing pairs well — they
are the ones who have learned to bet less (top-3 have 90–260 force-closes;
bottom-6 have 333–395). That's the wrong optimisation pressure: we want
agents to arb, not to silence.

This is a **design-review** question, not a bug. Options to evaluate in the
write-up:

- Tighter `max_back_price`/`max_lay_price` cap on the relaxed matcher path
  (force-close-only) so the worst spread-cost tail is clipped.
- Time-phased escalation: try `close_signal` at T−(N+5), then force-close at
  T−N — give the agent a brief window to close at its preferred price before
  the env crosses the book.
- Smaller hedge sizing on force-close (fractional equal-profit) when remaining
  book depth is thin.
- Per-race force-close budget cap (e.g. max 50 force-closes/race).
- Leave it alone if the follow-on analysis shows force-close P&L is already
  bounded and improving across episodes within an agent.

Don't pick one — write up the tradeoffs and let the operator choose.

---

## What to read first (read-only)

- **`CLAUDE.md`** — sections:
  - "Force-close at T−N (2026-04-21)" (bet accounting)
  - "PPO update stability — advantage normalisation"
  - "Reward centering: units contract"
  - "Entropy control — target-entropy controller (2026-04-19)"
- **`plans/policy-startup-stability/`** — the prior fix for fresh-init PPO
  explosion. Especially `lessons_learnt.md` and the final `progress.md`
  entry. Advantage normalisation was the fix; confirm it's actually live on
  the arb-signal-cleanup probe's code path.
- **`plans/naked-clip-and-stability/lessons_learnt.md`** — Session 03 reward
  centering units bug trace. Similar shape of failure — check whether the
  same bug class can recur here.
- **`plans/arb-signal-cleanup/purpose.md` + `hard_constraints.md`** —
  understand what the probe is testing and what it assumes the trainer
  is doing correctly. §16 pins `alpha_lr` as a per-agent gene; §7/§14
  exclude force-closes from matured-arb and close_signal bonuses.
- **`agents/ppo_trainer.py`** — grep for:
  - `advantages.mean()` / `adv_std` — where advantage norm happens, and
    whether it happens BEFORE or AFTER mini-batch split.
  - `_update_reward_baseline` — units contract. The 2026-04-18 bug was
    episode-sum vs per-step; confirm the caller is still passing per-step.
  - `approx_kl` — the computation. Is it per-transition mean or sum? Is it
    using the post-norm or pre-norm advantages?
  - `_compute_advantages` — GAE calculation. Check γ, λ, and whether the
    reward stream it consumes has already been EMA-centered.
  - `self._log_alpha`, `self._alpha_optimizer` — make sure alpha_lr gene is
    plumbed; an alpha_lr=0.1 agent saturating at clamp-max might produce KL
    via the entropy-bonus gradient, not the policy-surrogate gradient.
- **`env/betfair_env.py`** — force-close path:
  - `_attempt_close` (around line 1961) with `force_close=True`.
  - Step-loop time-to-off gate (around line 1458).
  - `_settle_current_race` (around line 2190) — where
    `scalping_force_closed_pnl` is accumulated. Confirm the sign convention
    and that the matcher's own refusal counts (`force_close_refused_*`)
    land on the JSONL row.
- **`env/exchange_matcher.py::ExchangeMatcher`** — the `force_close` flag
  path. LTP drop + ±50% junk-filter skip + hard cap still enforced. Understand
  the exact cap values in `config.yaml` and whether they're biting on the
  force-close path.
- **`logs/worker.log`** and **`logs/training/episodes.jsonl`** — primary
  evidence. Grep worker.log for `approx_kl=` across a single agent's
  18-episode run to get the KL trajectory. JSONL has `policy_loss`,
  `value_loss`, `entropy`, `alpha`, `force_close_*` fields.
- **`registry/training_plans/*.json`** — the three probe plan files. Check
  which cohort's agents' JSONL rows you're reading.

## Evidence already collected (2026-04-23 review)

Don't re-derive these; they're ground truth:

- Gen 0 complete: 54 models × 18 eps. Winner `313cec8e` (ppo_lstm_v1)
  composite=0.413, pnl=+£143.87. #2–#5 all ≤ +£0.45 pnl.
- Gen 1 in progress: 27/42 agents done (all cohort W, all
  `ppo_transformer_v1` ctx=256). Mean last-5 pnl=−£126, median=−£43.56.
  Positives: 5/27 (3 carried from gen 0, 2 pure gen 1 — one of which is
  near-silent at 83 bets/ep).
- `arbs_completed` jumped 0 → 22/race between gens — the one clear win.
  `arbs_closed`, `arbs_naked`, `arbs_force_closed` barely moved.
- KL explosion is *systemic*: appears on ep 1 of every agent (including
  BC-pretrained ones), continues every episode, doesn't decay.
- α controller is bimodal across the population: half at clamp-min (1e-5),
  half near clamp-max (0.08–0.10). Suggests `alpha_lr` gene values at the
  top of its range overshoot.

## Specific hypotheses to test (evidence-first, cheap wins)

Rank each by effort and payoff. Do the cheap ones first.

### H1 — Advantage normalisation is not on the live code path
- **Check:** grep `agents/ppo_trainer.py` for advantage mean/std subtraction.
  Confirm it runs inside `_ppo_update` on the tensor fed to the surrogate
  loss, not just in a helper that may be bypassed.
- **Test:** run a single-agent rollout with a debugger breakpoint; print
  advantage stats before and after the norm step.
- **Payoff:** if off, turning it on re-closes the 2026-04-16
  policy-startup-stability fix. Should drop KL by orders of magnitude.

### H2 — Reward centering units bug recurrence
- **Check:** `_update_reward_baseline` caller. 2026-04-18 bug was passing
  episode-sum; check whether any new code path (BC pretrain integration,
  the new `alpha_lr` gene plumbing) re-introduced the same mistake.
- **Test:** `tests/test_ppo_trainer.py::test_real_ppo_update_feeds_per_step_mean_to_baseline`
  should still be green — run it and confirm.

### H3 — KL is computed on the wrong distribution
- **Check:** is `approx_kl` computed on pre- or post-advantage-norm log-prob
  ratios? Is it using the old policy's log-probs from the rollout, or a
  stale snapshot?
- **Test:** instrument one update to log `(new_logp - old_logp).mean()`,
  `(new_logp - old_logp).std()`, `ratio.mean()`, `ratio.std()`. An honest KL
  should be <<1 on a first epoch.

### H4 — Reward magnitudes (force-close P&L dominating) are the root cause
- **Check:** does KL correlate with `scalping_force_closed_pnl` magnitude
  per-episode? If force-close-heavy episodes have higher KL than force-close-
  light ones (within the same agent), the reward shape itself is the
  gradient problem.
- **Test:** pull KL per episode from worker.log, join with force-close-pnl
  from episodes.jsonl, compute rank correlation.
- **Payoff:** if true, Problem 2 (force-close magnitude) is upstream of
  Problem 1 (KL explosion) and fixing sizing fixes both.

### H5 — α at clamp-max drives KL via entropy-bonus gradient
- **Check:** do agents with `alpha` near 0.1 have higher `approx_kl` than
  agents at 1e-5? Same rank-correlation test as H4.
- **Payoff:** if true, the `alpha_lr` gene range needs a ceiling.

## Deliverables

1. **`plans/ppo-stability-and-force-close-investigation/findings.md`** — a
   write-up with:
   - Which hypotheses passed / failed, with evidence (log excerpts,
     correlation numbers, test output).
   - Whether KL explosion is independent of force-close magnitude (H4) —
     this determines whether it's one problem or two.
   - Recommendation: fix order and whether the running probe can continue
     or needs to be killed.
2. **`plans/ppo-kl-fix/`** — follow-on plan skeleton (`purpose.md`,
   `hard_constraints.md`, `master_todo.md`) for the PPO fix. Scope to one
   session if possible.
3. **`plans/force-close-sizing-review/`** — follow-on plan skeleton for the
   force-close design review, listing the 4–5 design options with tradeoffs
   and an operator-pick section.
4. **Do NOT implement fixes in this session.** Investigation + plan
   skeletons only. Implementation plans go to the operator for approval
   and scheduling against the live training.

## What NOT to do

- Do not modify `config.yaml`, `env/betfair_env.py`, `agents/ppo_trainer.py`,
  `env/exchange_matcher.py`, or any file in the live-training code path.
- Do not run `pytest` with `-x` or any suite that rebuilds caches — the
  training worker has them mmap'd.
- Do not spawn a second training run, even a smoke test, until the running
  worker has finished cohort W and cohort A.
- Do not archive or rename anything under `registry/`.
- Do not create a new plan with `status: "active"` — drafts only.
- Do not touch memory files unless the investigation surfaces a durable
  lesson worth saving.

## Exit criteria

- `findings.md` answers: is KL explosion a bug in `agents/ppo_trainer.py`,
  a consequence of force-close P&L magnitude, or both independently? With
  evidence.
- Two follow-on plan skeletons exist and are ready for operator review.
- No running process was disturbed; `logs/worker.log` shows continuous
  progress from the moment you started to the moment you stopped.
