---
plan: rewrite
status: planning
opened: 2026-04-26
type: phased rewrite of trainer + policy + supervised pipeline
---

# Rewrite — replace the trainer and policy, keep the env

## Why a rewrite

`plans/greenfield_review.md` traced the recurring problems
(force-close rate stuck ~75 %, selectivity gap, partial H2 attenuation,
H1 label conflation) back to one upstream choice: a 70-dim continuous-
multi-head action space (5 heads × 14 runners, all independent
Gaussians) for a problem that's fundamentally "pick which horse to
open on this tick, if any".

That choice forced everything downstream:

- A scalar value head (per-runner critic is awkward on continuous-
  multi-head action spaces).
- Joint-trained auxiliary heads inside the policy (to bolt on the
  per-runner discrimination the action structure doesn't have).
- 8+ shaped reward terms (compensating for credit-assignment loss).
- Two-stage entropy controller + per-mini-batch KL early-stop +
  advantage normalisation + LR warmup (defending against high-dim
  continuous policy instability).

Each individual addition was the right local move. The cumulative
system is solving the wrong shape of problem. We're rewriting.

## What survives

| Component | Why |
|---|---|
| `env/` — `BetfairEnv`, `ExchangeMatcher`, `BetManager`, `_settle_current_race`, force-close logic | The simulator is the moat. Correct single-price matching, junk filter, hard caps, force-close, equal-profit sizing. **The env is the trainer's misstep, not the env's.** |
| Data pipeline (`data/episode_builder.py`, parquet ingestion, day loading) | Independent of policy choice. |
| Registry (`registry/model_store.py`, `models.db`) | Same shape works for new architecture (new arch_name, new state_dict shape). |
| Frontend (Vite/React UI, websocket events, scoreboard) | Keep. Adapter layer on the trainer side may be needed. |
| Test infrastructure for env (`tests/test_*` for env / matcher / bet_manager) | Stays valid. |
| GA infrastructure (worker pool, breeding, mutation) | Decision in `greenfield_review.md` Part 4: keep GA. The new architecture has fewer genes (~6–8) but the parallel-evaluation infrastructure earns its keep. |

## What gets ripped out

| Component | Replaced by |
|---|---|
| `agents/ppo_trainer.py` | New `agents_v2/discrete_ppo_trainer.py` — half the size, no entropy controller, no advantage normalisation gymnastics, no LR warmup, no reward centering. Discrete PPO is well-trodden. |
| `agents/policy_network.py` (LSTM/TimeLSTM/Transformer with 5-head per-runner) | New policy classes with discrete-categorical action head + small continuous heads (stake, aggression on the chosen runner). Per-runner value head from day one. |
| All auxiliary heads (`fill_prob_head`, `mature_prob_head`, `risk_head`) trained jointly | Standalone supervised scorer (Phase 0). Frozen, fed to actor as features. |
| 8+ reward shaping terms | Realised P&L only. |
| Entropy controllers (per-head floor + SAC alpha + BC handshake) | Plain categorical entropy bonus, fixed coefficient. |
| BC pretrain machinery | Not needed — supervised scorer pre-trains the discriminative half. |

## Phasing

| Phase | What | Deliverable | Estimate |
|---|---|---|---|
| **−1** | Env audit | Confirm the env matches the spec in `docs/betfair_market_model.md`. Read-only review. Any divergence is filed and gated before Phase 0 starts. | 1 session, 1–2 hr |
| **0** | Standalone supervised scorer | Frozen `P(mature \| features)` classifier trained on historical data, with calibration plot and held-out AUC. No RL touched. | 1 week, 2–3 sessions |
| **1** | New policy + env wiring | Discrete-action policy classes (categorical + small continuous heads), per-runner value head, env shim. Smoke test runs end-to-end with random weights. **No training yet.** | 3–4 days, 2 sessions |
| **2** | New trainer | Discrete-action PPO with per-runner GAE. Trains a single agent for one episode, loss curves sane. | 1 week, 3 sessions |
| **3** | GPU pathway + multi-day + GA cohort + frontend wiring | Wire v2 to CUDA (parity-tested vs CPU); extend train CLI to multi-day; build `training_v2/cohort/` (worker pool, gene schema, breeding); adapt trainer events to existing websocket schema; run 12-agent / 7-day cohort; compare to cohort-M. | ~1 week, 4 sessions |

**Total: ~3 weeks of focused work to a comparable cohort.**

## Success bar (Phase 3)

A v2 cohort beats v1 cohort-M on all three:

1. Mean force-close rate **< 50 %** (vs current ~75 %).
2. ρ(open_cost-equivalent gene, fc_rate) **≤ −0.5** (vs current ~0).
3. **At least one agent positive on raw P&L** on the held-out test
   window (vs current 0–7/66 historically).

If all three hit, the rewrite is the new baseline. If any miss, write
up findings, decide whether to iterate inside v2 or step back.

## Hard constraints (apply to ALL phases)

1. **Don't touch the env.** Phase −1 may surface bugs; if so, fix them
   in their own session BEFORE proceeding to Phase 0. No env changes
   bundled into rewrite phases.
2. **Don't touch the data pipeline.** The parquet ingestion and day
   loader are out of scope.
3. **Parallel tree.** New code lives under `agents_v2/`,
   `training_v2/`. Old code keeps running and is the comparison
   baseline. **Do not delete old code until Phase 3 succeeds.**
4. **No "while we're at it" refactors.** If we touch a non-rewrite
   file, that's scope creep. Flag it as a follow-on plan, do not
   bundle.
5. **No new shaped rewards in v2.** If a problem looks like it wants
   a shaping term, that's a sign the architecture isn't doing the
   work. Stop and rethink, don't shape.
6. **No new entropy / KL / advantage gymnastics in v2.** Same reason.
   Discrete PPO is supposed to be simple.
7. **Don't pre-empt cohort-M's verdict** as a justification. Cohort-M
   is being killed (operator decision 2026-04-26). The rewrite is
   justified by the architectural argument in `greenfield_review.md`,
   not by cohort-M's outcome.

## Out of scope (for this whole plan)

- Changing the env / matcher / bet_manager.
- Changing the data pipeline.
- Changing the live-inference repo (`ai-betfair`).
- Switching from GA to BayesOpt (decided: keep GA — see
  `greenfield_review.md` Part 4 D6 and the 2026-04-26 chat record).
- Adding new market types or new races.
- UI redesign (frontend stays as-is; only the data going INTO it
  changes).

## Navigation

- `phase-minus-1-env-audit/session_prompts/01_env_audit.md`
- `phase-0-supervised-scorer/purpose.md`
- `phase-0-supervised-scorer/session_prompts/01_label_and_feature_design.md`
- `phase-0-supervised-scorer/session_prompts/02_train_and_evaluate.md`
- `phase-1-policy-and-env-wiring/purpose.md` (GREEN, 2026-04-27)
- `phase-2-trainer/purpose.md` + `findings.md` (AMBER, 2026-04-29)
- `phase-3-cohort/purpose.md` + 4 session prompts (design-locked, 2026-04-29)

## Lessons carried over from v1

These are not architectural choices — they're hard-won correctness
facts about the env / data / Betfair mechanics. They survive into v2:

- Bet count counts matched orders, not netted positions
  (CLAUDE.md "Bet accounting").
- Single-price matching, no walking; LTP-required junk filter; hard
  price cap inside the matcher (CLAUDE.md "Order matching").
- Equal-profit pair sizing: `S_lay = S_back × [P_back × (1−c) + c] / (P_lay − c)` (CLAUDE.md "Equal-profit pair sizing").
- Force-close at T−N uses the relaxed matcher path AND overdrafts the
  per-race budget (CLAUDE.md "Force-close at T−N").
- `info["realised_pnl"]` is last-race-only; use `day_pnl` for episode
  P&L. `bet_manager.bets` is also last-race-only; use
  `env.all_settled_bets` for the full episode (CLAUDE.md "info[
  realised_pnl] is last-race-only").
