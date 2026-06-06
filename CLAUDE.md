# RL-Betfair — Claude Notes

**Current invariants and conventions** that aren't obvious from the code. Read
before touching `env/`, `agents_v2/`, the reward path, or the cohort runner.

This file is *current truth, slimmed*. For each item the dated rationale, worked
examples, scoreboard-comparability history, and design trail live in the linked
`plans/<name>/` folder — **read the plan before you change that knob.**

Three rules recur, so they're stated once here:
- **Byte-identity.** Most knobs default to a no-op value that reproduces
  pre-change runs bit-for-bit; the plan/test says which value.
- **Scoreboard comparability.** A reward-shape change shifts `shaped_bonus` /
  `total_reward` magnitudes but leaves `raw_pnl_reward` meaning-stable — compare
  rows across such a change on **`raw_pnl_reward`**, never on `shaped_bonus`.
- **`raw + shaped ≈ total_reward`** every episode — the master invariant. If it
  breaks, a term was added outside an accumulator; fix the accounting.

---

## Fast cohort training: `--parallel-agents` (multiprocess, default 16)

`training_v2/cohort/runner.py` trains the cohort as N parallel solo
`train_one_agent` PROCESSES (1 thread each) — **~8–9× cluster-day,
bit-identical** (each worker is the golden solo path at its own seed). Default
**16** (throughput peak on a 20-core box; `tools/measure_optimal_n.py`
calibrates per machine).

- **N is the concurrency CAP, not cohort size.** M>N runs `ceil(M/N)` waves; a
  warm persistent pool + per-worker caches span waves+generations.
- **Predictor runs supported** (the intended config): workers rebuild the bundle
  from manifest paths (`_worker_load_bundle`). Pass `--use-race-outcome-predictor
  --predictor-bundle-manifests …`.
- **Shared-memory day cache** (the predictors-ON OOM fix): the master bakes each
  day's downstream `static_obs` float32 arrays (~93 MB/day) as a per-day `.npy`
  + gate-cache sidecar; workers `np.load(mmap_mode='r')` so the OS page cache
  holds ONE shared copy. Automatic for predictors-ON multiprocess; predictor-OFF
  keeps the legacy per-day dict cache. RAM no longer caps N up to the core count.
  Engine: `training_v2/cohort/static_obs_cache.py`. Plan:
  `plans/shared-memory-day-cache/`.
- Mutually exclusive with `--batched` (older GPU-batched path). `0` = sequential.
  Per-gen wall is memory-bandwidth-contention-bound; past ~9× needs the
  tensor-env rewrite, not tuning.
- Guards: `tests/test_v2_multiproc_cluster.py` + the parallel-vs-sequential
  bit-identity probes + `tests/test_env_golden_parity.py::
  test_static_obs_cache_path_matches_from_scratch`. Plans:
  `plans/training-speedup-v2/`, `plans/shared-memory-day-cache/`.

---

## Bet accounting: matched orders, not netted positions

Simulator/market spec: [docs/betfair_market_model.md](docs/betfair_market_model.md)
— cross-check before changing any matching / passive-fill logic.

**"Bet count" = distinct matched orders, not distinct netted positions** — the
way Betfair's API rate-limit counts, not the way the UI shows open positions.
Backing the same runner twice (T1@5, T2@3) makes **two** `Bet` objects in
`BetManager.bets`; the UI would show one position averaged to 4. P&L is identical
(associative); the only divergence is a tiny extra `efficiency_penalty ×
bet_count` for building a position across fills — intentional (live, that really
costs market-impact + execution risk). Read `max_bets_per_race: 20` as "hit the
exchange up to 20 times per race", not "hold 20 positions".

### Force-close at T−N

When `force_close_before_off_seconds > 0` and `scalping_mode` is on, the env
force-closes any pair with an unfilled second leg once `time_to_off ≤` threshold.
Default `0` = disabled = byte-identical. Behaviour:

- **Relaxed matcher** (`force_close=True` flag on `match_back/lay/pick_top_price`):
  LTP requirement dropped, ±`max_price_deviation_pct` junk filter skipped, but
  the **hard `max_back_price`/`max_lay_price` cap still enforced** and the
  single-price no-walk contract still holds. Crossing a thin/unpriced book
  (±£0.50–£5 spread) always beats leaving a pair naked (±£100s variance).
  Agent-initiated `close_signal` keeps the STRICT match.
- **Overdraft allowed:** `place_back/lay` with `force_close=True` bypass the
  per-race budget gate (the live trader banks more than one race's capital); cost
  flows through `race_pnl` at settle. `MIN_BET_STAKE` (£2) still applies.
- **Sizing = equal-profit** (same `equal_profit_*_stake` helpers as
  `close_signal`), so the hedge is bounded by `~spread × stake` with no
  race-outcome variance. (A 1:1-stake variant was tried and reverted — at drifted
  prices it's wildly asymmetric and blew up KL.)
- `race_pnl = scalping_locked_pnl + scalping_closed_pnl +
  scalping_force_closed_pnl + scaled_naked_sum`. Force-closes are EXCLUDED from
  the matured-arb bonus and the close_signal bonus (the agent didn't choose
  them). Refusal counters `force_close_refused_{no_book,place,above_cap}` on the
  info dict. See `env/exchange_matcher.py::_match`, `env/bet_manager.py`,
  `plans/arb-signal-cleanup/`.

**Train vs deploy:** force_close stays **0 in training** (keep the naked-variance
signal so the agent learns selectivity); deploy/held-out eval applies
`force_close=120` via `tools/reevaluate_cohort.py --reward-overrides
force_close_before_off_seconds=120`. Memory: `project_force_close_train_vs_deploy`.

---

## Order matching: single-price, no walking

All matching goes through `env/exchange_matcher.py::ExchangeMatcher`, kept
dependency-free (only `dataclasses`/`typing` + a `PriceLevel` protocol) so it
**vendors into `ai-betfair` unmodified** — preserve that contract (the matcher
never imports `tick_ladder`; callers compute tick limits).

Three rules, priority order — **any PR that breaks one re-introduces the
phantom-profit bug; reject it:**
1. **No ladder walking (OPEN path).** A bet matches at ONE price — the best
   post-filter level. Excess stake is unmatched, not spilled to the next level.
2. **Junk filter.** Levels >`max_price_deviation_pct` (±50%) from the runner's
   **LTP** are dropped before matching. No LTP ⇒ unpriceable ⇒ refused. (Real
   ladders carry £1–£1000 parked orders; walking them = phantom profits of tens
   of thousands per bet.)
3. **Hard price cap** enforced INSIDE the matcher AFTER the junk filter (gating
   on the unfiltered `ladder[0]` fails open against a £1000 junk top-of-book).

### Sanctioned exception: bounded walk on the CLOSE path only

The no-walk ban is OPEN-only. The close/force-close path may walk a bounded
number of ticks to complete its hedge, gated by `close_walk_ticks` (default `0` =
OFF = byte-identical). Safe because it's **bounded** AND the hard cap is enforced
on every walked level (can't reach junk) — a named limit order, which
`docs/betfair_market_model.md §2` confirms is real Betfair behaviour ("fills at
the named price OR better"). Fixes under-hedged aggressive legs (~£339/agent/7d
of avoidable directional loss). Mechanism: `match_*(walk_to_price=…)` fills
best→limit returning a size-weighted avg; `_attempt_close` computes
`walk_to_price = tick_offset(close_price, close_walk_ticks, dir)` (dir=+1 lay
close, −1 back close); the OPEN path never passes it. Guards:
`tests/test_exchange_matcher.py::TestCloseWalk`,
`tests/test_forced_arbitrage.py::TestCloseWalkWiring`. Plan:
`plans/recipe-expansion-and-robustness/findings.md` (KEY FINDING #2).

---

## Pair sizing: equal-profit (not equal-exposure)

The auto-paired passive and the `close_signal` leg are sized to **equalise net
profit on both race outcomes after commission**, not exposure:

    S_lay  = S_back × [P_back·(1−c) + c] / (P_lay − c)
    S_back = S_lay  × (P_lay − c) / [P_back·(1−c) + c]    (lay-first, the true inverse)

So `locked_pnl = min(win, lose)` reports the real lock, not the near-zero floor
of an over-laid trade. At `c=0` this collapses to the legacy
`S_back·P_back/P_lay` exposure formula. Pre-`f7a09fc` (2026-04-18) equal-exposure
sizing is a valid pre-fix reference only. Plan: `plans/scalping-equal-profit-sizing/`.

---

## Price-adaptive arb_spread

The aggressive↔passive tick offset is computed from ONE per-agent gene —
`arb_spread_target_lock_pct` (the target locked-profit fraction) — NOT from a
policy action. In `_process_action` (open) and `_attempt_requote` (re-quote):

    arb_ticks = min_arb_ticks_for_profit(agg_price, side, commission,
                    profit_floor=arb_spread_target_lock_pct, max_ticks=MAX_ARB_TICKS)
    None ⇒ refuse pair (commission_infeasible);  else clip to [1, 25]

`min_arb_ticks_for_profit` returns the smallest tick offset whose
**equal-profit-sized** scalp locks ≥ `profit_floor` per £1 aggressive stake — so
the same gene value locks roughly the same % at every price (~5–15% price move
needed across the whole ladder; scalping high-odds horses IS viable). Always
active in `scalping_mode` (no opt-in flag). Gene ∈ [0.005, 0.05], default 0.02;
phenotype handle: 0.005 = fill-seeker (tight, high fill), 0.05 = profit-seeker
(wide, big lock). Operator pin: `--arb-spread-target-lock-pct` (mutually
exclusive with `--enable-gene arb_spread_target_lock_pct`).

- The floor function uses **equal-profit** sizing with a required `aggressive_side`
  arg (back-first ≠ lay-first under equal-profit). An earlier equal-EXPOSURE floor
  was 2–5× too wide. Call sites passing `aggressive_side`: `_attempt_requote`,
  `training_v2/arb_oracle.py`, `training/arb_oracle.py`.
- Force-close is UNAFFECTED (relaxed matcher, crosses the spread at market).
- The policy's per-runner `arb_spread` action dim is dead code (the v2 shim
  hardcodes it; the formula ignores the env-side read).
- Guards: `tests/test_forced_arbitrage.py::TestPriceAdaptiveArbSpread`,
  `tests/test_scalping_math.py::TestLockedPnlEqualProfit`. Plan:
  `plans/force_close_and_arb_spread/`.

---

## Reward function: raw vs shaped

`env/betfair_env.py::_settle_current_race` splits per-race reward into two
accumulators, both surfaced on `info["raw_pnl_reward"]` / `info["shaped_bonus"]`
and `logs/training/episodes.jsonl`:

- **Raw** = `race_pnl` (whole-race cashflow) + terminal `day_pnl/starting_budget`
  bonus. In scalping mode `race_pnl = scalping_locked_pnl + scalping_closed_pnl +
  scalping_force_closed_pnl + scaled_naked_sum` — truthful about every £ that
  moved, including loss-closed pairs.
- **Shaped** = zero-mean-in-expectation training-signal terms (below).

**Naked handling:** raw carries naked LOSSES at full cash value; shaped neuters
95% of naked WINDFALLS (`−0.95·Σ max(0, per_pair_naked)`) so directional luck
earns nothing. Aggregation is **per-pair**, so a lucky winner can't cancel an
unrelated loser. Plans: `plans/scalping-naked-asymmetry/`,
`plans/naked-clip-and-stability/`.

### Symmetry around random betting — DO NOT "fix" to non-negative
`early_pick_bonus` (applies to all settled back bets, winners AND losers) and
`precision_reward` (centred at 0.5: `(precision−0.5)·bonus`) are **zero-mean for
a random policy**. A previous non-symmetric version gave random betting positive
expected shaped reward — it taught the agent to bet more without caring whether it
won. Keep them symmetric.

### Per-step mark-to-market shaping
Per tick, `shaped += mark_to_market_weight · (MTM_t − MTM_{t-1})` where portfolio
MTM = Σ over open bets of `S·(P_matched−P_current)/P_current` (back; sign-flipped
for lay). **Telescopes to zero at settle** (resolved bets drop out; the settle
step emits `−MTM_{t-1}`), so it only REDISTRIBUTES existing race P&L to the ticks
that caused it — the raw+shaped invariant holds. Unpriceable runners (no LTP or
LTP≤1) contribute zero MTM; commission applies at settle, not MTM (no
double-count). Default weight **0.2** (gene `mark_to_market_weight` ∈ [0.05, 0.5];
raised from 0.05 — the per-tick gradient at the open decision was too weak vs
value-function noise). Guard: `tests/test_mark_to_market.py::
test_invariant_raw_plus_shaped_with_nonzero_weight`. Plan: `plans/reward-densification/`.

### Naked-loss annealing
`scaled_naked_sum = Σ (min(0,p)·naked_loss_scale + max(0,p))` over per-pair naked
P&L. Default scale 1.0 = byte-identical; plan-level `naked_loss_anneal:
{start_gen, end_gen}` ramps each agent's effective scale → 1.0 to bootstrap past
the naked valley early. scale<1 runs not comparable on `raw_pnl_reward`.

### Matured-arb bonus
Per-pair shaped reward for pairs that matured, zero-mean vs an expected-random
pair count, clipped to ±cap. Default weight 0.0. Counts ONLY natural maturation
(`scalping_arbs_completed`) — NOT agent-closed/force-closed (an agent-closed pair
cancelled its passive and crossed at market, not scalping). Probe with
`--reward-overrides matured_arb_bonus_weight=2`. Plan: `plans/arb-curriculum/`.

### Selective-open shaping
Per-pair symmetric term: charge `−open_cost` on the OPEN tick, refund `+open_cost`
on the RESOLUTION tick iff the pair matures or is agent-closed (force-close/naked
do NOT refund). Per-race total = `open_cost·(refund_count − pairs_opened)`. Safe
by three properties: zero-mean under "always mature/close"; **per-tick** credit
(NOT settle-time — GAE smears a settle-time chunk across ~5000 ticks and the
gradient drowns); net pressure is "be selective" not "stop opening" (the matured
bonus offsets it). Lives ENTIRELY in shaped (raw buckets untouched). Default 0.0;
hard-bound [0.0, 4.0] (widened from 2.0 in the spray-and-bail redesign, with a
high-biased prior). Guards: `tests/test_forced_arbitrage.py::TestSelectiveOpenShaping`
(8 tests). Plan: `plans/selective-open-shaping/`.

### Shaped-penalty warmup
`training.shaped_penalty_warmup_eps` linearly scales ONLY `efficiency_cost` +
`precision_reward` 0→1 over the first N PPO rollout episodes (default 0 = no-op),
giving a post-BC policy a penalty-lite window. Only the two penalties warm up —
warming the positive terms would reward "do nothing". BC episodes don't count
toward the index. Plan: `plans/arb-signal-cleanup/`.

### CLOSE_SIGNAL_BONUS = 0.0
The shaped per-`close_signal`-success bonus is **0** (was £1 → £0.5 → zeroed):
closing learns from raw cash only — the bonus structurally competed against
natural maturation, which has no equivalent shaped reward. Guard:
`tests/test_forced_arbitrage.py::TestScalpingReward`.

---

## PPO update stability

### Advantage normalisation (load-bearing)
The update normalises the per-mini-batch advantage to mean-0/std-1 before the
surrogate loss (`(adv − adv.mean())/(adv.std()+1e-8)`). Load-bearing for
large-magnitude rewards (every scalping run, ±£500/ep) — without it fresh agents
explode (`policy_loss` 10⁴–10¹⁴ on ep1) and permanently lose `close_signal`/
`requote_signal`. Reward magnitudes unchanged. Plan: `plans/policy-startup-stability/`.

### Reward centering: per-step units, NOT episode-sum
`PPOTrainer._update_reward_baseline(x)` expects `x = sum(training_reward)/n_steps`
(per-step), subtracted per-step in `_compute_advantages`. Passing the episode SUM
shifts every step by the whole-episode total → GAE returns ~`shifted/(1−γλ)` →
`value_loss` explodes to O(1e8+) next rollout. Guard (load-bearing, INTEGRATION
not unit — a caller-only drift silently passes the unit tests):
`tests/test_ppo_trainer.py::test_real_ppo_update_feeds_per_step_mean_to_baseline`.
Plan: `plans/naked-clip-and-stability/`.

### Recurrent PPO: hidden-state protocol on update
The update must condition on the hidden state the rollout saw, else stateful
`old_log_probs` vs stateless `new_log_probs` blow up `approx_kl` every update.
`Transition.hidden_state_in` carries the state passed INTO the forward that
produced the transition (captured BEFORE the forward; t=0 is zero). LSTM/TimeLSTM:
`(h,c)` each `(layers,1,hidden)`, batch axis dim 1 (those classes override
pack/slice to dim 1); Transformer: `(buffer, valid_count)`, batch dim 0
(BasePolicy default). `_ppo_update` packs/slices per mini-batch via
`policy.pack_hidden_states/slice_hidden_states`. **Action-clipping contract:** the
`Transition` stores the UN-clipped sampled action (the env gets `np.clip(±1)`) so
`dist.log_prob(stored)` at update matches rollout — storing the CLIPPED action
adds ~13 nats of KL drift. Guards: `tests/test_ppo_trainer.py::
TestRecurrentStateThroughPpoUpdate` (4 tests).

### Per-mini-batch KL early-stop
The KL check runs INSIDE the mini-batch loop (not end-of-epoch); on breach it
stops the whole update (current + remaining epochs). Reads
`(mb_old_log_probs − new_log_probs).mean()` after `optimiser.step()`. Default
threshold **0.15** (`hp["kl_early_stop_threshold"]`); end-of-epoch checks fired
only after ~156 steps, by which point healthy per-step drift accumulates past 3.0.
`loss_info["n_updates"]` surfaces the gradient steps actually run (near
`ppo_epochs × mini_batches` = healthy; low = KL tripped). Guard:
`test_kl_early_stop_is_per_mini_batch_not_per_epoch`. Plans: `plans/ppo-kl-fix/`,
`plans/ppo-stability-and-force-close-investigation/`.

---

## Entropy control — target-entropy controller

The entropy coefficient is a LEARNED variable. A separate **SGD(momentum=0)**
optimiser (`alpha_lr`) on `log_alpha = log(entropy_coeff)` holds forward-pass
entropy at `target_entropy = 150`. **SGD not Adam is deliberate** — SGD's update
`log_alpha −= lr·(current_entropy − target)` is literal proportional control;
Adam's per-param normalisation destroys that and can't track drift at our
one-call-per-`_ppo_update` cadence. `log_alpha` clamped `[log(1e-5), log(0.1)]`
(saturation = a valid failure signal, surface it); float64 for a clean log→exp
round trip. Without the controller, entropy drifts 139→200+ over 15 eps and
`close_signal`/`requote_signal` lose probability mass. Reward magnitudes
unchanged. Guard: `tests/test_ppo_trainer.py::TestTargetEntropyController::
test_real_ppo_update_updates_log_alpha`. Plan: `plans/entropy-control-v2/`.

- **`alpha_lr` is a per-agent gene** ([1e-2, 1e-1], default 1e-2 = byte-identical).
  Set once at construction, never mutated.
- **BC-pretrain warmup handshake:** when `bc_pretrain_steps > 0`, the EFFECTIVE
  target anneals from the (low) post-BC entropy → 150 over
  `bc_target_entropy_warmup_eps` episodes (gene, default 5; 0 disables), so the
  controller doesn't undo BC on update 1. `episodes.jsonl` logs the effective target.

---

## BC pretrain
Per-agent behavioural cloning on arb-oracle samples runs before PPO when
`bc_pretrain_steps > 0`. Only `actor_head` trains (value/LSTM/encoders frozen,
restored after); BC uses its own Adam (PPO optimiser untouched). **Per-agent,
NEVER shared** — sharing BC weights collapses GA diversity irreparably. Genes:
`bc_pretrain_steps`, `bc_learning_rate`, `bc_target_entropy_warmup_eps`,
`bc_direction_target_weight` (blends oracle CE with direction CE). Plan:
`plans/arb-curriculum/`.

## Curriculum day ordering
`training.curriculum_day_order ∈ {random (default), density_desc, density_asc}`
orders each agent's training days by arb-oracle density. `density_desc` (arb-rich
first) pairs with BC warm-start. Every day still seen once per epoch (order, not
membership). Missing oracle cache = density 0 + a warning; invalid mode → random +
error, never crashes.

---

## Policy architecture notes

### Transformer context window ∈ {32, 64, 128, 256}
`transformer_ctx_ticks` is a structural gene. Races average ~150–250 ticks, so
ctx32 sees ~13%, ctx256 the full median race. `position_embedding`/causal mask
size off the gene (no arch change needed). A transformer at one ctx CANNOT
cross-load weights at another (different `position_embedding` shape) — the
arch-hash check treats each value as a distinct variant. Fresh-blood SAMPLE is
capped ≤128 / d_model≤256 (ctx256/d512 are sequential-rollout stragglers that
don't out-champion smaller); the VALID set keeps 256/512 so prior big champions
warm-load. See `training_v2/cohort/genes.py`.

### fill_prob + mature_prob feed actor_head (v1 and v2)
Two per-runner auxiliary heads feed their sigmoid output into `actor_head` as
extra columns: `actor_input = [runner_embs, backbone, fill_prob, mature_prob]`, so
`actor_head[0].weight.shape[1] == runner_embed + backbone + 2`. Applied
unconditionally (no gene gate). **Do not detach** — the surrogate-loss path flows
through both heads so the policy can learn discriminative features, not just
oracle-matched ones.
- **The label distinction is the point.** `fill_prob` label = `count≥2` (conflates
  matured + agent-closed + **force-closed**). `mature_prob` label is strictly
  stricter: force-closed pairs → 0 (negative class). `fill_prob` alone steered the
  actor TOWARD runners that end in force-close (cohort-F
  `ρ(fill_prob_loss_weight, fc_rate)=+0.469`); `mature_prob` supplies the
  discrimination `fill_prob` structurally can't.
- **Arch-hash break:** pre-plan weights (narrower `actor_head[0].weight`) fail
  `load_state_dict(strict=True)` by design (else silently-garbled actions).
- Weights default 0.0 → the head runs but its column is a benign ~0.5 constant.
- Guards: `tests/test_policy_network.py::{TestFillProbInActor, TestMatureProbInActor}`
  (input-width, gradient-through forward+backward, cross-load-fails). Plans:
  `plans/fill-prob-in-actor/`, `plans/per-runner-credit/`.

### v2 stack consumes aux-head loss weights
`agents_v2/discrete_policy.py::DiscreteLSTMPolicy` +
`training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer` consume
`fill_prob_loss_weight`, `mature_prob_loss_weight`, `risk_loss_weight` (v1
contracts carry over verbatim). `risk_head` is per-runner `nn.Linear(hidden,
max_runners·2)` with a log-var clamp; does NOT feed actor_input — surfaces on
`PolicyOutput.predicted_locked_pnl_per_runner` and shapes the backbone via
Gaussian NLL (label `max(0, min(win_pnl, lose_pnl))` post-commission; naked pairs
NaN-masked). **v2 reads weights from `hp` ONLY (no config fallback)** — v2's `hp`
comes from `CohortGenes.to_dict()`, which always populates every key at its
default, so the v1 `hp.get(name, config[...])` fallback would silently swallow
`--reward-overrides`; the worker pre-merges overrides into `hp` via
`_build_trainer_hp` (Path A). `mature_prob_loss_weight` range [1.0, 5.0]. Plan:
`plans/rewrite/phase-7-port-aux-heads/`.

---

## Foot guns

### `info["realised_pnl"]` is last-race-only — use `info["day_pnl"]`
The env recreates a fresh `BetManager` per race, so `realised_pnl` accumulates
within one race then resets. The day's true P&L is `info["day_pnl"]` (PPO reads it
into `EpisodeStats.total_pnl`). Likewise **`env.bet_manager.bets` is
last-race-only** — for the full day's bet history read `env.all_settled_bets`
(accumulated in `step` before the BetManager is replaced). Reading the wrong one
is how the phantom-profit bug hid. Plan: `plans/next_steps/bugs.md` B1.

### Cohort launch flags: the Path-A precedence foot gun
A knob with BOTH a CLI flag and a `trainer_hp` entry must resolve with **OR
semantics**, not `.get(default)`:

    knob = bool(cli_flag_arg) or bool(trainer_hp.get("knob", False))   # correct
    knob = bool(trainer_hp.get("knob", cli_flag_arg))                  # ALWAYS BROKEN

`CohortGenes.to_dict()` always includes the key, so `.get(key, fallback)` never
uses the fallback → the CLI flag is silently discarded. It hides because the
env-side and policy-side gates have SEPARATE paths — the CLI flag flips one, the
broken `.get` leaves the other off. **Detection:** a knob's refusal/activity
counter (`gate_refusals`, `pwin_back_gate_refusals`) reading 0 on agent-1/day-1
when the knob should be active = broken wiring. Canonical fix: a `_resolve_<knob>`
helper + a both-sources test —
`training_v2/cohort/worker.py::_resolve_direction_gate_enabled`,
`tests/test_v2_direction_gate.py::TestResolvePolicyGateEnabled`. Separately, a knob
exposed BOTH as `--enable-gene` and as a cohort-flag/`--reward-overrides` enforces
one-source-of-truth (collision guard in `runner.py`). Memory:
`feedback_audit_launch_wiring`.
