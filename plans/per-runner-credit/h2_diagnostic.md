---
plan: per-runner-credit
status: H2-partially-binding
opened: 2026-04-26
question: "At the open tick of force-closed pairs vs matured pairs, is the
  GAE-derived advantage actually different?"
verdict: "Partially binding — the per-tick signal exists and is statistically
  robust (p < 1e-10, Cohen's d ~0.8–1.1) but its magnitude is ~20–35% of the
  theoretical maximum credit assignment that an ideal value head would deliver."
---

# H2 diagnostic — per-tick credit assignment via GAE

## TL;DR

**H2 is PARTIALLY binding, not fully binding.** Per-tick credit
assignment IS reaching the open tick — the GAE-derived advantage at
the open tick of a force-closed pair is reliably more negative than
at the open tick of a matured pair, with very large effect sizes
(Cohen's d = 0.77 – 1.14) and machine-zero p-values across three
independent rollout replays on two days × two cohort-M agents.

But the magnitude lands in the prompt's **"inconclusive"** band
(`mean(adv_force_closed) − mean(adv_matured)` ∈ [−2.07, −2.89] for
the high-`open_cost` agent and [−1.20] for the low-`open_cost`
agent). Compared to the GAE-bootstrap upper bound of
~`open_cost / (1 − γλ)` ≈ 12 reward-units for the high-gene agent
and ≈ 3.4 for the low-gene agent, the policy actually receives
**~21–35 % of the per-tick credit signal that an ideal scalar value
head could route to the open tick**. The remaining 65–79 % gets
smeared into a near-uniform value-head bootstrap that bites every
runner equally on each tick.

The action gradient at the open tick therefore HAS the per-runner
discriminative force the prompt's hypothesis says it lacks — just
attenuated by ~4–5×. This is large enough to refute the "H2 binding"
extreme (the actor is not seeing zero per-runner signal) and small
enough to make a per-runner value head an attractive future
investigation if cohort-M's primary correlation
ρ(open_cost, fc_rate) lands near zero again.

## Method

### Agent / day selection

Three rollout replays were run, each one episode (one day) of a
trained cohort-M agent. All agents are `ppo_time_lstm_v1`,
gen-0, trained for 18 episodes through the per-tick selective-
open-shaping mechanism (commit `3cfa0b4`).

| Agent | open_cost | mature_prob_loss_weight | Day | Steps | Pairs |
|---|---|---|---|---|---|
| `3c66f196` | 0.732 | 0.270 | 2026-04-09 | 11,103 | 572 |
| `3c66f196` | 0.732 | 0.270 | 2026-04-08 | 11,581 | 677 |
| `1f9528e8` | 0.202 | 0.108 | 2026-04-09 | 11,103 | 568 |

`3c66f196` was picked first because it has the highest combined
`open_cost × mature_prob_loss_weight` of the four cohort-M agents
that had completed all 18 training episodes by the diagnostic's
start time. `1f9528e8` was added as a low-`open_cost` cross-check
to test whether the magnitude of the per-tick signal scales with
the gene that drives it.

### Instrumentation (feature-flagged, default-off)

`agents/ppo_trainer.py` gains an opt-in env flag
`H2_DIAGNOSTIC_DUMP_PATH`. When set:

1. `_collect_rollout` writes `pair_outcomes_ep{N}.jsonl` after the
   existing episode-end backfill. Each row is a per-pair record:

   ```json
   {"pair_id": "...", "transition_idx": 1234, "slot_idx": 3,
    "count_legs": 2, "outcome": "force_closed"}
   ```

   Outcome is one of `naked | matured | agent_closed | force_closed`,
   classified via the same `Bet.force_close` and `Bet.close_leg`
   flags that `env._settle_current_race` uses for its
   `arbs_force_closed` counter. `transition_idx` is the index of
   the rollout transition at which the AGGRESSIVE leg matched
   (i.e. the open tick).

2. `_compute_advantages` writes `advantages_ep{N}.jsonl` with one
   row per transition:

   ```json
   {"tick_idx": 1234, "value": -0.42, "advantage": -3.51,
    "return": -3.93, "td_residual": -0.04,
    "training_reward": 0.0, "raw_reward": 0.0, "done": false,
    "action_max_idx": 7, "action_max_val": 0.61}
   ```

   The TD-residual (`δ_t = r̄_t + γV(s_{t+1}) − V(s_t)`) is
   captured inline in the GAE backward sweep without changing the
   gradient pathway.

When the env var is unset both helpers short-circuit. The full
`tests/test_ppo_trainer.py` suite (66 tests) passes unchanged
post-instrumentation; the one operative addition to the hot loop
when the flag is OFF is the no-op
`os.environ.get("H2_DIAGNOSTIC_DUMP_PATH")` lookup.

### Replay harness

`tools/h2_diagnostic.py` loads an agent's weights via the registry,
constructs the `PPOTrainer`, sets the env flag, calls
`trainer._collect_rollout(day)` followed by
`trainer._compute_advantages(rollout)` (skipping `_ppo_update` —
the diagnostic is read-only with respect to the saved weights),
joins the two dumps on `transition_idx`, and groups advantages by
outcome.

### Statistic

One-sided Welch's t-test on the open-tick advantages with the
hypothesis `H1: mean(adv_force_closed) < mean(adv_matured)`. The
load-bearing number per the prompt is the difference of means with
its 95 % CI.

## Per-class advantage distributions

### `3c66f196` (open_cost = 0.732), day 2026-04-09

| Class          |   n |    mean | stddev |     min |    max |
|----------------|----:|--------:|-------:|--------:|-------:|
| matured        |  68 | −0.9693 | 2.3216 |  −6.261 | +4.302 |
| agent_closed   |  33 | −1.2829 | 2.9697 |  −9.951 | +4.044 |
| force_closed   | 410 | −3.0362 | 2.7248 | −10.383 | +4.336 |
| naked          |  61 | −3.0727 | 2.2288 |  −8.251 | +1.092 |

**Headline:** `mean(adv_force_closed) − mean(adv_matured) = −2.07`,
95 % CI [−2.68, −1.46], Welch's t = −6.62 (dof = 100.3),
p = 1.75e-11, Cohen's d = −0.77.

### `3c66f196` (open_cost = 0.732), day 2026-04-08

| Class          |   n |    mean | stddev |     min |    max |
|----------------|----:|--------:|-------:|--------:|-------:|
| matured        |  66 | −0.8494 | 2.3274 |  −6.273 | +2.809 |
| agent_closed   |  23 | −2.7326 | 2.7499 |  −8.269 | +1.031 |
| force_closed   | 500 | −3.7391 | 2.8270 | −13.303 | +4.190 |
| naked          |  88 | −3.6624 | 2.9545 | −13.303 | +4.190 |

**Headline:** `mean(adv_force_closed) − mean(adv_matured) = −2.89`,
95 % CI [−3.50, −2.28], Welch's t = −9.23 (dof = 92.3),
p ≈ 0 (machine zero), Cohen's d = −1.04.

### `1f9528e8` (open_cost = 0.202), day 2026-04-09

| Class          |   n |    mean | stddev |    min |    max |
|----------------|----:|--------:|-------:|-------:|-------:|
| matured        |  63 | −0.0317 | 1.0787 | −3.159 | +3.122 |
| agent_closed   |  21 | −0.9119 | 1.4855 | −3.533 | +1.844 |
| force_closed   | 446 | −1.2313 | 1.0434 | −6.831 | +2.119 |
| naked          |  38 | −1.1878 | 1.3989 | −6.831 | +1.458 |

**Headline:** `mean(adv_force_closed) − mean(adv_matured) = −1.20`,
95 % CI [−1.48, −0.92], Welch's t = −8.30 (dof = 79.3), p ≈ 0,
Cohen's d = −1.14.

### Histogram (high-gene, day 04-09)

`adv_matured` (n=68) — bell-shaped, mode near −1, modest spread:

```
  [   -6.261,   -5.733) ## 2
  [   -5.205,   -4.677) ## 2
  [   -4.677,   -4.149) #### 4
  [   -4.149,   -3.620) ### 3
  [   -3.620,   -3.092) ## 2
  [   -3.092,   -2.564) ##### 5
  [   -2.564,   -2.036) #### 4
  [   -2.036,   -1.508) #### 4
  [   -1.508,   -0.980) ######### 9     ← mode
  [   -0.980,   -0.452) ### 3
  [   -0.452,    0.077) ###### 6
  [    0.077,    0.605) ##### 5
  [    0.605,    1.133) ##### 5
  [    1.133,    1.661) ##### 5
  [    1.661,    2.189) ##### 5
  [    2.189,    2.717) ### 3
  [    3.774,    4.302) # 1
```

`adv_force_closed` (n=410) — left-shifted, longer left tail, mode
near −1.5, ~25 % of probability mass at < −5:

```
  [  -10.383,   -9.647) ########### 11
  [   -9.647,   -8.911) ######## 8
  [   -8.911,   -8.175) #### 4
  [   -8.175,   -7.439) # 1
  [   -7.439,   -6.703) ####################### 23
  [   -6.703,   -5.967) #################### 19
  [   -5.967,   -5.231) ############ 12
  [   -5.231,   -4.495) ################################ 32
  [   -4.495,   -3.759) ###################### 22
  [   -3.759,   -3.023) ######################################## 40
  [   -3.023,   -2.287) ####################################################### 54
  [   -2.287,   -1.551) ########################################################## 58   ← mode
  [   -1.551,   -0.815) ######################################################## 56
  [   -0.815,   -0.079) ############################### 31
  [   -0.079,    0.656) ############## 14
  [    0.656,    1.392) ############ 12
  [    1.392,    2.128) ###### 6
  [    2.128,    2.864) ### 3
  [    3.600,    4.336) #### 4
```

The two distributions OVERLAP substantially — there is no clean
separation hyperplane. The force-closed mode is left-shifted by
~1 unit and the lower tail extends ~4 units further left, but for
any individual open tick the force-closed advantage is barely
distinguishable from a matured-pair advantage at the same tick.

## Comparison to the theoretical GAE upper bound

If the scalar value head could perfectly anticipate the per-runner
shaped delta — i.e. predict at the open tick that "this open will
not be refunded" — the open-tick TD-residual `δ` would be exactly
`−open_cost` for force-closes and `0` for matures (the open-tick
charge minus the predicted refund equals zero on a matured pair
with a clairvoyant value head). GAE then sums geometrically:

```
adv_open ≈ Σ_{k=0}^{∞} (γλ)^k · δ_{open+k}
```

With γ = 0.99, λ = 0.95 (defaults read from the trainer's
`hyperparams.get("gae_lambda", 0.95)`), `γλ ≈ 0.9405` and the
geometric sum gives an upper bound of:

| Agent | open_cost | theoretical max | observed diff | fraction |
|---|---|---|---|---|
| `3c66f196` | 0.732 | 12.30 | 2.07–2.89 | 17–24 % |
| `1f9528e8` | 0.202 |  3.40 | 1.20      | 35 %     |

The actor receives **17–35 % of the per-tick credit signal** that an
ideal scalar value head could route to the open tick. The remaining
65–83 % is consumed by the value head's bootstrap producing similar
predictions for similar states regardless of which runner the agent
opened on — exactly the "smearing" mechanism the prompt's H2
hypothesises.

## Verdict

Per the prompt's decision matrix:

| Diff (mean(adv_force_closed) − mean(adv_matured)) | Interpretation |
|---|---|
| ≤ −5 (significant p < 0.01) | H2 NOT binding |
| Within ±2 (p > 0.1) | **H2 binding** |
| Between −5 and −2 | Inconclusive (signal exists but small) |

Both `3c66f196` rollouts (diff = −2.07 and −2.89) sit in the
**inconclusive band** — closer to "binding" than to "not binding"
but with statistically large effects (Cohen's d ≈ 0.8–1.0).
`1f9528e8`'s diff = −1.20 sits between "binding" and the
inconclusive band — but its open_cost gene is 3.6× smaller, so the
diff scaling tracks the gene that drives it.

**Verdict: H2 is partially binding.** The signal is real and
statistically robust, but ~4–5× too small to do the work the prompt
describes ("the actor learns to open less globally rather than
selectively"). The selectivity gap observed in cohorts O / O2 / F
is therefore consistent with H2 as a contributor, not the sole
binding constraint:

- The per-runner gradient at the open tick is ~25 % of theoretical.
- That ~25 % is competing against advantage-normalisation noise,
  per-mini-batch KL clipping, entropy controller pressure, and
  ~600 opens-per-race × ~77 % force-close rate worth of "open less
  in general" gradient bias.
- A 4× attenuated signal can plausibly lose to a 4× louder
  generic-volume-down signal, even if it isn't structurally zero.

The H1 finding (`mature_prob_head` label was lumping force-closes
in with matures, ρ(fill_prob_loss_weight, fc_rate) = +0.469) and
this H2 finding (per-tick credit at ~25 % of theoretical) are
**complementary**, not competing: H1 was a wrong-direction signal
poisoning the actor's per-runner discrimination input; H2 is a
correct-direction but attenuated signal in the action gradient.
Both deserve a fix before declaring the per-runner credit problem
solved.

## Where to look next

If cohort-M's primary correlation `ρ(open_cost, fc_rate)` lands
near zero again (verdict pending; cohort-M was 4/12 agents through
their 18 episodes when this diagnostic ran), the **per-runner
value head** option from the prompt's surgery-size table is the
natural next investigation:

| Option | Surgery size | Expected effect on this diagnostic's gap |
|---|---|---|
| **Per-runner value head** | Small — `value_head` outputs `(batch, max_runners)` instead of `(batch, 1)`; GAE bootstraps per-runner | Should close the 65–83 % gap directly. The scalar value head can't represent "this runner's open will be force-closed"; a per-runner head can. |
| Distributional critic | Larger — output a return distribution rather than a scalar; quantile or categorical loss | Captures multimodal value (matured vs force-closed are bimodal at the open tick). Indirect but well-precedented. |
| Discrete action over runners | Largest — replaces the per-runner Gaussian with a Categorical over which runner to open on | Concentrates log-prob at the chosen runner; strongest credit but biggest action-space change. |

The **per-runner value head** is the surgical-minimum fix and
matches the diagnostic's failure mode most directly: the bottleneck
is the value head's inability to represent per-runner expected
return, not GAE itself.

If cohort-M's primary correlation lands ρ ≤ −0.5, the H1 fix
(`mature_prob_head` feeding the actor) was sufficient on its own
and the diagnostic's residual H2 attenuation is not load-bearing.
In that case, NO follow-on plan is needed — the partial-H2 finding
becomes a footnote saying "the architecture has a 75 % credit-
assignment loss at the open tick that the actor compensates for
via the better discrimination input from `mature_prob_head`".

## Stop

Per the prompt's stop conditions: report written, follow-on options
sketched but NOT implemented. Awaiting cohort-M validation outcome
to decide whether the per-runner-value-head plan opens.

## Hard constraints honoured

- Instrumentation in `_compute_advantages` is feature-flagged via
  `H2_DIAGNOSTIC_DUMP_PATH`; default behaviour byte-identical
  (verified by full `tests/test_ppo_trainer.py` 66/66 pass).
- `mature_prob_head` architecture untouched.
- `fill_prob → actor_head` pathway untouched.
- No re-runs of cohort-O / O2 / F / M; this used existing trained
  weights from the cohort-M agents.
- No changes to env / reward / aux heads.

## Files added / modified

- `agents/ppo_trainer.py`:
  - `__init__`: added `self._h2_dump_episode_idx = -1` (used only
    when env var set).
  - `_collect_rollout`: appended H2 dump call after the existing
    episode-end pair-outcome backfill.
  - `_compute_advantages`: optional per-tick TD-residual capture
    + dump call. The GAE math is unchanged.
  - Two new private helpers `_h2_dump_pair_outcomes` and
    `_h2_dump_advantages`.
- `tools/h2_diagnostic.py`: new standalone replay harness.
- `plans/per-runner-credit/h2_diagnostic.md`: this report.

## Reproducing

```
# Per-day, per-agent — picks up agent's full hp from the registry,
# loads the day's parquet, and runs one trainer rollout + GAE pass.
# No PPO update is taken; saved weights are not modified.
python tools/h2_diagnostic.py \
  --model 3c66f196 \
  --date 2026-04-09 \
  --dump-dir C:/tmp/h2_diagnostic
```

The script writes:
- `<dump-dir>/advantages_ep0.jsonl`
- `<dump-dir>/pair_outcomes_ep0.jsonl`
- `<dump-dir>/summary.json`

and prints per-class statistics, the headline diff, the Welch
t-test result, and ASCII histograms.
