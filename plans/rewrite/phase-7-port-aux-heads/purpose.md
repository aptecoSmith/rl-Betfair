---
plan: rewrite/phase-7-port-aux-heads
status: design-locked
opened: 2026-05-04
depends_on: rewrite/phase-3-cohort,
            rewrite/phase-5-restore-genes
---

# Phase 7 — port-aux-heads: bring `fill_prob_head` + `mature_prob_head` + `risk_head` into v2

## Purpose

The legacy stack (`agents/policy_network.py` + `agents/ppo_trainer.py
::PPOTrainer`) carries three auxiliary per-runner heads that share
the LSTM backbone with policy + value:

- `fill_prob_head` — BCE-trained forecast of "will this pair's second
  leg fill", per CLAUDE.md §"fill_prob feeds actor_head". Feeds
  actor_head as a per-runner column.
- `mature_prob_head` — BCE-trained forecast with a STRICT label that
  excludes force-closes from the positive class, per CLAUDE.md §
  "mature_prob_head feeds actor_head". Feeds actor_head as a
  per-runner column.
- `risk_head` — Gaussian-NLL-trained per-runner forecast of locked-
  P&L outcome `(mean, log_var)`. Does NOT feed actor_head; surfaces
  on `PolicyOutput.predicted_locked_pnl_per_runner` /
  `predicted_locked_log_var_per_runner` for downstream consumers
  (UI, parquet, NLL diagnostics) and shapes the shared LSTM
  backbone via its NLL gradient.

The first two heads' sigmoid outputs are concatenated into the
actor MLP's input so the policy can act on per-runner discriminative
signals it has learned to extract from the backbone. `risk_head`
influences the actor only indirectly via the backbone-shaping
gradient; its direct role is supervised forecasting on completed-
pair locked outcomes.

The v2 stack (`agents_v2/discrete_policy.py::DiscreteLSTMPolicy` +
`training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer`)
does not implement any of the three heads. `grep -E
'fill_prob|mature_prob|risk_head|aux|bce'` over `training_v2/`
and `agents_v2/` returns zero hits.
`mature_prob_loss_weight`, `fill_prob_loss_weight`, and
`risk_loss_weight` are CohortGenes fields but are never read at
training time — setting them via `--enable-gene` or
`--reward-overrides` is a silent no-op.

This was confirmed empirically on 2026-05-04: a cohort run with
`--reward-overrides mature_prob_loss_weight=0.5` produced eval
results **byte-identical** to the prior cohort that pinned only
`open_cost=1.0`. Same seed, same effective config (the override
reaches no consumer), same per-agent rollout, same eval numbers
to the penny.

The current cohort's selectivity bottleneck — maturation_rate stuck
at 0.19 across 4 GA generations — cannot be attacked via the
mature_prob_head lever until that lever physically exists in the
v2 stack.

## What this phase does

Three deliverables:

1. **Port all three heads into `DiscreteLSTMPolicy`**:
   - `fill_prob_head` and `mature_prob_head` —
     `nn.Linear(hidden, max_runners)`, sigmoid-then-concat into
     actor_input. Architecture-hash break: actor_head input dim
     grows by 2.
   - `risk_head` — `nn.Linear(hidden, max_runners * 2)`, output
     reshaped to `(batch, max_runners, 2)`, log-var clamped to
     `[RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX]` at the forward
     boundary. NOT concatenated into actor_input. Surfaces on
     `PolicyOutput` as `predicted_locked_pnl_per_runner` (mean)
     and `predicted_locked_log_var_per_runner` (clamped log-var).
2. **Wire all three auxiliary loss terms into
   `DiscretePPOTrainer`** — read `fill_prob_loss_weight`,
   `mature_prob_loss_weight`, and `risk_loss_weight` from the
   per-agent `hp` dict using the v2-only precedence (NOT v1's
   `config_fallback` pattern — see §"Trainer reads weights..."
   below). Compute per-step labels at rollout time from
   `BetManager` state:
   - BCE labels for fill_prob and mature_prob per the strict
     contract.
   - Locked-P&L regression labels for risk_head per
     `agents/ppo_trainer.py:1696-1712` (commission +
     win_pnl/lose_pnl from BACK/LAY pairing, completed pairs
     only; naked pairs contribute NaN and are masked out of NLL).
3. **Cohort plumbing parity** — verify the existing
   `--enable-gene {fill,mature,risk}_prob_loss_weight` /
   `--reward-overrides ...` flags actually move the trained-head
   behaviour. The runner's CLI already accepts all three; the
   worker's gene plumbing already lists them in
   `_PHASE5_GENES_VIA_REWARD_OVERRIDES`. After this phase, those
   flags do what their docstrings claim.

## Why this is its own phase

`phase-X-restore-architectures` (the next architecture-port phase)
is currently scoped to bundle the head ports inside Session 01's
TimeLSTM port and Session 02's Transformer port. That bundling is
wrong for two reasons:

1. **Head port + architecture port are independently testable
   units.** Bundling them into the same session means a v1↔v2
   parity failure can be in either the architecture adaptation
   or the head port, and bisection takes longer.
2. **The lever the operator needs first is the head, not the
   architecture.** Maturation_rate stuck at 0.19 is the current
   training failure; LSTM-only is fine for the current verdict
   surface. Phase X can re-use the head pattern this plan ships
   (in `DiscreteLSTMPolicy`) when porting TimeLSTM and Transformer.

After this phase ships:
- Phase X (architectures) keeps its scope but Sessions 01 and 02
  port heads using the established pattern (bigger reuse, smaller
  per-session risk surface).
- The operator gets a usable `mature_prob_loss_weight` lever
  without waiting on the architecture port.

## What's locked

### Heads carry v1's contract bit-for-bit

`fill_prob_head` and `mature_prob_head` in v2 are
`nn.Linear(lstm_hidden, max_runners)` — same shape as v1 (see
`agents/policy_network.py:619` and `:630`). `risk_head` is
`nn.Linear(lstm_hidden, max_runners * 2)` (line `:642`). Forward
pass:

```python
backbone_out = lstm_last  # (batch, hidden)

# Two BCE heads — feed actor_input
fill_logit = self.fill_prob_head(backbone_out)        # (batch, max_runners)
mature_logit = self.mature_prob_head(backbone_out)    # (batch, max_runners)
fill_prob = torch.sigmoid(fill_logit)
mature_prob = torch.sigmoid(mature_logit)
actor_input = torch.cat([
    runner_embs,                                       # (batch, R, embed)
    backbone_out.unsqueeze(1).expand(-1, R, -1),       # (batch, R, hidden)
    fill_prob.unsqueeze(-1),                           # (batch, R, 1)
    mature_prob.unsqueeze(-1),                         # (batch, R, 1)
], dim=-1)
actor_logits = self.actor_head(actor_input)           # (batch, R, n_actions)

# Risk head — surfaces on PolicyOutput, NOT in actor_input
risk_out = self.risk_head(backbone_out)               # (batch, max_runners * 2)
risk_out = risk_out.view(batch, R, 2)
risk_mean = risk_out[..., 0]
risk_log_var = risk_out[..., 1].clamp(
    RISK_LOG_VAR_MIN, RISK_LOG_VAR_MAX,
)
```

`actor_head[0].weight.shape[1] == runner_embed + lstm_hidden + 2`
(unchanged by `risk_head`'s presence — risk does not feed actor).
This matches CLAUDE.md §"mature_prob_head feeds actor_head".

`risk_mean` and `risk_log_var` are emitted on `PolicyOutput`. The
clamp on `log_var` is at the forward boundary so downstream
consumers (UI, parquet, NLL computation) can trust the bounds
without knowing the clamp values. `RISK_LOG_VAR_MIN` and
`RISK_LOG_VAR_MAX` constants port from
`agents/policy_network.py` (search for the existing values; do
not invent new ones).

### Gradient flows through all three heads from training losses

`actor_logits` depends on `fill_prob` and `mature_prob`, so
`policy_loss.backward()` produces non-None grads on
`fill_prob_head.weight` and `mature_prob_head.weight`. **Do not
detach.** This is the §10 gradient-through guard in CLAUDE.md
inherited verbatim.

`risk_head` does not feed `actor_logits` — its only training
signal is the Gaussian NLL auxiliary loss (when `risk_loss_weight
> 0`). The NLL gradient flows back through `risk_head` and into
the shared LSTM backbone, shaping the representation that
`actor_head`, `critic_head`, and the BCE heads all read from.
This is the indirect path by which `risk_head` influences action
selection. **Do not detach risk_head from the NLL term either.**

### Auxiliary labels match v1 semantics exactly

For each transition's `BetManager` state:

- `fill_prob` label per pair: `1.0 if matched_legs(pair) >= 2 else
  0.0`. (Conflates naturally-matured + agent-closed + force-closed
  pairs; this is the v1 behaviour, NOT a bug for fill_prob.) BCE
  loss.
- `mature_prob` label per pair: `0.0 if matched_legs(pair) < 2`,
  `0.0 if any leg has force_close=True`, else `1.0`. Strict by
  construction. See CLAUDE.md §"mature_prob_head feeds actor_head"
  for why force-closes go in the negative class. BCE loss.
- `risk` label per pair: `locked = max(0.0, min(win_pnl,
  lose_pnl))` for completed pairs (`matched_legs >= 2`); `NaN`
  for naked pairs (masked out of NLL — naked pairs have no
  realised locked outcome to supervise against). Inline the same
  arithmetic `BetManager.get_paired_positions` uses (see
  `agents/ppo_trainer.py:1696-1712` for the canonical port
  source: pick the highest-priced BACK and lowest-priced LAY in
  the pair, apply the 0.05 commission). Gaussian NLL loss
  against `(risk_mean, risk_log_var)`.

Per-runner labels are aggregated from per-pair labels by the
runner the pair's open leg fired on. Runners with no open pair
in the rollout window contribute zero to the BCE terms and NaN
to the NLL term (mask out at loss time).

### Trainer reads weights from hp dict only — NO config fallback

```python
self.mature_prob_loss_weight = float(
    hp.get("mature_prob_loss_weight", 0.0) or 0.0
)
```

Same shape for `fill_prob_loss_weight`. **Do NOT add a
`config["reward"][...]` fallback inside the trainer.** That is
the v1 pattern (`agents/ppo_trainer.py:1090-1098`) and it
silently swallows reward_overrides under v2's gene dict
semantics — see "How v1 precedence breaks in v2" below.

The worker is responsible for ensuring the hp dict carries
the right value before the trainer is constructed. When the
operator pins a knob via `--reward-overrides
mature_prob_loss_weight=0.5`, the worker pre-merges that
override INTO the per-agent hp dict (Session 02 calls this
"Path A"). Trainer-side code reads from one source only.

Operator can still pin via `--enable-gene` (per-agent
variation) OR `--reward-overrides` (cohort-wide pin) — but
not both, per the existing mutual-exclusion guard. Both
paths funnel into the hp dict before the trainer sees it.

#### How v1's precedence pattern breaks in v2 (load-bearing)

v1's `hp.get(name, config_fallback)` works because v1's
hp dict is sparse: it only contains keys for genes that are
actually being varied per-agent. When the operator pins a
knob cohort-wide, the key is ABSENT from hp, so the
`hp.get(...)` falls through to the config value. Precedence
behaves intuitively.

v2's hp dict comes from `CohortGenes.to_dict()`, which is a
dataclass that always carries every gene field with its
default value. `hp.get(name, fallback)` returns the gene
default (typically 0.0), and the fallback is never consulted.
The override is silently swallowed.

This is exactly what produced the byte-identical eval results
in cohort `v2_phase5_oc1_mpw05_clean5day_1777849498` on
2026-05-04: the operator pinned `mature_prob_loss_weight=0.5`
via `--reward-overrides`, the worker plumbed it through to a
notional `config["reward"]`, but the trainer's
`hp.get("mature_prob_loss_weight", config_fallback)` read
0.0 from hp before reaching the fallback. Same seed + same
effective trainer state = byte-identical rollouts. The v1 head
work was real; copying its precedence pattern without
understanding why it worked in v1 is how the work got
silently un-shipped in v2.

S02's `test_reward_overrides_reaches_trainer` integration
test is the load-bearing regression guard. The "how" of
the worker-side merge is a Session 02 decision; the "what"
(trainer reads hp only, no config fallback) is locked here.

### Architecture-hash break — pre-plan weights cannot cross-load

Two changes break weight compatibility, but operators only ever
see one error:

1. `actor_head[0].weight.shape[1]` grows by 2 (fill_prob +
   mature_prob columns).
2. New top-level keys appear in the state_dict:
   `fill_prob_head.weight`, `fill_prob_head.bias`,
   `mature_prob_head.weight`, `mature_prob_head.bias`,
   `risk_head.weight`, `risk_head.bias`.

PyTorch's `load_state_dict(..., strict=True)` correctly refuses
pre-plan weights — it raises a "Missing key(s) in state_dict"
error on the new head parameters AND a shape-mismatch on
`actor_head.0.weight`. The variant identity is carried by the
existing weight-shape check in `registry/model_store.py`; no new
explicit version field needed. This is the correct-by-default
behaviour: silent truncation or fresh-init of missing keys would
garble actions or skip head training.

Operators carrying weights forward across this phase MUST start
fresh cohorts. Documented in the phase-7 commit message.

Note: v1 has a `migrate_risk_head` helper at
`agents/policy_network.py:245` for inserting a fresh `risk_head`
into pre-existing weights. This phase does NOT need a
corresponding v2 helper — v2 is starting all three heads fresh
in the same break, no incremental migration path is useful.

### CUDA↔CUDA self-parity holds at fixed seed

Same load-bearing guard as Phase 3 Session 01b. Two CUDA cohort
runs at `--seed 42` and identical args produce bit-identical
per-agent `total_reward` and `value_loss_mean`. The aux-head
gradient is deterministic given the seeded backbone and oracle
labels.

### No env edits

All work in `agents_v2/discrete_policy.py` and
`training_v2/discrete_ppo/trainer.py`. The env's BetManager
already exposes `Bet.force_close` and the per-pair lifecycle
fields the BCE label computation reads.

### Schema growth, not break

`CohortGenes.fill_prob_loss_weight`,
`CohortGenes.mature_prob_loss_weight`, and
`CohortGenes.risk_loss_weight` all already exist. No new gene
fields. Existing scoreboard JSONL rows from pre-plan cohorts
stay readable; the three weights show up under `hyperparameters`
once consumed.

### Same `--seed 42` for cross-cohort comparison

Per CLAUDE.md and Phase 3 convention.

## Success bar

The plan ships GREEN iff:

1. **All three heads exist in `DiscreteLSTMPolicy`** with v1-shape
   `nn.Linear` modules. fill_prob and mature_prob feed
   actor_input via sigmoid-then-concat; risk_head surfaces on
   `PolicyOutput` with the log-var clamp at the forward
   boundary.
2. **Forward-pass parity** — given identical seeded weights and
   identical observation tensors, `DiscreteLSTMPolicy` and the
   v1 `PPOLSTMPolicy` produce action logits AND
   `predicted_locked_pnl_per_runner` /
   `predicted_locked_log_var_per_runner` within fp32 epsilon on
   a fixed test fixture.
3. **`DiscretePPOTrainer` consumes all three loss-weight knobs**
   via `--enable-gene` and `--reward-overrides`, using the
   v2-only hp-dict precedence (NOT v1's config_fallback —
   §"Trainer reads weights..."). Setting any knob to 0.5
   cohort-wide produces non-zero loss for that head in the
   per-update log line.
4. **Strict mature_prob label** is computed correctly: pairs with
   any `force_close=True` leg land in the negative class. New
   test asserts this on a fixture with three pair outcomes
   (matured, agent-closed, force-closed).
5. **Risk label semantics correct** — completed pairs supervise
   against `max(0, min(win_pnl, lose_pnl))` with 0.05 commission;
   naked pairs are NaN-masked out of the NLL term. New test
   asserts on a fixture with one completed and one naked pair.
6. **Architecture-hash break works** — pre-plan v2 weights raise
   on `load_state_dict(..., strict=True)` against a post-plan
   policy. The error must mention BOTH the actor_head shape
   mismatch AND the missing head keys. New test guards this.
7. **Validation cohort moves the BCE levers** — a cohort run
   with `mature_prob_loss_weight=0.5` cohort-wide pinned,
   against a reference run at 0.0, shows ≥ 1 of the following:
   - mean `mature_rate` differs by ≥ 2 percentage points
     (positive direction preferred but EITHER direction is
     evidence the lever is alive — this isn't a "does it
     improve" gate, it's a "does it do anything" gate).
   - per-agent action distribution shifts measurably (KL
     between the two cohorts' rollout action distributions
     ≥ 0.1).
   The validation gate is "the lever is no longer a no-op",
   not "maturation is fixed". Whether higher
   mature_prob_loss_weight helps or hurts is a separate
   question for a follow-on probe.
8. **Risk-head loss-term liveness check** — a cohort run with
   `risk_loss_weight=0.5` produces non-zero NLL contributions
   to total loss in the per-update log lines. This is a weaker
   gate than item 7 (we don't expect risk-head training in 5
   episodes to move per-agent eval metrics measurably — the
   signal it shapes the backbone with is small per-step), but
   it confirms the wiring is alive end-to-end.
9. **CUDA↔CUDA self-parity per architecture** at fixed seed
   continues to hold (existing Phase 3 guard).

## Sessions

### Session 01 — port `fill_prob_head` + `mature_prob_head` into `DiscreteLSTMPolicy`

Pure forward-path work. No trainer changes, no labels, no BCE
loss yet — just the modules + the actor-input concat. With
loss weights at 0 (default), the heads' sigmoid outputs sit at
~0.5 (random init); the actor sees a near-constant column (benign
per CLAUDE.md). Verifies the forward path lands cleanly without
needing the trainer.

Tests in `tests/test_v2_aux_heads.py` (new file):

1. `test_actor_input_dim_includes_aux_columns` — asserts
   `actor_head[0].weight.shape[1] == runner_embed + hidden + 2`.
2. `test_action_logits_depend_on_fill_prob_head_weights` —
   perturbing `fill_prob_head.weight` changes action logits for
   fixed obs / hidden_state. Forward-side gradient-through guard.
3. `test_action_logits_depend_on_mature_prob_head_weights` —
   same for mature_prob_head.
4. `test_pre_plan_weights_fail_to_load` — old state_dict (one
   smaller actor_head input dim) raises on
   `load_state_dict(..., strict=True)`.
5. `test_v1_v2_forward_parity_at_fixed_weights` — copy v1
   `PPOLSTMPolicy` weights into v2 `DiscreteLSTMPolicy`,
   identical obs in, action logits match within fp32 epsilon.
   Load-bearing parity test inherited from Phase 1 Session 02
   convention.

Session prompt: `session_prompts/01_port_heads_to_discrete_policy.md`.

### Session 02 — wire BCE auxiliary loss into `DiscretePPOTrainer`

Trainer-side work:

1. Add label-computation helpers that read per-pair lifecycle
   state from the rollout buffer's BetManager snapshots. Strict
   mature_prob label per §"What's locked". Aggregation from
   per-pair labels to per-runner targets matches v1.
2. Add `self.fill_prob_loss_weight` and
   `self.mature_prob_loss_weight` reads with v1 precedence
   (hp dict → config fallback → 0.0).
3. Add the BCE term to the per-update loss:
   `total_loss = policy_loss + value_loss * self.value_coeff
                 - entropy * self.entropy_coeff
                 + self.fill_prob_loss_weight * fill_prob_loss
                 + self.mature_prob_loss_weight * mature_prob_loss`.
4. Per-update log line includes
   `fill_prob_bce_mean`, `mature_prob_bce_mean` so operators can
   verify the term is non-zero when the weight is non-zero.
5. **Resolve the config-vs-hp plumbing decision** noted in
   §"What's locked": cleanest path is the worker pre-merges
   `reward_overrides` into the per-agent hp dict at construction
   time, so the trainer's existing `hp.get(...)` precedence works
   uniformly. Document the decision in the session's lessons-
   learnt.

Tests added to `tests/test_v2_aux_heads.py`:

6. `test_bce_loss_zero_when_weight_zero` — gene/override at 0.0
   means the BCE term contributes nothing to total loss.
7. `test_bce_loss_nonzero_when_weight_nonzero` — gene/override at
   0.5 means the BCE term lands in total loss with the right
   magnitude.
8. `test_strict_mature_label_excludes_force_closes` — fixture
   with three pairs (matured, agent-closed, force-closed) →
   labels [1.0, 1.0, 0.0].
9. `test_loss_weight_precedence_hp_then_config_then_default` —
   covers all three branches.
10. `test_reward_overrides_reaches_trainer` — operator-side
    integration: launch a small cohort with
    `--reward-overrides mature_prob_loss_weight=0.5`, assert the
    trainer's stored weight equals 0.5 for every agent.

Session prompt: `session_prompts/02_wire_bce_loss_in_trainer.md`.

### Session 03 — validation cohort + plan close

Run two small cohorts at `--seed 42`, identical args except for
the head loss weight:

- Reference: `--reward-overrides mature_prob_loss_weight=0.0`
  (default — same as the current pre-plan v2 behaviour).
- Probe: `--reward-overrides mature_prob_loss_weight=0.5`.

Each cohort: 12 agents × 1 generation × 5 days, eval on the
same held-out day as the recent runs (2026-05-02).

Compute per the §"Success bar" item 6:

- Per-agent eval action-distribution KL between the two cohorts.
- Per-agent maturation_rate delta.
- Per-agent total_reward and day_pnl deltas (informational
  only — this is not a "did it help" gate).

Update `findings.md` with:

- Whether the lever is now alive (KL or mature_rate moved).
- Direction of the effect (does higher weight help, hurt, or
  noise out).
- Recommended follow-up plan if higher-weight didn't help (e.g.
  examine label noise, check whether the actor is using the
  column, etc.).

Update CLAUDE.md to add a section under "## Reward function" or a
new top-level section noting that the v2 stack now consumes
`fill_prob_loss_weight` and `mature_prob_loss_weight`.

Session prompt: `session_prompts/03_validation_and_writeup.md`.

## Hard constraints

In addition to all rewrite hard constraints + phase-3 + phase-5
inherited:

1. **No env edits.** All work in `agents_v2/` and
   `training_v2/discrete_ppo/`.
2. **v1↔v2 forward-pass parity** for `DiscreteLSTMPolicy` after
   the head port. The class is an extension, not a rewrite.
3. **Strict mature_prob label** — pairs with any
   `Bet.force_close=True` leg are in the negative class. This is
   the load-bearing semantic difference vs `fill_prob`.
4. **Heads must NOT be detached** — gradient must flow from
   surrogate loss back through both heads. Forward-side AND
   backward-side tests guard this.
5. **Architecture-hash break is intentional.** Pre-plan v2
   weights raise on `load_state_dict(..., strict=True)`. No
   shim, no truncation, no fallback init.
6. **Schema is forward-only.** No new gene fields; reuse the
   existing `fill_prob_loss_weight` and `mature_prob_loss_weight`.
7. **Default behaviour is byte-identical** to pre-plan when both
   weights are 0.0 — the heads run, output ~0.5 columns, and
   the actor's behaviour is unchanged because the columns carry
   no signal.
8. **Same `--seed 42`** for any cross-cohort comparison.
9. **NEW output dirs** for every cohort run.
10. **Validation gate is "lever is alive", not "lever helps".**
    Whether higher mature_prob_loss_weight improves training is
    a separate question. This phase ships when the knob does
    something measurable.

## Out of scope

- TimeLSTM and Transformer architecture ports — those are
  `phase-X-restore-architectures`. This phase establishes the
  head pattern in `DiscreteLSTMPolicy` so Phase X can reuse it
  verbatim in the new architectures.
- Oracle-label BC pretraining for the heads. The CLAUDE.md
  §"BC pretrain (2026-04-19)" describes a v1 mechanism that is
  also absent in v2; that's its own port plan.
- Tuning what value of `mature_prob_loss_weight` is best. This
  phase ships when the lever moves measurably; selecting the
  best value is a follow-on probe.
- 66-agent scale-up.
- v1 deletion.

## Useful pointers

- v1 head definitions:
  [`agents/policy_network.py:619`](../../../agents/policy_network.py)
  (`fill_prob_head` on `PPOLSTMPolicy`),
  [`:630`](../../../agents/policy_network.py)
  (`mature_prob_head`).
- v1 trainer aux-loss term:
  [`agents/ppo_trainer.py:2447`](../../../agents/ppo_trainer.py).
- v1 weight-precedence pattern:
  [`agents/ppo_trainer.py:1090-1098`](../../../agents/ppo_trainer.py).
- v2 single-policy class to extend:
  [`agents_v2/discrete_policy.py`](../../../agents_v2/discrete_policy.py).
- v2 trainer to extend:
  [`training_v2/discrete_ppo/trainer.py`](../../../training_v2/discrete_ppo/trainer.py).
- Worker plumbing for reward_overrides → per-agent dict:
  [`training_v2/cohort/worker.py:245`](../../../training_v2/cohort/worker.py).
- CLAUDE.md §"fill_prob feeds actor_head" and §"mature_prob_head
  feeds actor_head" — the v1 contract these heads carry, ported
  unchanged.
- Empirical no-op evidence (2026-05-04 cohort
  `v2_phase5_oc1_mpw05_clean5day_1777849498` byte-identical to
  prior cohort):
  [`registry/v2_phase5_oc1_mpw05_clean5day_1777849498/`](../../../registry/v2_phase5_oc1_mpw05_clean5day_1777849498/).

## Estimate

- Session 01 (three heads in policy + forward parity tests):
  ~3.5 h (was 3 h pre-risk-head; the additional shape /
  log-var clamp / PolicyOutput plumbing adds ~30 min).
- Session 02 (auxiliary losses in trainer + label gen + tests):
  ~6.5 h (was 5 h pre-risk-head). The risk label generator is
  more involved than the BCE labels — commission, BACK/LAY
  pairing arithmetic, NaN masking — but the canonical
  arithmetic ports verbatim from `agents/ppo_trainer.py:
  1696-1712`. Tests for the third weight knob mirror the
  existing two.
- Session 03 (validation cohort + writeup): ~2 h wall + cohort
  compute time (~30–60 min on a 12-agent × 5-day run).
  Single validation run probes the BCE lever (mature_prob);
  the risk-head liveness check piggybacks on the same run via
  per-update log inspection — no separate cohort needed.

Total: ~12 h human + ~1 h GPU.

If past 8 h on Session 02 excluding tests, stop and check
scope — the label computations should reuse the existing
`BetManager.bets` field directly. A long session means the
rollout buffer doesn't expose what the label aggregator needs;
that's a separate refactor that should be pulled out into its
own session.

## When to do this

Now. The current cohort proves the v2 stack's selectivity
bottleneck (mature_rate stuck at 0.19 across 4 GA generations)
cannot be attacked via the documented head lever because the
lever doesn't exist in v2. Until it does, every cohort run that
sets `mature_prob_loss_weight` is silently a no-op.

Phase X (architectures) can wait — single-architecture LSTM
cohorts are fine for the current verdict surface. Phase 7
(this plan) is the unblocker for the active investigation
into selectivity.
