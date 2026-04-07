# Session 16 — `ppo_hierarchical_v1` architecture

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md`
- `../master_todo.md` (Session 16)
- `../progress.md`
- `../lessons_learnt.md`
- `../../arch-exploration/session_6_transformer_arch.md` — reuse
  the registration pattern, causal masking, and rolling-buffer
  ideas from the first transformer
- `agents/policy_network.py` — look at how the current architectures
  pool runners (mean + max), this session replaces that pooling
- `../ui_additions.md`
- `../initial_testing.md`

## Goal

Build `ppo_hierarchical_v1`: an architecture that replaces the
existing mean/max pooling of per-runner embeddings with a
**per-tick transformer attending across runners**. A second-level
sequence model (LSTM or transformer — choose one) then runs over
the tick-level outputs. The hierarchy is "runners pooled via
attention → ticks processed recurrently".

Motivation: the current architectures treat the set of runners in
a race as bag-of-embeddings via mean/max pooling, which loses
information about relationships between runners (favourite vs
longshot, form spread, etc.). A transformer attending across
runners *within a tick* should let the model learn those
relationships.

## Scope decision required at session start

The hierarchy is fixed at "attention over runners, then sequence
over ticks". But the outer sequence model can be:

- **(A)** LSTM — cheaper, matches existing `ppo_lstm_v1` temporal
  dynamics, fewer moving parts
- **(B)** Transformer — matches `ppo_transformer_v1`, more
  expressive, more parameters
- **(C)** Time-LSTM — matches `ppo_time_lstm_v1`, includes the
  time-delta forget gate

**Pick one before writing any code.** Document the choice and the
tradeoff in `design_decisions.md`. Recommendation: **(A) LSTM** on
the grounds that (1) we already have two transformer and two LSTM
architectures, so the LSTM outer gives us a novel runner-attention
signal *without* a confound from a second transformer, (2) parameter
count stays sane, (3) training dynamics are better understood.

## Scope

**In scope:**

1. **New class `PPOHierarchicalPolicy`** in
   `agents/policy_network.py` (or a new file). Class attribute
   `architecture_name = "ppo_hierarchical_v1"`. Registered via
   `@register_architecture`.

2. **Structure:**
   - **Same** market encoder and per-runner shared MLP encoder.
   - **New:** a per-tick multi-head self-attention block over the
     runner-embedding set. Output: one "attended pool" vector per
     tick, possibly concatenated with a query-style global summary
     token (choose one; document in design decision).
   - **New:** concatenation of market embedding + attended runner
     pool, fed into the chosen outer sequence model (LSTM per
     above recommendation).
   - **Same** actor head (per-runner, using the pre-attention
     runner embeddings so each runner sees its own features
     directly — do not force the actor to go through the pool)
     and critic head (global).

3. **Genes:**
   - `hier_runner_attn_heads` ∈ {2, 4} — attention heads for the
     per-tick runner-attention block
   - `hier_runner_attn_depth` ∈ {1, 2} — number of attention
     layers
   - Reuse existing `lstm_hidden_size` / `lstm_num_layers` /
     `lstm_dropout` / `lstm_layer_norm` for the outer sequence
     model (matches option A above — if you pick B or C, use the
     transformer genes instead)
   - All gated via the structural-gene scoping from Session 14 so
     these genes only appear for `ppo_hierarchical_v1` agents

4. **Positional encoding is not required inside runner
   attention.** Runners are an unordered set (the selection_id
   ordering is arbitrary). No positional embedding on the
   runner axis. Document this explicitly in the policy's
   docstring.

5. **Masking.** Per-race runner counts vary (up to
   `max_runners=14`). The runner-attention block must mask out
   padded-runner slots so attention doesn't look at absent
   runners. Use the existing runner-mask from the observation
   pipeline — do NOT invent a new mask source.

**Out of scope:**

- A transformer outer sequence model. That's option B; if Session
  11 results later justify it, it becomes its own follow-up.
- Changing the market encoder or the shared runner encoder MLP.
- Any optimiser tweaks specific to this architecture.
- Deleting the existing mean/max pool from other architectures.

## Tests to add

Create `tests/next_steps/test_hierarchical_arch.py`:

1. **Registry.** `get_architecture("ppo_hierarchical_v1")` returns
   the class.

2. **Instantiation grid.** For every combination of `{attn_heads:
   [2, 4]} × {attn_depth: [1, 2]} × {lstm_hidden_size: [128, 256]}
   × {num_layers: [1, 2]}`, instantiate and run `forward()` twice
   in sequence. 16 combinations.

3. **Runner masking.** Construct two inputs that are identical
   on all valid runner slots but differ in the padded slots.
   Assert the policy output is identical. This catches leakage
   from padding.

4. **Runner-order invariance.** Permute the runner ordering in a
   batch; assert the per-runner actor outputs are permuted the
   same way (up to the bypass that lets the actor see the
   pre-attention embedding directly — which is per-runner and
   therefore also permutation-equivariant). Do not assert exact
   bitwise equality; use a tolerance.

5. **Structural gene scoping.** A `ppo_hierarchical_v1` agent has
   `hier_runner_attn_*` genes; a `ppo_lstm_v1` agent does not.
   This test lives alongside the Session 14 scoping tests.

6. **Causal masking on the outer LSTM.** An LSTM is causal by
   definition, but the runner-attention block within a tick is
   not — confirm that the per-tick attention does not accidentally
   introduce a path from tick T+1 to tick T.

7. **Architecture mutation.** A population can mutate into and
   out of `ppo_hierarchical_v1` respecting the cooldown.

All CPU, all fast.

## Manual tests

- **M1 (UI smoke)** — new architecture in the choice widget, new
  genes in the plan editor (gated by architecture selection from
  Session 14).

## Session exit criteria

- Outer sequence model choice committed in `design_decisions.md`
  **before** any code.
- All 7 tests pass.
- Architecture registered and mutable; cooldown respected.
- `ui_additions.md` Session 16 entries added.
- `lessons_learnt.md` updated — expect wrinkles around
  runner-mask reuse and permutation invariance tests.
- `master_todo.md` Session 16 ticked.
- Commit.

## Do not

- Do not start without committing the outer-sequence-model
  choice. This is the single biggest decision in the session.
- Do not add a positional embedding on the runner axis. Runners
  are a set.
- Do not invent a new runner mask. Reuse the one the observation
  pipeline already produces.
- Do not gate this session on Session 15 (the feedforward
  baseline). They are independent; order them based on whichever
  Session 11 signal is stronger.
- Do not run a GPU benchmark. Leave it to a future integration
  session.
