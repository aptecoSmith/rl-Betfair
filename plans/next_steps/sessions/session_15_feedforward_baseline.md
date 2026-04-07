# Session 15 — `ppo_feedforward_v1` baseline architecture

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md`
- `../master_todo.md` (Session 15)
- `../progress.md`
- `../lessons_learnt.md`
- `../../arch-exploration/session_6_transformer_arch.md` — the
  model for registering a new architecture via the existing
  decorator pattern
- `agents/architecture_registry.py`
- `agents/policy_network.py` — study the shared encoder blocks
- `../ui_additions.md`
- `../initial_testing.md`

## Goal

Build `ppo_feedforward_v1`: a **non-recurrent** baseline
architecture that shares the existing market and runner encoders
but processes each tick's fused embedding in isolation. Its job is
to answer the question *"do we actually need recurrence for this
problem?"*

A baseline that performs near the recurrent architectures is a
useful negative result — it means the recurrence is mostly dead
weight. A baseline that underperforms clearly is also useful — it
gives the recurrent architectures a floor to justify themselves
against.

## Scope

**In scope:**

1. **New class `PPOFeedforwardPolicy`** in `agents/policy_network.py`
   (or a new file). Class attribute `architecture_name =
   "ppo_feedforward_v1"`. Registered via `@register_architecture`.

2. **Structure:**
   - **Same** market encoder (market + velocity + agent_state MLP).
   - **Same** per-runner shared MLP encoder + mean/max pool.
   - **New:** a deeper MLP head fusing market + pooled-runner
     embedding, instead of an LSTM/transformer sequence model.
     Use `mlp_layers` + `mlp_hidden_size` for depth/width, matching
     the existing structural gene naming.
   - **Same** actor and critic heads.
   - `init_hidden()` returns a no-op placeholder so the training
     loop's hidden-state plumbing doesn't need a special case.

3. **No new genes.** Reuse `mlp_hidden_size`, `mlp_layers`, and
   general PPO genes. Do NOT invent `ff_hidden_size` or similar.
   The point of this session is the minimal baseline; extra
   configurability defeats the purpose.

4. **Architecture registry** — add `"ppo_feedforward_v1"` to the
   `architecture_name` str_choice in `config.yaml`. The new arch
   must be mutable into/out of via the existing cooldown logic
   (from Session 6).

5. **No training run.** This session produces the architecture,
   tests, and UI wiring. Actual comparison vs LSTMs/transformer
   happens in a future multi-gen run, not here.

**Out of scope:**

- A hierarchical / transformer-over-runners variant. That is
  Session 16.
- New encoder MLPs, new pooling strategies, or any change that
  would also affect the existing three architectures.
- Any optimiser tweaks specific to the feedforward baseline.

## Tests to add

Create `tests/next_steps/test_feedforward_arch.py`:

1. **Registry.** `get_architecture("ppo_feedforward_v1")` returns
   the class.

2. **Instantiation grid.** For every combination of
   `{mlp_hidden_size: [64, 128]} × {mlp_layers: [1, 3]}`,
   instantiate the policy on CPU and call `forward()` twice in
   sequence. Assert output shapes match the existing policies'
   output shapes.

3. **No hidden-state bleed.** Call `forward()` on two sequences
   that differ at timestep T. Assert the outputs at timestep T
   are independent of any prior timestep. This is the
   "feedforward really is feedforward" check — if any hidden
   state is accidentally carried across calls, this catches it.

4. **Architecture mutation.** Build a population that can mutate
   between all four architectures. Assert mutation respects the
   cooldown from Session 6 and correctly scopes structural genes
   (`lstm_*` and `transformer_*` disappear, `mlp_*` stays).

5. **Parameter count is smaller than LSTM v1.** Sanity check — a
   feedforward baseline should have strictly fewer parameters
   than an LSTM of equivalent `mlp_hidden_size`. If it doesn't,
   something is wrong with the MLP head construction.

All CPU, all fast.

## Manual tests

- **M1 (UI smoke)** — architecture choice widget now includes
  `ppo_feedforward_v1`.

## Session exit criteria

- All tests pass.
- Architecture registered and mutable.
- Backward compatibility: existing plans that hardcode
  `architectures: [ppo_lstm_v1, ppo_time_lstm_v1,
  ppo_transformer_v1]` continue to work.
- `ui_additions.md` Session 15 entries added.
- `lessons_learnt.md` updated.
- `master_todo.md` Session 15 ticked.
- Commit.

## Do not

- Do not accidentally leak recurrence. A persistent tensor on the
  module (e.g. a running-stats buffer) counts as a hidden state
  and makes the baseline dishonest.
- Do not add attention-over-runners here. That is Session 16.
- Do not reuse `lstm_hidden_size` or `transformer_*` genes for
  this architecture. The whole point is that it's simpler.
- Do not run a GPU benchmark in this session to prove the
  baseline is fast. That's a Session 11-style integration run.
