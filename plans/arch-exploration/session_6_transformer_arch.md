# Session 6 — `ppo_transformer_v1` architecture

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 6)
- `plans/arch-exploration/testing.md`
- `plans/arch-exploration/progress.md` — confirm Sessions 1-5 done.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md`
- `agents/policy_network.py` — study the existing architectures.
- `agents/architecture_registry.py` — confirm the decorator pattern
  still looks the way the design review described.
- Repo root `PLAN.md:299-304` and `TODO.md:476-485` — the original
  intent for this architecture.

## Goal

Build `ppo_transformer_v1`: a third architecture sharing the existing
market / runner encoders and actor / critic heads, but replacing the
LSTM sequence model with a small transformer encoder over a bounded
tick context window.

## Scope

**In scope:**
- New class `PPOTransformerPolicy` in `agents/policy_network.py` (or
  a new file, whichever keeps diffs small). Class attribute
  `architecture_name = "ppo_transformer_v1"`. Registered via the
  `@register_architecture` decorator.
- Structure:
  - **Same** market encoder (market + velocity + agent_state → MLP).
  - **Same** per-runner shared MLP encoder + mean/max pool.
  - **New:** a small transformer encoder operating over the last
    `transformer_ctx_ticks` timesteps' fused embeddings. Use
    `nn.TransformerEncoder` with `batch_first=True` and causal
    masking (an agent must not peek at future ticks).
  - **Same** actor head shape (per-runner) and critic head.
- Three new genes in `config.yaml`:
  - `transformer_heads` ∈ {2, 4, 8}
  - `transformer_depth` ∈ {1, 2, 3}
  - `transformer_ctx_ticks` ∈ {32, 64, 128}
- Positional encoding: use the standard additive sinusoidal
  encoding, or a learned positional embedding (your choice — pick
  one and document it).
- Tick-context buffer: because the environment is a per-tick step
  loop, the policy must maintain a rolling buffer of the last
  `transformer_ctx_ticks` fused embeddings as "hidden state". This
  is the transformer equivalent of `init_hidden` + LSTM state.
  Extend the `BasePolicy` hidden-state protocol to accommodate
  either an LSTM tuple OR a rolling buffer tensor.
- **Arch cooldown** (population_manager): an agent whose architecture
  was mutated in generation N cannot have its architecture mutated
  again in generation N+1. Add a `arch_change_cooldown` field to the
  agent metadata. Document the mechanism.
- **Arch-specific LR range** in the planner from Session 4: allow a
  `TrainingPlan` to override `learning_rate` range per architecture.
  Transformer agents typically want a different (often lower) LR
  distribution. If no override is provided, fall back to the global
  range.

**Out of scope:**
- LR warmup, weight decay, any optimiser change. The existing
  Adam/AdamW and single-LR setup stays.
- Attention over runners within a tick (that would be a different
  architecture — a hierarchical one, mentioned in the design review
  but not in this session's scope).
- Deleting either LSTM architecture. Keep all three live.

## Tests to add

Create `tests/arch_exploration/test_transformer_arch.py`:

1. **Registry.** `get_architecture("ppo_transformer_v1")` returns the
   class, and `list_architectures()` (or equivalent) includes it.

2. **Genes sampled.** All three transformer genes present and in
   range.

3. **Instantiation grid.** For every combination of
   `{heads: [2, 4]} × {depth: [1, 2]} × {ctx_ticks: [32, 64]}`,
   instantiate the policy on CPU and call `forward()` twice in
   sequence (to exercise the rolling buffer). Assert output shapes.
   8 combinations.

4. **Causal masking.** Construct two input sequences that are
   identical up to timestep T but differ at T+1. Call `forward` on
   both at timestep T; assert the policy outputs at timestep T are
   identical. This catches leakage from future ticks.

5. **Rolling-buffer overflow.** Feed `ctx_ticks + 5` timesteps in
   sequence; assert no crash and that the buffer retains the most
   recent `ctx_ticks` entries (test by feeding known fingerprints and
   checking which are retained).

6. **Arch cooldown.** Construct a population with an agent whose
   `arch_change_cooldown > 0` and run `mutate`. Assert that agent's
   `architecture_name` did not change. Reset cooldown, mutate again,
   assert change is now possible.

7. **Planner arch-specific LR.** Create a `TrainingPlan` with a
   per-arch LR override. Build Gen-0 under that plan. Assert
   transformer agents' sampled `learning_rate` values come from the
   override range and LSTM agents' values come from the global
   range.

All CPU. No GPU. No training.

## Session exit criteria

- All tests pass.
- `progress.md` Session 6 entry.
- `lessons_learnt.md` — expect meaningful notes here; transformer
  integration always has wrinkles.
- `ui_additions.md` — confirm Session 6 items are present and
  complete.
- Commit.

## Do not

- Do not start with a huge transformer. `heads=2, depth=1,
  ctx_ticks=32` is a valid starting point and trains in reasonable
  time. Bigger variants exist so the genetic search can discover
  they help; they are not the default.
- Do not skip causal masking. A non-causal transformer leaks
  information from future ticks and makes the training signal
  unusable.
- Do not conflate `observation_window_ticks` with
  `transformer_ctx_ticks`. `observation_window_ticks` was retired
  cleanly in Session 1. Session 6 introduces `transformer_ctx_ticks`
  as a fresh gene. No silent aliasing.
- Do not change the market or runner encoders. Those are shared
  across all three architectures and any change belongs in a
  dedicated session.
