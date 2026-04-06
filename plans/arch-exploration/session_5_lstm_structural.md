# Session 5 — LSTM structural knobs

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 5)
- `plans/arch-exploration/testing.md`
- `plans/arch-exploration/progress.md` — confirm Sessions 1-4 are done.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md`
- `agents/policy_network.py` — the file you will touch. Read
  `PPOLSTMPolicy` and `PPOTimeLSTMPolicy` in full.

## Goal

Promote three LSTM structural parameters to mutable genes:

- `lstm_num_layers` ∈ {1, 2}
- `lstm_dropout` ∈ [0, 0.3]  (float)
- `lstm_layer_norm` ∈ {false, true}

Thread them through both `PPOLSTMPolicy` and `PPOTimeLSTMPolicy`.

## Scope

**In scope:**
- Add genes to `config.yaml` search_ranges.
- Extend `PPOLSTMPolicy.__init__` to read the three new values from
  the hyperparams dict (with defaults) and apply them to the LSTM
  module. For `num_layers=2`, use the stock `nn.LSTM(num_layers=2,
  dropout=lstm_dropout)` — note that PyTorch's `nn.LSTM` only applies
  dropout between layers if `num_layers > 1`. Document this.
- Layer norm: wrap the LSTM output (or inputs) in a `nn.LayerNorm`
  if enabled. Choose one location and document it.
- `PPOTimeLSTMPolicy` uses a custom `TimeLSTMCell`, not `nn.LSTM`.
  Extend the cell to support stacked layers (a list/ModuleList of
  cells) and optional layer norm. This is the trickier half of the
  session — step through it carefully. Dropout between layers can be
  a simple `F.dropout` applied to intermediate hidden states during
  training.
- Register policy training a Policy factory / `create_policy` call to
  pass the new keys through. If the factory currently uses
  `**hyperparams`, it may Just Work; verify.
- Ensure checkpoint loading still works: older checkpoints without
  these keys should default to the existing single-layer no-dropout
  behaviour.

**Out of scope:**
- Transformer architecture. Session 6.
- Any change to the encoder MLPs, pooling, or actor/critic heads.
- Tuning default values.

## Tests to add

Create `tests/arch_exploration/test_lstm_structural.py`:

1. **Gene sampling.** All three genes present in sampled dict,
   values in range.

2. **Policy instantiation grid.** For every combination of
   `{num_layers: [1, 2]} × {dropout: [0.0, 0.2]} × {layer_norm:
   [False, True]} × {architecture: ["ppo_lstm_v1",
   "ppo_time_lstm_v1"]}`, instantiate the policy on CPU and call
   `forward()` once with a zero tensor of the correct observation
   shape. Assert output shapes (action logits, value) match the
   existing policy's output shapes. 16 combinations × 2 arches = 32
   cases. Should run in <5 seconds.

3. **Hidden-state init across num_layers.** Call `init_hidden()` on a
   `num_layers=2` policy and assert the hidden state has the right
   shape for stacked layers.

4. **TimeLSTMCell stacking.** Unit test the stacked `TimeLSTMCell`:
   feed two timesteps, assert the output shape, assert dropout is
   disabled during `.eval()` and active during `.train()`.

5. **Backward compat.** Load a policy with hyperparams missing all
   three new keys — must default to `num_layers=1`, `dropout=0.0`,
   `layer_norm=False` and behave identically to the pre-session-5
   policy. (Freeze a small golden forward output from the main
   branch and compare? — optional; at minimum assert the policy
   instantiates without error.)

All CPU-only. No GPU.

## Session exit criteria

- All tests pass.
- `progress.md` Session 5 entry.
- `lessons_learnt.md` — expect something to be surprising about
  stacked `TimeLSTMCell`; record it.
- `ui_additions.md` Session 5 items already listed; tick off any
  server-side work.
- Commit.

## Do not

- Do not change the `W_dt` time-modulation mechanism in
  `TimeLSTMCell`. That's the whole point of `ppo_time_lstm_v1`.
- Do not switch from `nn.LSTM` to a custom cell in `PPOLSTMPolicy`.
  Keep the v1 architecture using the stock PyTorch module.
- Do not add num_layers > 2 to the search space. The sequence of
  1338-dim observations is short enough that deep LSTMs are mostly
  a regularization disaster. Revisit only if Session 9 results show
  a compelling reason.
