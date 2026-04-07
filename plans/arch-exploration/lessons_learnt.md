# Lessons Learnt — Architecture & Hyperparameter Exploration

Anything surprising, counter-intuitive, or that would have saved time
if we'd known it earlier. Append at the end of every session.

Project-wide conventions (e.g. CLAUDE.md) take precedence over anything
recorded here — this file is for learnings that are too narrow or too
provisional to belong there yet.

---

## 2026-04-06 — Design review findings

- **Sampled ≠ used.** The genetic algorithm can happily mutate values
  that no downstream code ever reads. `reward_precision_bonus` was in
  the schema for weeks, got mutated in every generation, and never
  changed any agent's reward. Rule: **every gene must have a test that
  asserts the env (or trainer) actually uses the sampled value**, not
  just that the value was sampled. A grep for a gene name should turn
  up at least one "read from hp" site AND one "passed to downstream
  consumer" site.

- **Architecture 2 is a tweak, not a replacement.** `ppo_time_lstm_v1`
  differs from `ppo_lstm_v1` only in a learnable `W_dt` parameter on
  the forget gate. Same encoders, same head, same pooling. So any
  perceived gap between the two architectures on Gen 0 is a signal
  about that one parameter, not about fundamentally different model
  families. Keep this in mind when reading results.

- **Gen 0 is already varied, not cloned.** Contrary to my initial
  suspicion, `population_manager.py:220` does call `sample_hyperparams`
  independently per agent. The problem wasn't "Gen 0 is uniform" —
  it was "several of the sampled values never reach the place that
  would use them".

- **Mutation is single-parameter, not all-at-once.** Good news —
  credit assignment across generations is possible.

- **Terminal bonus is raw.** `day_pnl / starting_budget` is added to
  `_cum_raw_reward`, not shaped. This is correct (it's real money) but
  was worth double-checking because the CLAUDE.md invariant
  `raw + shaped ≈ total_reward` breaks silently if you get this wrong.

## 2026-04-06 — Session 1 (reward plumbing)

- **The gene-name → config-key mapping is not 1:1.** The three reward
  genes sound like config keys but only two of them are: the existing
  `early_pick_bonus_min` / `_max` pair doesn't match the single scalar
  gene `reward_early_pick_bonus`. Session 1's fix maps the scalar to
  *both* ends of the interval (min == max → constant multiplier), which
  plumbs the gene through without changing the reward formula. Session 3
  will split it into proper min/max genes. This was the one non-obvious
  decision in an otherwise mechanical session; noting it so that
  reading `reward_overrides={"early_pick_bonus_min": 1.4,
  "early_pick_bonus_max": 1.4}` in a trainer log doesn't look like a bug.

- **Shared-config mutation is an easy trap.** My first draft of
  `BetfairEnv.__init__` merged overrides into `config["reward"]`
  directly. That would have been a horrible sleeper bug the moment two
  agents shared a config object — which is exactly what
  `PPOTrainer(... config=self.config ...)` does. Added an explicit
  regression test (`test_env_overrides_do_not_mutate_shared_config`) so
  nobody can accidentally reintroduce the aliasing.

- **Retiring `observation_window_ticks` was clean — no migration
  needed.** The sampler iterates over whatever specs exist in
  `search_ranges`, so removing the entry from `config.yaml` removes the
  gene from every checkpoint created after the change. Old checkpoints
  still contain the stale key but nothing reads it, so there's no
  crash path. The only code that *asserted* its presence was
  `tests/test_config.py:94` — updated to match.

- **Pre-existing failure `test_obs_dim_matches_env`.** Unrelated to
  this session: `tests/test_population_manager.py:408` hardcodes
  `obs_dim == 1630`, but current value is 1636. Confirmed stale on
  plain `master` via `git stash`. Left alone — fixing drive-by failures
  is exactly the "while I'm here" scope creep Session 1's plan warns
  against. Flag for a future cleanup pass.

## 2026-04-06 — Session 3 (reward schema expansion)

- **Repair the genome at TWO layers, not one.** The Session 3 plan
  said "after sampling or mutation, swap inverted intervals", which
  reads as a single fix in the population manager. But a directly
  constructed `BetfairEnv(reward_overrides={...})` (e.g. from a unit
  test, or from an as-yet-unwritten live-inference call site) would
  bypass the population manager entirely. Putting the swap in BOTH
  places — `population_manager._repair_reward_gene_pairs` AND
  `BetfairEnv.__init__` — is cheap and means there is no path that
  can produce an inverted interval at training time. Tests 3 and 6
  cover the env-level and sampler-level paths respectively.

- **The terminal bonus lives in `step()`, not `_settle_current_race`.**
  The session plan wording said "apply the multiplier inside
  `_settle_current_race` exactly where the existing terminal bonus is
  computed" — but the terminal bonus is actually computed in
  `step()`, in the `if terminated:` branch, *after* the per-race
  settlement returns. `_settle_current_race` only computes the
  per-race shaping/raw split. I applied the multiplier where the code
  actually lives. Worth noting because anyone reading the plan in
  isolation will look in the wrong function.

- **Patching `BetManager.settle_race` is the cheapest way to inject a
  known race P&L into a synthetic env.** The terminal-bonus test
  needed `_day_pnl != 0` to verify the weight is applied to the raw
  bucket. Engineering a real winning bet from `_make_day` synthetic
  data is fiddly (action thresholds, stake fractions, ladder
  filtering). `unittest.mock.patch.object(BetManager, "settle_race",
  return_value=10.0)` short-circuits the entire match-and-settle
  pipeline and lets the test focus purely on the `step()` terminal
  branch. Recording the technique because the same trick will be
  useful for Session 7's drawdown tests.

## 2026-04-06 — Session 4 (training plan / coverage tracker)

- **Coverage history must include discarded models.** The first draft
  of `historical_agents_from_model_store` filtered to `status='active'`
  on the (reasonable-sounding) basis that "discarded models are bad
  examples". That's the wrong framing for a *coverage* metric: the
  question is "did we ever sample this corner of the search space?",
  not "did we like the result?". A discarded ppo_lstm_v1 with
  `gamma=0.998` is still evidence that we explored that bucket. Fixed
  to read every record. The same logic applies to garaged models.

- **`float_log` genes need log-space buckets.** First draft used linear
  deciles for every numeric gene, which made the bottom decile of
  `learning_rate` cover [1e-5, 5e-5] and the top decile cover
  [4.5e-4, 5e-4] — i.e. 99 % of plausible historical samples landed
  in the bottom bucket regardless of how well-explored the range
  actually was. The hand-counted coverage test caught this immediately.
  `_bucket_edges_for` now switches to log-space edges when
  `spec.type == "float_log"`, matching how `sample_hyperparams`
  actually generates values.

- **Out-of-range historical samples are clamped, not dropped.** When a
  range is tightened in `config.yaml` (Session 1 did this for
  `early_pick_bonus_max`), old agents persisted with values outside
  the new bounds. Dropping them silently would understate coverage in
  exactly the buckets that *did* historically have samples. The
  clamp-to-end-bucket behaviour is documented inline in
  `_assign_bucket` so a future tightening doesn't get re-litigated.

- **Path-traversal guard on `plan_id`.** The POST endpoint takes a
  client-supplied payload but `plan_id` is generated server-side via
  `uuid.uuid4()`, so in practice traversal is impossible — *unless* a
  future endpoint accepts a raw plan_id from the client. The
  `_path_for` guard (rejects `..`, `/`, `\`) is cheap belt-and-braces
  so this can't regress later.

- **Bias-sampler integration is opt-in, not auto.** I deliberately did
  *not* call `bias_sampler` from `population_manager.initialise_population`
  even when a plan is supplied. Reason: the session plan says "Keep
  the bias gentle — it should tilt, not override", and silently
  altering Gen 0 sampling distributions whenever a plan exists would
  violate the principle of least surprise for anyone running
  `start_training.sh`. The plan author can apply `bias_sampler` to
  produce a modified `hp_ranges` block before calling `TrainingPlan.new`
  — making the bias visible in the plan file rather than implicit in
  the runtime. Session 8's UI will surface a "Bias toward uncovered"
  toggle that does exactly this.

- **`PopulationManager.initialise_population(plan=...)` is small but
  load-bearing.** The plan-aware branch is only ~15 lines but it
  changes four things at once (pop size, hp specs, arch choices, arch
  mix). I kept the legacy code path bit-identical (`plan=None`) and
  put all overrides behind the `if plan is not None` block so the
  diff is reviewable and the regression risk to existing config-only
  users is zero. Tests that already exercised the legacy path
  (`test_population_manager.py`, 188 passed) confirm this.

- **`reward_early_pick_bonus` was easy to remove cleanly.** Same
  reasoning as Session 1's `observation_window_ticks` retirement: the
  sampler iterates over whatever's in `search_ranges`, removing the
  entry just removes the gene from new genomes. Old checkpoints still
  have the stale key, but with the entry gone from `_REWARD_GENE_MAP`
  the trainer's extractor silently drops it — no crash path. Three
  test files referenced the gene and were updated; no production code
  did.

## 2026-04-07 — Session 5 (LSTM structural knobs)

- **Stacked `TimeLSTMCell` can't reuse the old squeeze-the-layer-dim
  shortcut.** The pre-Session-5 Time-LSTM forward did
  `h = hidden_state[0].squeeze(0)` to drop the layer dim because
  there was only ever one layer. For a stack this silently drops
  shape information. The fix is to keep `h_layers` / `c_layers` as
  Python lists of per-layer tensors (one entry per stacked cell),
  assign into the list as each layer runs, then `torch.stack(..., 0)`
  at the end to restore `(num_layers, batch, hidden)` for the
  outgoing hidden state. Assigning into the list avoids in-place
  mutation of autograd tensors.

- **Inter-layer dropout in the stacked Time-LSTM must gate on
  `self.training`, not a constructor flag.** `nn.LSTM` does this
  implicitly because it owns the dropout layer internally. For the
  hand-rolled stack, `F.dropout(h_new, p=..., training=self.training)`
  is the right call — the `.eval()` / `.train()` switch propagates
  to sub-modules automatically, so the cell stack doesn't need any
  extra plumbing. Test 4 in `test_lstm_structural.py` asserts both
  directions (eval → bit-for-bit identical, train → diverges under
  heavy dropout).

- **Actor-head gain=0.01 masks dropout divergence.** First draft of
  the eval-vs-train dropout test checked `out.action_mean`. The
  attenuation from the 0.01-gain orthogonal init collapsed the
  signal into ~1e-7 float noise — both passes looked identical to
  `torch.allclose(..., atol=1e-6)` regardless of dropout. Switched
  the assertion to the value head (critic uses gain=1.0 init) and
  the train-mode divergence became obvious at `atol=1e-4`. Rule:
  when writing a test that's meant to prove a middle-of-the-network
  source of variance reaches the output, route the assertion
  through the head with the **largest** init gain, or use a scaled
  input large enough to dominate the head.

- **PyTorch silently ignores `dropout=...` on `nn.LSTM(num_layers=1)`.**
  But the runtime *warning* it emits is loud and shows up in test
  output. Gating the kwarg on `num_layers > 1` in
  `PPOLSTMPolicy.__init__` keeps the test log clean without any
  behavioural change.

- **`int_choice` for booleans is fine.** I considered adding a
  dedicated `bool_choice` sampler branch for `lstm_layer_norm` but
  the payoff is near zero: `int_choice` with choices `[0, 1]`
  round-trips through the existing sampler, mutator, JSON
  checkpoint, and the `bool(...)` cast in the policy constructor
  without any new code path. Would have been pure scope creep. The
  only soft downside is the UI widget label ("0/1" instead of
  "true/false") — that'll be handled in Session 8.

- **`init_hidden` shape change is load-bearing but invisible.**
  Before Session 5 both policies returned `(1, batch, hidden)`.
  Now they return `(num_layers, batch, hidden)`. `PPOTrainer` only
  treats the hidden state as an opaque tuple (it calls
  `out.hidden_state` and feeds it back next step), so no trainer
  change was needed — but if a future caller hardcodes
  `hidden_state[0][0]` expecting the single layer, it'll silently
  take layer 0 of a stacked policy. Worth noting because the change
  is a landmine for anyone reading only one half of the diff.

## 2026-04-07 — Session 6 (transformer architecture)

- **The "hidden state" slot is now genuinely polymorphic.** For
  LSTM variants `hidden_state = (h, c)` — two tensors of the same
  shape. For the transformer it's `(rolling_buffer, valid_count)` —
  two tensors of *different* shapes and dtypes. The only invariant
  across architectures is "2-tuple of tensors, both of which can be
  moved to a device via `.to(device)`". The `BasePolicy` docstring
  now spells this out explicitly because the next architecture to
  land (hierarchical runner-attention?) will probably want its own
  bespoke state layout and should know the contract is "2-tuple of
  tensors", not "specifically an (h, c) pair". The reason this
  matters: `PPOTrainer._collect_rollout` has a hardcoded
  `hidden_state[0].to(device), hidden_state[1].to(device)` idiom and
  can't be easily changed without touching every architecture.
  Keeping the 2-tuple-of-tensors contract means no trainer edit is
  ever required when adding a new architecture.

- **Rolling-buffer warmup is "zero pad + learn to ignore".** I
  considered adding a `key_padding_mask` that tells the attention
  layer to skip unfilled slots during the warmup window. Then I
  realised this adds complexity for an effect the model can learn
  in a few hundred ticks (zero vectors produce zero attention
  scores from any reasonable linear Q/K projection; after layer
  norm their residual stream stays near zero too). Skipping the
  padding mask keeps the buffer tensor shape fixed and makes the
  batched forward path branch-free. The test for this behaviour
  (`test_rolling_buffer_retains_most_recent_ctx_ticks`) feeds
  distinctive fingerprint inputs and verifies the buffer contents
  slot-by-slot, which would catch any silent "pad is leaking" bug.

- **Causal-masking tests need `encode_sequence`, not `forward`.**
  `forward` returns only the last-position output (what the actor
  head consumes). To verify that position T's encoder output is
  independent of position T+1's input, the test needs to inspect
  an intermediate position. Added an `encode_sequence(obs)` helper
  that mirrors `forward` up to the encoder output but returns the
  full `(batch, ctx_ticks, d_model)` tensor. Kept out of
  `BasePolicy` because it's transformer-specific and the LSTM
  architectures have no analogous "per-position output" semantics.

- **`arch_change_cooldown` lives on the hp dict as metadata, not a
  spec.** Two reasons: (1) putting it in `search_ranges` would make
  the sampler treat it as a mutable gene and the repair step in
  Session 3's `_repair_reward_gene_pairs` would have to special-
  case it, and (2) the cooldown value carries *state* (gen-to-gen
  decrement + re-arm), not a *policy choice* the sampler should be
  drawing from a distribution. Keeping it as a plain dict entry
  that `PopulationManager.mutate` reads/writes means it travels
  with the agent through crossover and the model registry's
  `hyperparameters` column without schema plumbing. The only cost
  is that `crossover` doesn't propagate it (child starts with
  cooldown=0) — which is arguably correct: a child that *inherits*
  an architecture hasn't *mutated* into it.

- **The middle choice is the cleanest cooldown test fixture.**
  `PopulationManager.mutate`'s `str_choice` branch moves the index
  by ±1 with equal probability. Starting from index 0 or index
  `len-1`, half the rolls clamp (no change); only from a middle
  index does every ±1 always produce a real flip. Using
  `"ppo_time_lstm_v1"` (middle of the 3-choice list) as the test
  starting arch makes the assertion unambiguous regardless of RNG
  seed: with `mutation_rate=1.0` and cooldown=0, the arch MUST
  change. With cooldown>0, the arch MUST NOT change. No "unlucky
  RNG → false negative" failure mode.

- **Arch-specific LR override is opt-in and narrow.** The session
  plan mentions a *per-architecture LR range* as the canonical
  example of a knob transformers might want to set differently
  (they traditionally prefer lower LRs than LSTMs). I resisted the
  temptation to generalise this into a full per-arch hp-range
  override table because (a) nobody currently wants the generalised
  form, (b) the narrow form is two lines in `initialise_population`
  plus one optional field on `TrainingPlan`, and (c) the
  generalised form would need UI work to expose per-arch editing.
  Pure YAGNI — wait until a concrete need shows up, then widen.

- **`torch.cat([buffer[:, 1:, :], fused[:, t:t+1, :]], dim=1)` is
  simpler than `torch.roll` + slice-assign.** First draft used
  `torch.roll(buffer, -1, dim=1); buffer[:, -1, :] = fused[t]` which
  mutates in place and needs a `.clone()` to avoid aliasing
  complaints during training updates. The `torch.cat` version
  produces a fresh tensor every iteration, avoids the clone, and
  is bit-for-bit equivalent. Slightly more allocations, but
  `ctx_ticks ≤ 128` and rollout `seq_len == 1` make the cost
  negligible. Readability wins.

- **`enable_nested_tensor=False` is required with `norm_first=True`.**
  PyTorch emits a `UserWarning` on every `nn.TransformerEncoder`
  construction when `norm_first=True` because the nested-tensor
  fast path is disabled in that configuration. The warning appears
  once per policy instance, which means once per agent — so
  building a 50-agent population would spray 50 warnings into the
  training log. Passing `enable_nested_tensor=False` explicitly
  tells PyTorch we know what we're doing and silences the warning.
  We never pass `src_key_padding_mask` anyway (see "zero pad + learn
  to ignore" above), so the nested-tensor path wouldn't help us.

- **`register_buffer("causal_mask", ..., persistent=False)` is the
  right way to attach a static mask.** The mask depends only on
  `ctx_ticks`, which is frozen at construction. Registering it as
  a buffer means `.to(device)` / `.cuda()` move it with the module
  automatically, so the rollout path doesn't need
  `self.causal_mask.to(obs.device)` per call. `persistent=False`
  keeps it out of `state_dict()` so checkpoints stay clean — the
  mask is fully reconstructable from `ctx_ticks`.

- spinning up the ui really needs to spin up the api too, as many ui elements need api calls.

## Session 7 — Drawdown shaping

- **The obvious drawdown formulation is a trap.** The first
  instinct — `shaped = −ε × max(peak − current, 0) / budget` — is
  strictly non-positive, so a random policy accumulates negative
  shaped reward and "bet less" looks strictly better than
  "bet more". This is the same asymmetric-shaping bug the
  phantom-profit investigation spent a whole session fixing; the
  session plan flags it loudly up front and the design pass
  rules it out before any code is written. If a future session
  adds a new shaped term, write the zero-mean proof into the
  design pass **before** touching `_settle_current_race`. No
  exceptions.

- **Reflection symmetry is the cheapest zero-mean tool.** Option
  D (`(2·day_pnl − peak − trough) / budget`) is zero-mean not
  because the expectation integrates to zero by luck, but
  because every path and its sign-flipped reflection cancel
  *algebraically* when `peak_0 = trough_0 = 0`. Under `X → −X`,
  `peak ↔ −trough`, so the per-race term maps to its own
  negation. That means the zero-mean invariant can be guarded
  with a much stronger "reflection pairs cancel exactly" test
  (1e-9 tolerance) as well as the statistical "N=1000 within 2
  SE of zero" test the session plan asks for — and if an
  algebraic regression happens, the exact-cancellation test
  catches it even when the statistical test would get lucky.

- **`peak_0 = trough_0 = 0` is load-bearing.** The reflection
  proof breaks if you initialise peak or trough to anything other
  than zero (e.g. `-inf` / `+inf`), because then the initial
  values don't map to each other under `X → −X`. The helper is
  commented, but the invariant is easy to accidentally regress
  during a "cleanup" — if you touch `reset()` and change either
  default, the reflection test in `test_drawdown_shaping.py`
  will fail immediately.

- **Extract shaping formulas into helpers for unit testing.**
  The drawdown formula originally lived inline in
  `_settle_current_race`. Testing it that way would have
  required running the full bet-matching pipeline to produce
  a realistic `race_pnl` trajectory, which (a) is slow, (b)
  couples the invariant test to `ExchangeMatcher` + `BetManager`
  behaviour that has nothing to do with the reward math, and
  (c) picks up a small commission drift that would force the
  zero-mean test to loosen its tolerance. Pulling the formula
  out into `_update_drawdown_shaping()` — a method that only
  reads `_day_pnl` and mutates `_day_pnl_peak` / `_day_pnl_trough`
  — let the tests drive it directly with arbitrary synthetic
  trajectories. The settlement method is barely changed; the
  extract is purely a testability win.

- **Commission drift matters once you isolate the term.** The
  existing `early_pick_bonus` and `precision_bonus` zero-mean
  claims in `CLAUDE.md` are "zero-mean modulo commission". That
  drift is small enough to hide inside a mixed-term test, but
  if you run the drawdown term through the full env with the
  default 5 % commission and N=1000 trials, the drift is large
  enough to blow past 2 SE. The test fixture in
  `test_drawdown_shaping.py` sets `commission: 0.0` for exactly
  this reason — it isolates the mathematical property from the
  unrelated commission bias. Future zero-mean tests on new
  shaping terms should follow the same pattern.

