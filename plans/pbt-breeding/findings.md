# pbt-breeding — findings

Running record of what each step delivered and what it proved. Honest
reporting per HC#6/#7 — mechanism claims are separated from (not-yet-run)
empirical results.

---

## Step 1 — Weight-threading + forward-match gate ✅ (2026-06-03)

**Built.** Warm-start weight inheritance in the v2 cohort worker:

- `train_one_agent(init_weights_path=...)` — when set, loads an inherited
  `state_dict` into the freshly-built policy BEFORE any BC/PPO, then
  continues as a PPO fine-tune. Default `None` = cold-start.
- `worker.load_warm_start_weights(policy, path)` — THE single warm-start
  load path (worker + future reeval/factory both call it, HC#11). Unwraps
  the `ModelStore` envelope (`{"weights": ..., "obs_schema_version": ...}`),
  then `load_state_dict(strict=True)`.
- Warm-started agents **skip BC pretrain** — they inherit a trained
  `actor_head`; re-running BC would overwrite it and re-set the
  `input_norm` buffers away from the parent's inherited stats.
- Threads through the multiprocess pool **automatically** — `init_weights_path`
  is a picklable kwarg in the per-agent spec dict that `_train_agent_worker`
  already forwards to `train_one_agent(**spec)`. No `multiproc_worker` change.

**GATE — inheritance is REAL, and verified (HC#5).**
`tests/test_v2_pbt_warm_start.py` (5 tests, all pass):

- A warm-started child's gen-0 forward is **bit-identical** (`torch.equal`)
  to the parent's final forward on a fixed obs, BEFORE any new gradient
  step — across **both** the real `ModelStore.save_weights` envelope and a
  bare `state_dict`. Compared 11 forward output tensors (logits,
  masked_logits, per-runner value, stake α/β, fill/mature/risk/direction
  heads). The pre-load child is asserted to genuinely differ first, so the
  test can catch a no-op load.
- The `input_norm` buffers (`obs_mean`/`obs_std`) — registered buffers, not
  `nn.Parameter`s — are explicitly asserted to transfer (the thing most
  likely to be silently dropped).
- Strict load **raises** on a structural-gene mismatch (different
  `hidden_size`) — the loud-failure guarantee the breed step's
  structural-gene freeze depends on (HC#10).
- Missing path raises `FileNotFoundError`.

**Default-off byte-identity (HC#1).** The existing 38 worker+genes tests
(`test_v2_cohort_worker.py`, `test_v2_cohort_genes.py`) still pass; the
foundation golden-parity static_obs case (3 cases) passes.

**No empirical results yet** — heritability-across-gens, selection
spread÷signal, lineage diversity, fresh-blood survival, and held-out
`locked_per_std` all require the breed step (Step 2) + rotation (Step 3) +
instrumentation (Step 4) + the A/B (Step 5). Step 1 proves only that the
inheritance *mechanism* is correct, which is the precondition for all of
them.

**Tail folded into Step 2/3.** master_todo's Step 1 also lists "parent→child
weight COPY" and "extend the resume/checkpoint to carry weight pointers."
Both are inseparable from the breed step: the registry already stores each
agent's weights at `registry/weights/<model_id>.pt`, so a parent→child
"copy" is just pointing the offspring's `init_weights_path` at the parent's
existing file (Step 2 wiring), and the checkpoint pointers ride with the
lineage/rotation state introduced in Steps 2–3. No physical copy is needed.

**Commits (branch `pbt-breeding`):**
- `9ffc333` — foundation: shared-memory static_obs day cache + reeval
  input_norm fixes (the infra pbt builds on, committed first per the brief).
- `814eadf` — Step 1: warm-start weight-threading + forward-match gate.

→ **STOPPED here for operator review** (the brief's first mandatory
stop-point: "after Step 1's forward-match gate"). Operator: "proceed, and
keep going to completion without asking me" → continuing autonomously
through the remaining steps; the only remaining hard stop is before the
Step 5 A/B burns compute, which I'll launch detached + logged + monitored
(per the autonomous-when-away pattern) rather than block on.

---

## Step 1b — Architecture genes + v2 transformer + ONE policy factory ✅ (2026-06-03)

**Built.** The architecture tournament's foundation.

- **4 structural genes** on `CohortGenes`: `architecture` ∈ {lstm,
  transformer} + transformer `depth`/`heads`/`ctx_ticks` (hidden_size
  doubles as d_model). Frozen within a lineage (HC#10).
  `sample_fresh_blood_genes(rng)` draws them across the full choice sets
  (HC#9); the base `sample_genes` PINS them to the LSTM default and
  `crossover`/`mutate` SKIP them with **no rng draw** — so `--breeding pbt`
  off stays byte-identical to the gene-only GA (HC#1; the
  "adding-a-field-shifts-the-RNG-stream" trap, guarded by a test).
- **`DiscreteTransformerPolicy`** (`agents_v2/discrete_policy.py`): ports
  v1's `PPOTransformerPolicy` backbone (rolling ctx-tick buffer + learned
  positional embedding + causal `nn.TransformerEncoder`) onto
  `BaseDiscretePolicy`. It SUBCLASSES `DiscreteLSTMPolicy` purely to reuse
  the intricate head construction (fill/mature/risk + per-runner direction
  head w/ frozen-manifest loading + actor + both gates + input_norm), then
  `del`s the LSTM and swaps in the transformer backbone. **The LSTM is left
  byte-for-byte untouched** (its 28-file test suite is the safety net). The
  forward tail is duplicated (commented "keep in sync") rather than shared
  via a mixin, to avoid touching the load-bearing LSTM. Hidden state is
  `(buffer, valid_count)` (batch on dim 0) → re-overrides the LSTM's dim-1
  pack/slice helpers back to the BasePolicy defaults.
- **`agents_v2/policy_factory.py::build_policy(genes, ...)`** — THE single
  genome→policy constructor (HC#11), duck-typed on the genome (no
  training_v2 import). `policy_arch_name(genes)` gives the registry
  discriminator. The worker AND `tools/reevaluate_cohort.py` both build
  through it; reeval now also threads `runner_dim` from its env (it was
  omitted before, which strict-load-rejected every lean-obs checkpoint —
  the same class of bug input_norm was).

**Gates — all pass** (`tests/test_v2_transformer_policy.py` ×10,
`test_v2_policy_factory.py` ×13, `test_v2_pbt_fresh_blood.py` ×8):

- Factory builds + a forward runs for every architecture/sizing.
- Transformer checkpoint round-trips strict (no `lstm.*` keys; shares the
  exact head-module keys with the LSTM).
- **Transformer TRAINS END-TO-END through the real v2 PPO trainer** — the
  load-bearing de-risk: its `(buffer, valid_count)` state survives the
  rollout collector's pre-allocated capture buffers and the PPO update's
  pack/slice helpers, and backbone weights move after an update.
- Warm-start (Step 1) composes with the transformer (forward reproduces).
- LSTM factory path is byte-identical to a direct `DiscreteLSTMPolicy(...)`.
- Fresh blood covers both architectures; structural genes never shift the
  base sampler / crossover / mutate RNG stream.

**Regression:** the wider v2 suite (cohort runner, multiproc cluster,
discrete-ppo trainer/rollout, multi-day, gates) passes; two schema-count
guards updated for the 4 new gene keys (39 total).

**Commit:** `pbt-breeding` — see `git log`.
