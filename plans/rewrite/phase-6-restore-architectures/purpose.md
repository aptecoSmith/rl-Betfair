---
plan: rewrite/phase-6-restore-architectures
status: design-locked
opened: 2026-05-03
depends_on: rewrite/phase-3-cohort,
            rewrite/phase-4-restore-speed,
            rewrite/phase-5-restore-genes
---

# Phase 6 — restore-architectures: bring TimeLSTM and Transformer into v2 cohorts

## Purpose

v1 supported three policy architectures: `PPOLSTMPolicy`,
`PPOTimeLSTMPolicy` (time-aware variant), and `PPOTransformerPolicy`.
v2 currently supports only `DiscreteLSTMPolicy`. The cohort runner
instantiates `DiscreteLSTMPolicy` unconditionally; there's no
architecture-choice gene; agents in a cohort vary only by hidden
size (64, 128, 256).

When the rewrite is otherwise complete and ready to scale up,
**single-architecture cohorts are not sufficient.** Different
problems prefer different architectures:

- **LSTM** — strong default; recurrent state captures pre-race
  history; cheap per-step cost.
- **TimeLSTM** — explicit time-step embedding; better at
  extrapolating from sparse early-race data.
- **Transformer** — full-context attention up to
  `transformer_ctx_ticks` (32–256 per CLAUDE.md); strongest at
  capturing long-range dependencies in the pre-off window.

A cohort that lets the GA select across all three lets diversity
emerge — agents converge to the architecture that best fits their
gene combination's other parameters. The current rewrite locks in
LSTM, which is fine for the rewrite verdict but wrong for
production scale-up.

## What this phase does

Three policy ports + cohort plumbing:

1. **Port `PPOTimeLSTMPolicy` → `DiscreteTimeLSTMPolicy`** in
   `agents_v2/`. Same time-step embedding, same auxiliary heads
   (fill_prob, mature_prob, risk), same hidden-state-on-update
   contract.
2. **Port `PPOTransformerPolicy` → `DiscreteTransformerPolicy`** in
   `agents_v2/`. Same context-window attention, same
   `transformer_ctx_ticks` choice set `{32, 64, 128, 256}`, same
   auxiliary heads, same architecture-hash break behaviour.
3. **Architecture as a per-agent gene** — add `policy_class:
   Literal["lstm", "time_lstm", "transformer"]` to `CohortGenes`
   plus the architecture-specific genes (`transformer_ctx_ticks`
   only fires when `policy_class == "transformer"`).
4. **Cohort plumbing** — worker dispatches on `policy_class` to
   the correct constructor; registry's `architecture_name`
   reflects the actual class (`v2_discrete_ppo_lstm_h128`,
   `v2_discrete_ppo_time_lstm_h128`, `v2_discrete_ppo_transformer_
   h128_ctx256`); breeding respects architecture compatibility
   (cross-class crossover not allowed — would corrupt weight
   shapes).

The result: a single cohort run can have a mix of LSTM, TimeLSTM,
and Transformer agents, and GA selects the architecture that wins
under the cohort's training budget.

## Why this is its own phase

`phase-3-cohort` made a single-architecture choice deliberately
(simplicity for the first cohort scaffold). `phase-4-restore-speed`
attacked per-tick overhead. `phase-5-restore-genes` brought the
search-dim count up to v1 levels. **Architecture diversity is the
last v1-feature gap.** It has its own non-trivial work surface:
porting two whole policy classes (each with multiple aux heads),
plus careful cohort breeding rules to prevent cross-architecture
weight corruption.

This phase is also the prerequisite for **Phase-4-style speed
work on TimeLSTM and Transformer** — Phase 4's per-tick wins are
LSTM-specific. Once we have all three architectures in v2, we can
re-do per-tick speed analysis on TimeLSTM and Transformer if
their cohort-wall demands it.

## What's locked

### Each policy port keeps v1's contract bit-for-bit

`DiscreteTimeLSTMPolicy` and `DiscreteTransformerPolicy` are
adaptations to v2's discrete-action interface, not behavioural
rewrites. Same forward-pass semantics, same auxiliary heads, same
hidden-state-on-update contract (per CLAUDE.md §"Recurrent PPO:
hidden-state protocol on update"), same masked-categorical action
sampling. Each port ships with v1↔v2 parity tests that compare
forward outputs at fixed weights and inputs.

### CUDA↔CUDA self-parity holds per architecture

Two CUDA runs at the same seed and same architecture choice
produce bit-identical `total_reward` and `value_loss_mean`
per agent. Same load-bearing parity guard from Phase 3
Session 01b. Different architecture choices produce DIFFERENT
results (that's the whole point), but architecture × seed
locks the result.

### No GA crossover across architectures

A child agent's `policy_class` MUST equal one of the parents'
`policy_class` (uniform pick from {parent_a, parent_b}). When the
two parents have the SAME `policy_class`, the child inherits all
weight-shape-compatible genes via standard crossover. When the
two parents have DIFFERENT `policy_class`, the child:
- Inherits `policy_class` from one parent (50/50).
- Inherits weight-shape-compatible genes (LR, entropy_coeff,
  etc.) from either parent normally.
- Inherits architecture-specific genes (`transformer_ctx_ticks`,
  hidden_size) ONLY from the parent matching the child's
  `policy_class` — values from the OTHER parent are not
  shape-compatible.
- The child's policy weights are initialised fresh (no inherited
  weights) — cross-architecture weight transfer isn't physically
  possible.

This is the load-bearing breeding rule: cross-architecture
mating produces a fresh-init child of one parent's class, with
hyperparameters shaped to that class.

### Architecture-name discriminator extended

Registry `architecture_name` strings:
- `v2_discrete_ppo_lstm_h{hidden_size}` (existing)
- `v2_discrete_ppo_time_lstm_h{hidden_size}` (NEW)
- `v2_discrete_ppo_transformer_h{hidden_size}_ctx{ctx_ticks}` (NEW)

The transformer's `ctx_ticks` is part of the architecture-name
because it changes the position-embedding shape (per CLAUDE.md
§"Transformer context window — 256 available"). Different
ctx_ticks values are different weight-shape variants.

### Schema growth, not break

`CohortGenes` adds `policy_class: str` and
`transformer_ctx_ticks: int` (optional). Existing serialised
genes (registry rows, scoreboard JSONL) read with default-
tolerance: `policy_class` defaults to `"lstm"`, ctx_ticks to
default 64.

### No env edits

The env (`env/betfair_env.py`, `env/bet_manager.py`,
`env/exchange_matcher.py`) is untouched. All work in
`agents_v2/` and `training_v2/cohort/`.

### Same `--seed 42` for cross-cohort comparison

Per-architecture cohort rewards are noisy on small datasets;
cross-cohort comparison with fixed seed and fixed gene set is
the only meaningful diff.

## Success bar

The plan ships GREEN iff:

1. **All three architectures available in v2** as
   `DiscreteLSTMPolicy`, `DiscreteTimeLSTMPolicy`,
   `DiscreteTransformerPolicy`.
2. **v1↔v2 forward-pass parity** for the two new policies at fixed
   weights/inputs (within fp32 epsilon).
3. **`policy_class` is a CohortGenes field** sampleable in
   `{lstm, time_lstm, transformer}` via an `--enable-arch-gene`
   CLI flag (or unconditionally per the gene-promotion design).
4. **Architecture-respecting breeding** — child policy_class
   is one of parents'; cross-architecture crossover doesn't
   corrupt weight shapes; new tests verify this.
5. **Mixed-architecture validation cohort runs cleanly** — 4
   agents per architecture × 1 generation produces 12
   AgentResult rows, scoreboard reflects each agent's actual
   architecture, no crashes from cross-architecture handling.
6. **CUDA↔CUDA self-parity holds per architecture** at fixed
   seed.

## Sessions

### Session 01 — port `DiscreteTimeLSTMPolicy`

Adapt `agents/policy_network.py::PPOTimeLSTMPolicy` to v2's
discrete-action interface. The class lives in
`agents_v2/discrete_policy.py` (or a new `agents_v2/
discrete_time_lstm.py` if file size grows too much).

- Inherit from `BaseDiscretePolicy`.
- Re-implement the time-step embedding path.
- Port all auxiliary heads (fill_prob_head, mature_prob_head,
  risk_head) with their v1-shape outputs.
- Preserve hidden-state-on-update contract (CLAUDE.md
  §"Recurrent PPO: hidden-state protocol on update").
- Add `tests/test_v2_discrete_time_lstm_policy.py` with v1-parity
  guard + masked-categorical sampling tests + hidden-state
  protocol test.

Session prompt: `session_prompts/01_port_time_lstm.md`.

### Session 02 — port `DiscreteTransformerPolicy`

Same shape as Session 01 but for `PPOTransformerPolicy`.
Additional concerns:

- `transformer_ctx_ticks ∈ {32, 64, 128, 256}` (per CLAUDE.md)
  must be a constructor parameter, not a magic constant.
- Position embedding shape depends on ctx_ticks — different
  values produce shape-incompatible weights (architecture-hash
  break per CLAUDE.md).
- Causal mask construction must match v1.

Session prompt: `session_prompts/02_port_transformer.md`.

### Session 03 — architecture as a gene + cohort plumbing

Extend `CohortGenes` with:
- `policy_class: str` ∈ `{"lstm", "time_lstm", "transformer"}`.
- `transformer_ctx_ticks: int` ∈ `{32, 64, 128, 256}` (only
  consulted when `policy_class == "transformer"`).

Update breeding (`crossover`, `mutate`) for architecture
compatibility per the §"What's locked" rules.

In `training_v2/cohort/worker.py::train_one_agent`, dispatch on
`genes.policy_class` to construct the right policy class.

Update `arch_name_for_genes()` in worker.py to produce the new
architecture-name strings.

Add `--enable-gene policy_class` CLI flag (to phase-5-restore-
genes' --enable-gene mechanism). When enabled, GA samples
architecture per agent. When disabled, all agents are LSTM
(legacy default).

Session prompt: `session_prompts/03_architecture_gene_and_plumbing.md`.

### Session 04 — validation cohort + writeup

Mixed-architecture validation: 12 agents × 1 generation, all
three architectures evolving via gene flag. Sanity-check:
- Each policy_class appears at least once in the 12 agents.
- All 12 agents complete eval cleanly.
- Scoreboard rows reflect the actual architecture.
- Per-architecture mean total_reward shows whether one
  architecture clearly dominates on the dataset.

Update `findings.md` with the validation summary.

Session prompt: `session_prompts/04_validation_and_writeup.md`.

## Hard constraints

In addition to all rewrite hard constraints + phase-3-cohort
constraints + inherited from phase-3-followups, phase-4, phase-5:

1. **No env edits.** Architecture work happens in
   `agents_v2/` and `training_v2/cohort/`.
2. **No GA crossover across architectures.** See §"What's
   locked" for breeding rules.
3. **v1↔v2 forward-pass parity** for ported policies. The new
   classes are adaptations, not rewrites.
4. **Architecture-name strings change.** Registry readers must
   accept the new patterns; they already do via the
   architecture-hash check, but spot-check the v1 UI.
5. **CUDA↔CUDA self-parity holds per architecture** at fixed
   seed.
6. **Hidden-state contract preserved** for time-LSTM. The
   transformer's hidden state is its (ctx_buffer, valid_count)
   tuple — different shape entirely, but follows the same
   capture-before-forward rule.
7. **Schema is forward-only.** New genes append; no removes,
   no renames.
8. **Default behaviour unchanged.** Without `--enable-gene
   policy_class`, all agents are LSTM (the existing default).
   Byte-identity for legacy launches.
9. **Same `--seed 42`** for any cross-cohort comparison cohort.
10. **NEW output dirs** for every cohort run.

## Out of scope

- Per-architecture speed work (Phase 4 was LSTM-only; if
  TimeLSTM or Transformer cohorts are too slow, that's a
  follow-on plan).
- New auxiliary heads. The three existing heads (`fill_prob`,
  `mature_prob`, `risk`) port verbatim.
- New architectures beyond the three named (no Mamba, S4, etc.
  — those are research questions, not "restore parity with v1").
- Ensemble methods or architecture mixing within a single agent.
- 66-agent scale-up.
- v1 deletion.

## Useful pointers

- v1 policy classes:
  [`agents/policy_network.py`](../../../agents/policy_network.py).
- v2 single-policy class:
  [`agents_v2/discrete_policy.py`](../../../agents_v2/discrete_policy.py).
- Cohort worker (where policy gets instantiated):
  [`training_v2/cohort/worker.py`](../../../training_v2/cohort/worker.py).
- Hidden-state contract: CLAUDE.md §"Recurrent PPO: hidden-state
  protocol on update".
- Transformer ctx_ticks: CLAUDE.md §"Transformer context window —
  256 available".
- Architecture-hash break convention: CLAUDE.md §"fill_prob feeds
  actor_head" + §"mature_prob_head feeds actor_head" — same
  pattern for any constructor change that shifts weight shapes.

## Estimate

Per session:

- Session 01 (port TimeLSTM): ~4 h (3 h port + 1 h tests).
- Session 02 (port Transformer): ~5 h (4 h port + 1 h tests; the
  transformer is more complex).
- Session 03 (gene + plumbing): ~2 h (gene schema + worker
  dispatch + breeding rules + tests).
- Session 04 (validation): ~1.5 h (small cohort + writeup).

Total: ~12.5 h. No GPU cohort wall in any session.

If past 8 h on Session 01 or 02 excluding tests, stop and check
scope — porting a policy class is largely mechanical (copy-paste
from v1, adapt the action interface, port the aux heads). A long
session means the BaseDiscretePolicy interface needs updating to
support the new policy's quirks; that's a separate refactor.

## When to do this

After the rewrite ships its mechanism + Bar-6c verdicts (per
phase-3-followups/force-close-architecture, possibly phase-7
mechanic plans). Architecture diversity is **the last v1-feature
gap before scale-up**, but it's not on the critical path for
current verdicts — those use single-architecture cohorts and are
comparable across runs at fixed architecture.

A reasonable trigger to start this phase: when the rewrite's
mechanic and gene work has stabilised AND we've decided to scale
beyond 12-agent / 1-architecture cohorts.
