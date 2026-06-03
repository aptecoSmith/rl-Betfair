# pbt-breeding — design (rotation gauntlet)

Operator design, 2026-06-03. Refines `purpose.md`'s 3-way split into a
**day-rotation gauntlet**: winning earns exposure to *unseen* data, so the
champions are progressively cross-validated across disjoint folds.

## Day structure
- **Sealed final test unchanged** (May 20–29) — never trained or selected on,
  ever, for any arm.
- Remaining pool → **3 rotations, random equal splits** (~10 days each).
  **Random, NOT difficulty-ordered** (load-bearing): i.i.d. rotations make a
  rotation-1 score comparable to a rotation-3 score. Shuffle, don't curate.
- Each rotation carries **a few eval days** (within-rotation held-out, e.g.
  ~3–4 of its 10) used to score an agent that TRAINED on that rotation's train
  days. An agent is **never scored on a rotation it trained on that round** in
  the train-set sense; the per-rotation eval days are its held-out.

## The gauntlet (core rule)
- Every lineage carries a **`rotations_seen` set**.
- **Fresh blood enters at rotation 1.**
- **Winning earns the lowest UNSEEN rotation** (1→2→3). Warm-start: the agent
  carries its weights into the new rotation (continue training), so it
  accumulates learning across *disjoint, increasingly-complete* data without
  ever re-seeing a fold while still climbing.
- A lineage that has seen all 3 rotations and keeps winning = **frozen
  champion**: it stops training (further training would re-overfit rotation 3),
  graduates to the **hall-of-fame leaderboard**, and is only re-scored.
- **Rotation 4+ (new data arrives):** thaw the champions, warm-start-train on
  the new rotation, re-score. Continuous validation as data accumulates.

Worked trace (operator's):
- Gen 1: 30 fresh, all rotation 1. seen={1}.
- Gen 2: top keepers + perturbed-offspring → rotation 2 (seen={1,2}); fresh
  blood → rotation 1 (seen={1}).
- Gen 3: original winners still top → rotation 3 (seen={1,2,3}); gen-2 fresh
  blood now winning → rotation 2; new fresh blood → rotation 1.
- Gen 4+: gen-2 fresh blood still winning → rotation 3; etc.
- Endgame: a leaderboard of recipes that each won their way through all 3
  rotations — proven generalists.

## DECIDED — promotion ladder / divisions (operator 2026-06-03)

Compete only *within your rotation tier* (rank rookies vs rookies, etc.). This
controls for the training-volume confound (a senior has warm-started N gens vs
a rookie's 1) and dissolves cross-rotation comparability. **Per-tier
composition, gen-on-gen:**

- **Rotation 1:** 100% fresh blood (a pure rookie division — so new blood only
  races peers, never seasoned veterans; this is what solves the "fresh blood
  drowns" handicap by construction).
- **Rotation 2+:** **50% graduated elites** (the prior gen's winners of the
  tier below, weights intact) + **50% offspring** bred from those winners.

Each gen: rank within each tier → tier-below winners promote (become next
tier's elites + spawn offspring), tier winners climb, R3 winners FREEZE (hall
of fame), losers culled, R1 refilled with fresh blood.

Sizing strawman (16 cores, ~30/gen, steady state from gen 3):
**R1 = 14 fresh · R2 = 10 (5 elite + 5 offspring) · R3 = 6 (3 elite + 3
offspring)**, promote top-5 of R1, top-3 of R2. Gens 1–2 are transients filling
the pipeline.

Sub-decisions — DECIDED (operator 2026-06-03):
- **Offspring = (a):** copy ONE winner's recipe (weights inherited from that
  one parent) + perturb ±20%. Implement as a pluggable `_make_offspring`
  function so (b) two-winner hyperparameter crossover can drop in later behind
  a flag. Weights are always inherited from one parent (can't cross brains).
- **Monoculture: NO cap.** We're hunting winning architectures; a dominant
  lineage is useful signal, and every R3 champion is recorded on the
  leaderboard so nothing is lost. The system is self-correcting WITHOUT a cap:
  it's *open* (fresh blood every gen) and the ladder gives equal training
  footing within a tier, so a better fresh lineage can topple an incumbent on
  merit. → **Measure lineage share per tier as an OBSERVABLE (insight), do not
  intervene.** The real protector of "let the best dominate" is **eval
  reliability**, not a diversity guard (see day split).

## Day split — DECIDED
- ~42 non-sealed days. Use **30 for rotations 1–3 (3 × 10 days)**; reserve the
  ~12 spare for the future rotation 4 (continuous-validation side-benefit).
- Each 10-day rotation: **6 train / 4 eval**. The extra eval day (vs a
  reflexive 7/3) buys a more stable per-tier `locked_per_std` ranking — which
  is what keeps a dominant lineage honest under the no-cap rule. The gauntlet
  compounds this: a champion is validated on 4+4+4 = 12 disjoint eval days
  across its climb. Tunable; rotations can grow to 3 × 14 if more train days
  are wanted.
- Random equal splits (i.i.d.), shuffled per `cohort_seed` so the A/B is paired.

## Other open decisions (carried from purpose.md)
- Tier sizes / promotion+relegation counts (ladder) OR 40/30/30 shares (single).
- Fresh-blood share + whether any are truly from-scratch vs inherited-brain +
  bold-recipe. (Rotation-1 entry already acts as a rookie bracket.)
- Perturbation magnitude (offspring ±20%, immigrant full-resample) — which genes.
- Pool size + per-rotation day count + eval-days-per-rotation.
- Weight-threading checkpoint (the big code lift; see master_todo Step 1).
- Episodes/gen under warm-start (tune down to keep wall-clock at parity).

## Execution / scheduling (keep the machine busy — operator 2026-06-03)

Goal: don't lose the ~9× multiprocess speed while running the gauntlet's
pipeline. Key distinction:

- **Share a day ≠ synchronise on a day.** The static_obs memmap
  (`shared-memory-day-cache`) already loads each day once and shares the one
  physical copy across every agent that maps it (OS page cache). So "a tranche
  of agents against the same day" is automatic in the FAST one-process-per-
  agent model — no orchestration needed.
- **Do NOT lockstep agents through a day** (the `--batched` GPU path). Measured
  ~2.55× vs ~9× for one-process-per-agent. Synchronising on days would cost the
  advantage. Keep agent-per-process; keep the memmap sharing.

**Model (simplified under "one rotation-step per generation"):** there is NO
within-generation barrier. Every generation trains the WHOLE population — R1,
R2 and R3 agents all at once — in parallel via the existing warm
`ProcessPoolExecutor`, because each agent's warm-start weights are already on
disk from the *previous* generation (gen N's weights → gen N+1's warm-starts).
The "pipeline" is across generations (an agent climbs one rung per gen), not
within one. So:

- Each gen, each agent = `train_one_agent(init_weights_path=<own/parent
  weights from last gen>, days=<its current rotation's days>)`. PPO is
  sequential within an agent (day N needs N−1); parallelism is across the ~30
  agents. Pool runs 16 at a time → 2 waves → full cores.
- After the gen: rank WITHIN each tier (cheap, post-gen), assign next gen's
  rotations (gauntlet promotion + breed offspring + refill fresh blood).
- No task-DAG, no cross-class juggling, no idle-at-the-tail — the population is
  ~constant ~30 every gen (composition shifts; size doesn't).

**Cost:** barely-new infra = the multiprocess pool + static_obs cache you
already have + a warm-start `init_weights_path` load + per-tier ranking &
promotion. Each rotation's ~10 days bake once (~0.9 GB) and stay shared across
that rotation's agents AND across generations. Memory ≈ the measured N=16
plateau (~50 GB) + ~1.8 GB for the three concurrent rotations' day caches.

NB: "share a day" (static_obs memmap, automatic) ≠ "synchronise on a day"
(lockstep `--batched`, ~2.55× — slower). Keep agent-per-process + memmap.

## Gene space + architecture exploration (operator 2026-06-03)

**Fresh blood gets the FULL gene space — no disbarring.** Drop the
`enabled_set` restriction for the pbt path: every rookie samples *every* gene
across its full range/choices (not just the `--enable-gene` subset the
gene-only GA pins). Offspring may perturb any non-structural gene (below).
Maximal exploration; the gauntlet + held-out selection sort it out.

**Architecture is a STRUCTURAL gene the gauntlet tournaments.** Fresh blood
draws an architecture at birth:
- `architecture` ∈ {`lstm`, (`time_lstm`), `transformer`}
- sizing: `hidden_size`/`d_model`, and for transformer `depth` (layers),
  `n_heads`, `ctx_ticks` (the v1 transformer's structural gene, values
  {32,64,128,256}).

Because every R3 champion's architecture lands on the **hall-of-fame
leaderboard**, the system *reports which architecture wins* — the operator's
question answered by construction, not guessed.

**Structural genes FREEZE within a lineage (load-bearing for warm-start).**
You can't warm-start a transformer from LSTM weights, nor a h256 brain from a
h64 one — the shapes differ. So an offspring/elite MUST keep its parent's
architecture + sizing to inherit the weights. Therefore:
- **Structural genes** (`architecture`, `hidden_size`/`d_model`, transformer
  `depth`/`n_heads`/`ctx_ticks`, lstm `num_layers`) are set ONLY at fresh-blood
  birth and frozen for the lineage's life.
- **Non-structural genes** (lr, entropy, clip, gae, value_coeff,
  mini_batch_size, all reward knobs, BC + direction params, gates) are
  inherited + perturbed ±20%.
- NB: this reclassifies `hidden_size` — freely mutated under gene-only GA, it
  becomes a fresh-blood-only gene under warm-start (mutating it would break
  weight inheritance).

**Build dependency:** the v2 stack (`agents_v2/discrete_policy.py`) has the
abstract `BaseDiscretePolicy` but only `DiscreteLSTMPolicy` implemented. Need a
`DiscreteTransformerPolicy` on the same interface (PORT v1's
`agents/policy_network.py::PPOTransformerPolicy`, which already carries
`ctx_ticks` + the architecture-hash weight-shape checks) + a **policy factory**
that builds the backbone from a genome. See master_todo Step 1b.

## Why this is better than vanilla PBT
Vanilla PBT exploits+explores on a fixed data stream → can still overfit the
stream. The rotation gauntlet forces every promotion onto unseen data, so the
selection signal is generalization by construction, and the hall-of-fame is a
set of recipes validated on 3 disjoint folds in sequence — a temporal
cross-validation the single-population GA never had.
