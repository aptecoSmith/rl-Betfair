# Build sketch — the rough surface area (a starting point, NOT a finished plan)

The operator will design the build in a dedicated session. This is just the
surface area so that session starts from something, not a blank page.

## Pieces a Tick-Tock loop needs
1. **Tock launcher** — start an era with the hypothesised DRIVERS pinned and the
   rest full-sampled. Today: `--reward-overrides` / per-gene flags pin to a value;
   `--enable-all-genes` samples full ranges. Likely v1 = "pin drivers + sample
   rest" (already possible); v2 = a per-gene *narrow-range* sampling option so a
   Tock seeds a band around the hypothesis rather than a single point. The pin
   must land in **R1 fresh-blood sampling** (the rookie-injection path), not just
   the cohort-wide overrides — check how `init_pbt_population` draws fresh blood.
2. **Era tagging** — a per-cohort metadata file (`era_meta.json`: `tick|tock`,
   hypothesis, pinned genes, parent report) written at launch; the phenotype tool
   reads it and stamps its outputs.
3. **Cross-era pooling** — extend `tools/phenotype_analysis.py` to pool multiple
   cohort dirs (bigger n) and split correlations by tag (tick vs tock).
4. **Held-out comparison harness** — a fixed held-out day set + a script that
   re-evals Tock champions vs Tick champions on it (reuse
   `tools/reevaluate_cohort.py`), reporting **locked + naked-variance** (the real
   success metric, per `design_decisions.md`).
5. **Loop orchestrator (optional, later)** — wrapper that runs Tick → analyse →
   form hypothesis (human or rule) → run Tock → held-out compare → repeat. v1 can
   be fully manual (operator drives each beat via the `.bat`s + the tool).

## Suggested v1 (minimal, manual)
The phenotype tool (have it) + a Tock launcher that pins drivers (mostly have it)
+ manual era tagging + the held-out harness. The operator drives the cadence
(Tick, run tool, decide a hypothesis, Tock, held-out compare). Prove the loop
adds value before automating any of it.

## Later
Narrow-range sampling, cross-era pooling, an auto-hypothesis (Bayesian-opt
surrogate over the gene→outcome data), a full orchestrator, adaptive scheduling.

## Existing assets to reuse
- Phenotype tool: `tools/phenotype_analysis.py` (commit `1e57c7a`).
- Campaign wrapper + controls (a Tick today):
  `plans/pbt-gpu-forward/_scripts/run_genes_campaign.ps1`, root
  `start_pbt_training.bat` / `stop_pbt_loop.bat` (the `-Eras N` param).
- Held-out re-eval: `tools/reevaluate_cohort.py` (used for the earlier 6-champion
  sealed-day eval; reads a cohort scoreboard, supports `--reward-overrides`,
  `--filter-agent-ids`).
- Gene inventory + ranges: `genes_census.md` (repo root).
- PBT mechanics: `training_v2/cohort/runner.py` (gen loop, tier→rotation) and the
  pbt module (`init_pbt_population`, `breed_pbt`, `make_rotations`).

## First concrete experiment, ready to run
The first hypothesis already exists (see `current_state.md` §2): the
direction-machinery recipe (`direction_gate_enabled→1`, `use_direction_predictor
→1`, higher `bc_learning_rate`, `stop_loss→~0.22`, `bc_pretrain_steps→0`). When the
current Tick (`pbt_genes_v2`) finishes accumulating, that recipe is a ready-made
first Tock to validate against it on held-out days.
