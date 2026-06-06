# Kickoff — build Tick-Tock

Open this session **in the rl-betfair repo**. You're building the Tick-Tock
system; the design is **settled** — this session implements it. Heads-up: Phase 1
touches **core training code** (`genes.py`, `pbt.py`, `runner.py`), so move
carefully and keep the regression gate green.

## Read first (the settled spec)
- `plans/tick-tock/build_plan.md` — **your spec.** Settled decisions, the loop +
  marker protocol, build pieces A–F, the sequence/gates, and the first concrete tock.
- `plans/tick-tock/purpose.md` — why (directed search on top of the PBT's local
  search; a Tock is a warm-start, not a clean experiment).
- `plans/tick-tock/current_state.md` — the bricks that exist (the phenotype tool,
  the PBT mechanics, the first analysis result + the first-tock recipe).
- `plans/tick-tock/design_decisions.md` §RESOLVED — the settled calls + pitfalls.
- For the code you'll edit: CLAUDE.md "Fast cohort training" +
  `training_v2/cohort/{genes,pbt,runner}.py`.

## Placement (settled)
Design docs stay in `plans/tick-tock/`. The **built system** goes to **top-level
`tick-tock/`** (orchestrator script, start/stop bats, `worker/`, `work/`) — a
standing operating loop like `wiki/`, NOT under `plans/`.

## Build order — DO NOT skip the de-risk gate

**Phase 1 — build A–D, then ONE manual cycle:**
1. **A — fresh-blood band-seed (the keystone).** `seed_bands` in `genes.py` →
   thread through `pbt.py` (`_fresh` / `init_pbt_population` / `breed_pbt` R1
   refill) → `--seed-gene NAME=LO:HI` + `--era-type` / `--hypothesis-id` in
   `runner.py`. **Load-bearing test: a no-seed tick is byte-identical to today at
   the same seed.** Also test: a band draw stays in range AND is *recorded* in the
   row's `hyperparameters`; a structural seed holds era-wide; offspring drift from
   the seed (not the gene default). Wiring guards: auto-enable non-structural
   seeded genes, collision-guard vs `--reward-overrides`, enforce the
   `use_direction_predictor ⇄ direction_gate_enabled` coupling.
2. **B — row tagging** (`era_id` / `era_type` / `hypothesis_id` at the scoreboard
   + `model_register` writer; tolerant readers).
3. **C — phenotype `--tick-only`** (filter discovery correlations to tick rows).
4. **D — held-out compare harness** (wrapper over `tools/reevaluate_cohort.py`:
   tick-vs-tock champions on the sealed-7, **fc=0 AND fc=120**, report
   **locked_pnl + σ_naked_leg + paired delta**, append a peek-ledger row).
5. **The manual cycle (the proof):**
   - Run the phenotype analysis on the completed **first Tick** — the running
     campaign era `registry/pbt_genes_v2` (confirm era 1 finished first).
   - **Resolve the recipe contradiction** (build_plan.md "First concrete tock":
     `bc_learning_rate` high + `bc_pretrain_steps=0` is inert — pick BC-on+LR OR
     BC-off and drop the LR seed), then author the first hypothesis (the direction
     recipe) as `tick-tock/work/hypotheses/hypothesis_001.md`.
   - **⛔ STOP — show the operator the analysis + the hypothesis and get sign-off
     before launching the tock.** This is the ONE human gate (per the build order).
   - Launch the direction tock manually via `--seed-gene …`; run the held-out
     compare vs the first Tick. Confirm the seed **lands + records + drifts** and
     the locked / σ_naked numbers are sane.

**Phase 2 — only after Phase 1 proves out: build E–F at top-level `tick-tock/`.**
- **E — file-handshake era-loop** (`run_tick_tock.ps1` + start/stop bats + `work/`).
- **F — hypothesis brain + scheduled worker** (`tick-tock/worker/`, multi-candidate
  + self-critique).
Don't build the autonomous loop before the manual cycle has shown the mechanism
and metrics are sound.

## Operating constraints
- **The box is CPU-core-bound at N=16.** Editing source is fine anytime, but
  **running tests / a tock / the held-out compare needs cores free** — check
  whether a campaign is running first. (The pbt relaunch wrapper was killed
  2026-06-06 so no new eras auto-start, but era 1 of `registry/pbt_genes_v2` may
  still be finishing.) Don't disturb a running era.
- **force_close stays 0 in training**; fc=120 only at held-out/deploy eval.
- Select on **locked_pnl + σ_naked_leg**, NEVER day_pnl.
- Commit in revertible chunks (A; B; C; D; the manual-cycle artifacts; then E; F).

## Independent of
The memory-improvements / wiki work (`plans/memory-improvements/`) — separate
session, separate system. Don't interleave them.
