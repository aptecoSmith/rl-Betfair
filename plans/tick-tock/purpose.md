# Tick-Tock — hypothesis-driven era scheduling for PBT recipe discovery

**New here? Read in order:** `purpose.md` (this) → `current_state.md` (what already
exists to build on) → `design_decisions.md` (the pitfalls + open questions) →
`build_sketch.md` (the rough build surface).

This folder captures an idea + its first brick. The build-out will be designed in
a dedicated session; these docs exist so that session starts informed, not cold.

---

## The problem this solves

The PBT campaign searches the gene space by **local** moves only — perturb ±20%
around the current population, breed neighbours. It **cannot jump** to a designed
recipe in a region nobody is near. So good gene combinations are found by luck
(rolling the dice across eras), never by design. We want to add **directed,
hypothesis-driven** search on top of the PBT's local search — without giving up
the broad exploration that keeps us from converging prematurely.

## The idea: alternate Tick and Tock eras

- **Tick (explore).** A full-width era — every gene sampled across its full range
  (today's `--enable-all-genes` default). Builds up a population of agents+results
  (datapoints), explores regions we haven't, and *feeds the analysis*.
- **Phenotype analysis.** Correlate every gene against every behaviour
  (maturation / close / force-close / stop-close rates, locked_pnl, naked
  variance) across all accumulated agents → surface candidate drivers.
  Tool already built: `tools/phenotype_analysis.py`.
- **Hypothesis.** From the analysis, propose a recipe — the genes (and rough
  values/ranges) we believe drive the behaviour we want.
- **Tock (exploit).** An era whose fresh blood (the R1 rookie tier) is seeded
  **close to the hypothesis** (drivers pinned, the rest sampled). Run its full
  era. It tests whether the designed recipe actually performs.
- **Alternate.** Tick → Tock → Tick → Tock … The Ticks keep the net wide (never
  close off exploration); the Tocks test and narrow toward a good recipe.

## Why it's sound

- It adds the **one capability the PBT structurally lacks** — a teleport into a
  human-designed region (the Tock). The PBT then refines locally from there.
- Forced alternation is the **explore/exploit dial made explicit and safe**:
  always-Tick = never commit to a recipe; always-Tock = premature convergence and
  blindness to everything else.
- Both regimes **accumulate data** for future analysis — even a "failed" Tock adds
  datapoints in the hypothesised region.
- It is **interpretable human-in-the-loop active learning** (Tick = sample the
  space, the phenotype tool = a crude surrogate, Tock = exploit the surrogate).
  Can be partly automated later (a Bayesian-opt surrogate proposing the next
  Tock), but keep it human while proving the loop converges — the human can inject
  domain knowledge a black box can't.

## The key reframe: a Tock is a WARM-START, not a clean experiment

"Let the Tock run its full era" means the PBT will perturb/breed the seeded recipe
**away** from the hypothesis over the 5 generations. So the Tock's end-of-era
champions are "hypothesis + drift," not the pristine hypothesis. Don't fight that
— lean into it: **the Tock's job is to drop the PBT into a good neighbourhood and
let it refine.** The success question becomes *"does a hypothesis-warm-started era
beat a cold full-width Tick?"* (measured on held-out — see `design_decisions.md`).
That leverages the PBT instead of fighting it.
