# Hard constraints — gauntlet-pipeline

Violating any of these breaks the science or the comparison. Check before
building or launching.

## Recipe purity (the load-bearing one)
- **A model is cooked under ONE gene config, from T1, warm-starting only its
  OWN weights between tranches.** NEVER warm-start an agent from weights cooked
  under a *different* recipe. This is why mutants climb from T1 rather than
  inheriting a survivor's deep weights — the cheap warm-start is rejected on
  principle (`purpose.md`). A reported winning recipe MUST reproduce its model
  if re-run clean from T1.
- Survivors keep their genes fixed across every tranche they clear (already
  true). Breeding may change genes ONLY for a NEW recipe entering at T1, never
  for an in-flight agent.

## Gauntlet shape
- **Tranches are fixed-size.** The gauntlet grows by APPENDING T(N+1); tranches
  are never resized. Every model must face the identical T1..TN sequence —
  this is what makes cross-agent and cross-era comparison fair.
- **Full fair shot:** a recipe climbs the whole current gauntlet uninterrupted;
  it is culled only after completing the same depth as the incumbents, judged in
  a same-depth comparable pool. No early-culling of climbers.

## Execution decoupling
- The **executor (`run_tranche`) contains NO selection** — it trains + evals a
  batch on one tranche and returns scores. Selection lives ONLY in the breeder.
  (A selection step hidden in the executor re-couples the loop and re-creates the
  catch-up-in-one-generation problem.)
- Per-run cost stays uniform: `batch × one tranche`. No run replays multiple
  tranches in one shot.

## Selection / evaluation (carried over from holdout-selection.md)
- Select among **same-depth** agents only, on the **fixed validation set at
  fc=0** (the held-out-selection regime). Rank on the fc=0 composite
  (`locked_weighted` / `locked_over_sigma`), never day_pnl, never maturation
  rate. Hard ceiling σ_naked_leg ≤ £30.
- The **sealed final-test** (`--holdout-recent`) stays inviolate — scored
  post-run at fc=120 (deploy) + fc=0 (rail diagnostic), never trained or
  selected on. Leakage asserts: `validation ∩ train = ∅`,
  `final_test ∩ (train ∪ validation) = ∅`.

## Memory / performance
- The **shared-memory day cache must stay intact** (predictors-ON: per-day
  `static_obs` `.npy` memmapped `mmap_mode='r'`, ONE OS-page-cache copy across
  workers). Verified present 2026-06-13; any refactor must not regress it to
  per-worker copies. Day growth must scale the SHARED cache (one copy), never
  ×workers.
- Predictors ALWAYS ON; fast path `--parallel-agents 16 --device cpu`. Never
  `--batched`.

## Migration safety
- New path is NOT byte-identical — gate behind a flag; keep `--breeding
  lockstep` working until cutover.
- **A/B vs current lockstep on the same data before adopting:** held-out
  locked/σ on the sealed-7 must match-or-beat the old loop. No cutover on faith.
- The launch-flag foot gun (`feedback_audit_launch_wiring`): any new knob with
  both a CLI flag and an hp/gene source resolves with OR-semantics + a
  both-sources test; verify activity counters on agent-1/day-1.
