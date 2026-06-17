# CLI cleanup audit — `training_v2/cohort/runner.py` (79 flags)

**Status: AUDIT ONLY (2026-06-13).** Nothing removed yet — this is the classified
kill-list for operator sign-off. Operator decisions (2026-06-13): **gauntlet is
the only breeding mode we're keeping** (lockstep → fallback then retire; tier-
ladder pbt + gene-only ga → retire). Cleanup depth = audit doc first.

Removing a flag is destructive (breaks saved commands, bats, tests) and some are
mode-coupled, so removals are **sequenced by dependency**, not done all at once.
Where a live bat uses a flag, that bat must be migrated FIRST.

`feedback_improvements_become_default`: cutover (gauntlet = default + bats
repointed) is the trigger for Phase C/D removals.

---

## Phase A — remove NOW (truly dead / superseded / dangerous; no mode dependency)

| Flag | Why retire | Code-path impact |
|---|---|---|
| `--batched` | Superseded by `--parallel-agents`; **dangerous** — silently drops predictors / feature-cache / input-norm / BC (`project_batched_path_silent_drops`). Used by NO live launcher. | Remove the `batched` branch in `run_cohort` + `train_cluster_batched` / `batched_worker.py` / `cluster_agents_by_arch` + `tests/test_v2_*batched*`. Biggest of the Phase-A removals. |
| `--pbt-r2-size` | Superseded by `--pbt-tier-sizes` | dead `PbtConfig` fields / arg plumbing |
| `--pbt-r3-size` | Superseded by `--pbt-tier-sizes` | " |
| `--pbt-promote-from-r1` | Superseded by `--pbt-promote-counts` | " |
| `--pbt-promote-from-r2` | Superseded by `--pbt-promote-counts` | " |
| `--pbt-freeze-top-r3` | Superseded by `--pbt-freeze-top` | " |

These 6 are safe to drop independent of the cutover (the newer equivalents are
the ones the live tick-tock scripts use).

---

## Phase B — remove WITH `ga` retirement (the gene-only GA + its machinery)

The original pre-PBT path. None used by live launchers.

| Flag | Role |
|---|---|
| `--mutation-rate` | GA-only mutation rate | - DISCUSS do we still use mutation rate for our mutant creation in gauntlet?
| `--monitor-days` | GA monitor-set eval |
| `--monitor-early-stop-patience` | GA monitor early-stop |
| `--monitor-eval-top-k` | GA monitor top-k |
| `--early-stop-patience` | GA generation early-stop |
| `--early-stop-min-gens` | GA early-stop floor |
| `--rotating-eval-sample` | GA rotating eval subsample |
| `--cohort-eval-days` | GA explicit eval-day pool |
| `--n-eval-days` | GA eval-day count (gauntlet uses the fixed validation set) |
| `--training-days-explicit` | GA explicit train-day list (gauntlet derives tranches from the pool) |

Code: the `breeding=="ga"` branch, `_breed_next_generation` (GA crossover),
`_evaluate_agents_on_monitor_days`, `_gen_early_stop_stats`,
`_early_stop_improved`, the rotating-eval logic, and their tests.

---

## Phase C — remove AFTER cutover (lockstep + tier-ladder pbt retired)

Gated on: (1) the Phase 6 A/B passing, (2) gauntlet made the default, (3) the
tick-tock bats migrated off `--breeding pbt`/`lockstep`. **`--breeding lockstep`
stays as the fallback flag for a grace period, then retires.**

| Flag | Why | Note |
|---|---|---|
| `--pbt-rotations` | tier-ladder rotation count | gauntlet derives tranche count from the pool |
| `--pbt-rotation-mode` | random/chronological | gauntlet is always chronological |
| `--pbt-tier-sizes` | tier-ladder sizing | no tiers in the gauntlet |
| `--pbt-promote-counts` | tier-ladder promotion | " |
| `--pbt-freeze-top` | tier-ladder hall-of-fame freeze | gauntlet uses frontier selection |
| `--maturation-gens` | pbt maturation-gate window | tied to the pbt/lockstep maturation path |
| `--resume-from` | lockstep/pbt checkpoint resume | gauntlet resumes via the LEDGER automatically; repoint or drop | - KEEP but discuss
| `--breeding` choices | collapse `{ga,pbt,lockstep,gauntlet}` → `{gauntlet}` (+ `lockstep` during the grace period) | default flips to `gauntlet` |

Code: `breeding in {pbt,lockstep}` branches, `pbt.py` ladder logic, `lockstep.py`,
hall-of-fame / lineage writers tied to them, and their tests.

---

## Phase D — REVIEW individually (gene-duplicated overrides + deferred features)

Not clearly dead — decide per-flag. The gate/BC override flags mirror genes via
the **one-source-of-truth** pattern (cohort-wide pin vs per-agent gene); they're
guarded and intentional. Keep the ones you actually pin cohort-wide; drop the
rest.

| Flag | Consideration |
|---|---|
| `--bc-include-negative-samples` | BC variant; not in any live bat → likely retire |
| `--bc-include-close-hold-samples` | " |
| `--bc-positive-weight` | " |
| `--bc-learning-rate` | overlaps gene `bc_learning_rate` (one-source pin) — keep iff you pin it |- KEEP but discuss, so long as we can still adjust this
| `--bc-target-entropy-warmup-eps` | overlaps gene — keep iff pinned |
| `--per-transition-credit` | cohort flag; not in live bats → likely retire |
| `--direction-head-manifest` | frozen-direction-head path; not in live bats → confirm unused, then retire | - KEEP, I thought we used the direction head, and I thought we used thre frozen one.
| `--strategy-mode` + value/each-way genes | value betting DEFERRED (`feedback_reliability_over_upside`) — keep dormant or retire |
| `--sortino-lambda` | composite component — keep iff used in a composite mode you run |
| `--maturation-bonus-weight` | composite component; NB gauntlet breeder ranks on **locked**, not the scoreboard composite — confirm it still matters before keeping |- KEEP

---

## KEEP — the gauntlet core (~40)

Run/IO: `--n-agents --generations --days --data-dir --device --seed
--output-dir --parallel-agents`.
Day split: `--holdout-recent --validation-holdout-recent --validation-holdout-mode
--exclude-days`.
Predictors (always on): `--use-race-outcome-predictor --use-direction-predictor
--predictor-bundle-manifests --predictor-lean-obs`.
Genes: `--enable-gene --enable-all-genes --seed-gene`.
Tranche/breeding sizing (gauntlet reuses these): `--pbt-train-per-rotation
--pbt-eval-per-rotation --pbt-perturb-frac --survivor-fraction`.
Compute lanes: `--big-model-threads --gpu-policy-lane --gpu-lane-max-concurrent`.
Selection/eval: `--composite-score-mode --argmax-eval
--force-close-rate-penalty-weight`.
Tagging: `--era-id --era-type --hypothesis-id`.
Reward: `--reward-overrides`.
Gate pins (one-source overrides): `--arb-spread-target-lock-pct
--direction-gate-enabled --mature-prob-open-threshold --race-confidence-threshold
--lay-price-max --predictor-p-win-back-threshold
--predictor-p-win-back-max-threshold --predictor-p-win-lay-threshold
--bc-pretrain-steps`.
UI: `--emit-websocket --ws-host --ws-port`.

### Rename opportunity (do at cutover, with hidden aliases)
The gauntlet reuses **pbt-named** flags for tranche sizing / breeding:
`--pbt-train-per-rotation`, `--pbt-eval-per-rotation`, `--pbt-perturb-frac`,
`--survivor-fraction`. Once pbt/lockstep retire, these names mislead. Alias to
`--tranche-train-days`, `--tranche-eval-days`, `--perturb-frac`,
`--keep-fraction` (keep the old spellings as hidden deprecated aliases so the
gauntlet bats don't break). Also: `--era-type` only accepts `tick`/`tock` — add
a gauntlet-friendly value or make it free-text.

---

## Recommended sequence
1. **Now (safe):** Phase A (6 flags) + Phase B (10 flags) once you confirm `ga`
   is dead to you — ~16 flags + their code, zero impact on gauntlet/lockstep/pbt
   runs. Removes the largest cruft (incl. the dangerous `--batched`).
2. **At cutover:** flip default to gauntlet + repoint the bats, then Phase C
   (lockstep+pbt removal, ~7 flags) + the renames. Keep `--breeding lockstep`
   as a grace-period fallback before its final removal.
3. **Per-flag:** walk Phase D and decide each.

Net: **79 → ~40** flags, all of them load-bearing for the gauntlet.
Each removal lands with: arg deleted, dead code + tests removed, any bat using it
migrated, and a one-line note in CLAUDE.md / findings.

---

## EXECUTION PROGRESS (branch `cli-cleanup`, 2026-06-14)

**Removed + verified (kept-path suite green: 103 passed; the 2 predictor-test
failures are PRE-EXISTING — stale `OBS_SCHEMA_VERSION==8` assert, now 9 — and
unrelated to this work):**

- **Chunk 1 — `--batched`** (Phase A): flag + `elif batched:` generation-loop
  branch + `train_cluster_batched`/`cluster_agents_by_arch` imports +
  `_resolve_parallel_agents` mutual-exclusion + gauntlet/pbt/main wiring removed.
  `batched_worker.py` left orphaned (only its own test imports it) for a later
  module sweep — the dangerous flag is gone.
- **Chunk 2 — Phase-D isolated flags** (flag + `main()` wiring; `run_cohort`
  keeps its defaulted params as internal API, so the feature code + its unit
  tests are untouched): `--strategy-mode`, `--sortino-lambda`,
  `--per-transition-credit`, `--bc-include-negative-samples`,
  `--bc-positive-weight`, `--bc-include-close-hold-samples`,
  `--bc-target-entropy-warmup-eps`. Deleted one obsolete test
  (`test_enable_all_genes_collides_with_bc_warmup_flag`) that asserted a removed
  flag's guard.

**KEEP confirmed (your marks):** `--bc-learning-rate`, `--direction-head-manifest`,
`--maturation-bonus-weight`, `--resume-from`.

**Reframe — `ga` retirement is CUTOVER-GATED, not now.** Retiring `ga` means
changing the `run_cohort` default breeding mode (currently `"ga"`), which is part
of the cutover (→ default `gauntlet`) and needs the A/B verdict. So the ga branch
+ its leaf flags (`--mutation-rate`, `--monitor-*`, `--early-stop-*`,
`--rotating-eval-sample`, `--cohort-eval-days`, `--n-eval-days`,
`--training-days-explicit`) move into the **cutover bucket** alongside
lockstep/pbt (Phase C). `--mutation-rate` is GA-only (gauntlet mutation =
`--pbt-perturb-frac`), so it retires with the ga branch.

**Net so far: 8 flags removed (79 → ~71 real options). Remaining removals are all
cutover-gated (ga + lockstep + tier-ladder pbt) → wait for the A/B verdict +
bat migration.**

---

## CUTOVER EXECUTED — behavioural part  (2026-06-16, branch `cutover-gauntlet`)

A/B passed (gauntlet champion £19.32 locked @ σ22.8 vs lockstep £15.83 @ σ22.7;
findings.md Phase 6). Cutover done:
- **Default breeding flipped `ga` → `gauntlet`** (argparse default; `run_cohort`
  param default left `ga` for programmatic/test compat until the ga code is removed).
- **`start_tick.bat` + `start_tock.bat` repointed** to `--breeding gauntlet
  --generations 5` (both already pass `--validation-holdout-recent`). `resume_tock.bat`
  left on lockstep (it's a `--resume-from` helper; gauntlet auto-resumes via the ledger).
- 77 kept-path tests green; lockstep + pbt still fully work as the fallback.

**STILL PENDING — deep ga/pbt CODE removal (Phase B + C):** large + coupled.
Blockers/notes: the `launch_tock_*.sh` / `launch_tick_001.sh` era scripts still use
`--breeding pbt` (must be migrated/retired before pbt code is deleted, per the
"bat migrated FIRST" rule); `test_v2_pbt_runner.py` + the ga-default tests in
`test_v2_cohort_runner.py` need pruning; the ga branch lives inside the 2700-line
`run_cohort`. To be done as a separate staged refactor with the kept-path suite
green at each step. `--breeding lockstep` stays as the grace-period fallback.
