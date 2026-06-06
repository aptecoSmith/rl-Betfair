# Current state — the bricks that already exist (2026-06-06)

## 1. The phenotype tool — DONE (commit `1e57c7a`)
`tools/phenotype_analysis.py` — `python -m tools.phenotype_analysis --cohort-dir <dir> [--out PATH]`.
- Reads a cohort's `model_register.csv` (auto-fallback to `scoreboard.jsonl`;
  cross-checked byte-identical) — a flat per-agent table of the full gene vector
  (`gene_*`) + outcome counts (`pairs_opened`, `arbs_completed/closed/naked/
  force_closed/stop_closed`, `locked_pnl`, `naked_std`, `day_pnl`, `bet_count`).
- Derives per-agent behaviour RATES off `pairs_opened` (the 5 buckets sum to it),
  plus locked_pnl + naked_std.
- Correlates (Pearson **and** Spearman, with p-values, n) every *varied* gene
  against every behaviour. Skips pinned/zero-variance genes (flags them).
- Saves a timestamped `phenotype_analysis_<ts>.md` (top drivers per behaviour +
  a proposed combined recipe + caveats) and `phenotype_corr_<ts>.csv` (full
  gene×behaviour matrix). Read-only on the cohort; reports are gitignored (data).

## 2. First analysis result — pbt_genes_v2, n=48 (gens 0-2)
The two desirable champion phenotypes both trace to the **direction machinery**:
- **close_rate** ← `direction_gate_enabled` (Spearman ρ = **+0.63**, dominant).
- **maturation_rate** ← `use_direction_predictor` (+0.38) + `bc_learning_rate`
  (+0.52) + `direction_prob_loss_weight` (+0.40).
- bail (stop-close) ← `stop_loss_pnl_threshold` (ρ = −0.91).

**Proposed combined-recipe HYPOTHESIS** (raise maturation + close, cut bail):
`direction_gate_enabled→1`, `use_direction_predictor→1`, higher `bc_learning_rate`,
`stop_loss_pnl_threshold→~0.22`, `bc_pretrain_steps→0`.

**Load-bearing caveats** (this is a *candidate to test*, not proof): n=48 with
uncorrected multiple comparisons; the four rates are **compositional** (sum to ~1,
so correlations are partly mechanical — "drive maturation" partly just means
"shift the mix"); correlation≠causation; architecture-as-gene confound.
Report: `registry/pbt_genes_v2/phenotype_analysis_20260606_1852.md`.

## 3. The running campaign IS the first Tick
`registry/pbt_genes_v2` — a full-width `--enable-all-genes` era (the reward-redesign
campaign). It is a Tick (wide exploration). Launched via the wrapper
`plans/pbt-gpu-forward/_scripts/run_genes_campaign.ps1` (`-Eras N`) / the root
`start_pbt_training.bat`; stopped via `stop_pbt_loop.bat`.

## 4. The PBT structure (so the Tock mechanics make sense)
- **Era** = one seed = fresh blood; runs N generations (5); shares the
  hall-of-fame. The `.bat`'s `ERAS` counts eras.
- **Generation** = one evolutionary time-step ("a season"). Each gen the whole
  16-agent population trains+evals (each agent on *its tier's* rotation day-set),
  then: R3 top-2 freeze as champions; winners promote up a tier; the bottom is
  refilled with fresh blood + perturbed offspring.
- **R1/R2/R3 = promotion TIERS** (rookie / mid / top divisions), NOT three stages
  inside one gen. An agent's `tier` (in its pbt spec) picks which rotation's days
  it sees. Gen 0 = all 16 in R1 (fresh); the pipeline *fills* over gens (R3 only
  populates once agents survive ~2 promotions — that is why no champions appear
  until ~gen 3). Steady-state split ≈ R1=6 / R2=6 / R3=4 of 16.
- Code: gen loop `training_v2/cohort/runner.py::run_cohort` (~line 1305);
  tier→rotation days (~1341); `init_pbt_population` / `breed_pbt` / `make_rotations`
  imported from the pbt module. **A Tock = seed R1's fresh blood with narrow/pinned
  genes** (the rookie-injection path is where a hypothesis recipe enters).

## 5. Reward context (why the genes are what they are)
A 2026-06-06 reward redesign (commit `8f48fdf`) fixed a "spray-and-bail" pathology
(agents opened ~249 pairs/race, only 2.4% matured, 74% force-closed). Changes:
promoted 4 gate flags to genes (`mature_prob_open_threshold`, `race_confidence_
threshold`, `lay_price_max`, pwin back/lay); `force_close` pinned 0 in training
(train-naked, keep the naked-variance signal; deploy-eval uses 120); a
`--force-close-rate-penalty-weight` composite penalty; `open_cost`→[0,4];
`mark_to_market_weight`→0.2; BC switched to a **maturation-conditioned** oracle.
The full gene inventory is `genes_census.md` (repo root): 50 fields,
`--enable-all-genes` samples 28. The new reward's first champions DID fix the
structure (force-close 74%→0-33%, maturation 2.4%→20%) but **P&L + naked variance
still need work** — the reason for the directed Tick-Tock search.
