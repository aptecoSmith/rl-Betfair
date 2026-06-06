# PBT Gene Census — post 2026-06-06 reward redesign (commit `8f48fdf`)

**Scope.** A knob-by-knob audit of the PBT gene system as it stands on branch
`pbt-gpu-forward` at `HEAD = 8f48fdf` ("spray-and-bail redesign"). Verified
directly against the code — `training_v2/cohort/genes.py` (the `CohortGenes`
dataclass, `PHASE5_GENE_*`, `_PHASE5_RANGES`, the three samplers,
`assert_in_range`, `to_dict`), `training_v2/cohort/runner.py` (the
`--enable-all-genes` block + per-knob override guards + the force-close-rate
composite penalty), and `training_v2/cohort/worker.py` (predictor-coupled gate
pinning). Range-sanity is cross-checked against CLAUDE.md and the user memory;
sources are cited per row. **The code is ground truth** — `plans/feature-gene-
census.md` (2026-06-05) was the starting map but predates this redesign, so its
A/B/C/D buckets have shifted.

## Categories used

- **legacy-7** — the Phase-3 PPO/size genes. ALWAYS sampled (base GA + fresh
  blood), never gated by `enabled_set`.
- **architecture-structural** — sampled ONLY at fresh-blood birth
  (`sample_fresh_blood_genes` via `_sample_architecture_field`), then FROZEN for
  the lineage (warm-start weight inheritance needs matching shapes, HC#10). In
  `ARCHITECTURE_GENE_NAMES`; the base `sample_genes` pins them.
- **PHASE5** — in `PHASE5_GENE_NAMES`. Sampled per-agent **iff** the name is in
  the cohort's `enabled_set` (`--enable-gene NAME` or `--enable-all-genes`),
  otherwise pinned to `PHASE5_GENE_DEFAULTS[name]` (byte-identical when disabled).
- **hard-pinned (`_sample_field`)** — returns a constant in `_sample_field`;
  NOT a sampled gene on any path. After this redesign only **one** field remains
  here.
- **cohort-flag-not-a-gene** — an argparse flag on the runner, identical for
  every agent, not a `CohortGenes` field at all.

`CohortGenes` has **50 fields**, partitioned cleanly into exactly three
categories with NO overlap (verified by enumerating the dataclass): **7 legacy**
+ **12 architecture-structural** (`ARCHITECTURE_GENE_NAMES`) + **31 PHASE5**
(`PHASE5_GENE_NAMES`) = 50. Field-count and category tallies are in the closing
section.

---

## 1. Legacy-7 (always sampled)

| Gene | Sampled by fresh blood? | Range + distribution | Range sanity |
|---|---|---|---|
| `learning_rate` | always | `[1e-5, 1e-3]` **log-uniform** | Sensible; standard PPO LR span (an order of magnitude → log-uniform is correct). |
| `entropy_coeff` | always | `[1e-4, 1e-1]` **log-uniform** | OK. NB the target-entropy controller (CLAUDE.md) treats the live coefficient as a learned variable clamped `[log(1e-5), log(0.1)]`; this gene is the init. Fine. |
| `clip_range` | always | `[0.1, 0.3]` uniform | Standard PPO clip band. Fine. |
| `gae_lambda` | always | `[0.9, 0.99]` uniform | Standard. Fine. |
| `value_coeff` | always | `[0.25, 1.0]` uniform | Fine. |
| `mini_batch_size` | always | choice `{32, 64, 128}` | Fine. CLAUDE.md per-mini-batch KL math assumes `mini_batch_size=64`; 32/128 still valid. |
| `hidden_size` | always (fresh blood conditions on arch: LSTM `{64..1024}`, transformer `{64,128,256}`) | base GA: choice `{64,128,256}`; `assert_in_range` accepts `{64,128,256,512,1024}` | Sensible. Doubles as transformer `d_model`; every `(hidden, n_heads)` pair divides. Widened to 1024 for fresh-blood LSTMs (2026-06-04, "bigger LSTMs ranked higher in gen-0 R1"). **Structural under warm-start** — handled in legacy code, not in `ARCHITECTURE_GENE_NAMES`, but in practice frozen per lineage like the architecture genes. |

---

## 2. Architecture-structural (fresh-blood birth only, frozen per lineage)

All in `ARCHITECTURE_GENE_NAMES`. Drawn by `sample_fresh_blood_genes`; the base
`sample_genes`/GA pins them to the LSTM/default value, so a gene-only launch is
byte-identical. The **SAMPLE** set (fresh-blood draw) can be narrower than the
**CHOICES/VALID** set (warm-load acceptance) — noted where they differ.

| Gene | Fresh-blood sample set | Valid (warm-load) set | Range sanity |
|---|---|---|---|
| `architecture` | `{lstm, transformer}` ~50/50 | same | Fine — the architecture tournament's entry point. |
| `transformer_depth` | `{1,2,3,4,6}` | same | Fine. |
| `transformer_heads` | `{2,4,8}` | same | Fine (`d_model % heads == 0` always holds for the sample sets). |
| `transformer_ctx_ticks` | `{32,64,128}` | `{32,64,128,256}` | **Re-capped to ≤128 for fresh blood** (2026-06-05): ctx256/d512 transformers are sequential-batch=1-rollout stragglers (~1.4h/agent, gated whole gens) and never out-championed ctx≤64. 256 stays VALID so a warm-loaded big champion still loads. Sensible given the GPU-lane NO-GO finding (memory `project_gpu_speedup_decision`). |
| `transformer_ffn_mult` | `{2,4}` | same | Fine. `dim_feedforward = mult × d_model`; 2× is the narrow default, 4× the standard width. Structural. |
| `transformer_pos_encoding` | `{learned, rope}` | same | Fine. RoPE (rotary/relative) vs learned-absolute; both live (pbt-gpu-forward task #8), disjoint backbone weights → distinct arch-hash → frozen. |
| `predictor_lean_obs` | `{False, True}` ~50/50 | bool | Sensible — lets the gauntlet compare the lean predictor obs (~370-d, well-scaled) vs full obs (~2254-d, needs BC input-norm; memory `feedback_full_obs_needs_input_norm`). Structural (changes obs_dim). |
| `use_direction_predictor` | `{False, True}` ~50/50 | bool | Sensible. Structural (adds live per-tick direction-predictor features → obs_dim). **Also has a cohort flag `--use-direction-predictor`** — see note ‡ below. |
| `direction_gate_enabled` | coupled: `True` only when `use_direction_predictor=True`, else `False` (then ~50/50) | bool | Correct coupling — the env raises if the gate is on without the predictor signal (`assert_in_range` enforces the invariant). Good. |
| `force_close_before_off_seconds` | `{0.0}` only (`FORCE_CLOSE_BEFORE_OFF_SAMPLE`) | `>= 0` | **⚠️ DELIBERATELY pinned to 0 in training** — see the dedicated note below. Was `{0.0, 60.0, 120.0}` pre-redesign. Sensible per memory `project_force_close_train_vs_deploy`. |
| `close_walk_ticks` | `{0,5,10}` | `>= 0` | Sensible — bounded close-path walk (CLAUDE.md "sanctioned exception: bounded walk on the CLOSE path"). 10 is the operator-pinned value from recipe-expansion. |
| `bc_pretrain_steps` | `{0, 500}` (`BC_PRETRAIN_STEPS_SAMPLE`) | int `>= 0` | Sensible. Base sampler hard-pins to `0`; fresh blood draws `{0,500}`. (The two BC *optimiser* knobs `bc_learning_rate`/`bc_target_entropy_warmup_eps` are now separate PHASE5 genes — §3.) |

---

## 3. PHASE5 genes (sampled iff in `enabled_set`)

In `PHASE5_GENE_NAMES`. Disabled → pinned to `PHASE5_GENE_DEFAULTS`. Enabled via
`--enable-gene NAME` (per knob) or `--enable-all-genes` (whole set, **minus the
3 direction-label-stem knobs**, see §4). Distribution is **uniform** unless
marked log-uniform (`_LOG_UNIFORM_FLOATS`) or int (`_PHASE5_INT_GENES`).

### 3a. Reward-shaping / training-signal PHASE5 genes

| Gene | Range + dist | Default (disabled) | Range sanity |
|---|---|---|---|
| `open_cost` | `[0.0, 4.0]` **high-biased** (`u**(1/2)`, median ≈3.0) | 0.0 | **Range WIDENED `[0,2]→[0,4]` + biased high (2026-06-06).** Sensible *given the redesign*: the new fc-rate composite penalty (§4) makes the toll bite, so open_cost needs headroom + a high prior. The env clamp is widened to `[0,4]` to match (verified `env/betfair_env.py:1734`). The "collapse above 2.0" risk (memory/CLAUDE.md selective-open-shaping) was under UNRELATED penalty genes with no positive offset; matured-arb + MTM provide that offset here, so the wider range is defensible — **but flagged as the most aggressive single change** and worth watching for bet_count→0 collapse. |
| `matured_arb_bonus_weight` | `[0.0, 5.0]` uniform | 0.0 | Sensible (CLAUDE.md matured-arb-bonus). Counts natural maturation only since the 2026-05-23 scope-narrowing. |
| `mark_to_market_weight` | `[0.05, 0.5]` uniform | **0.2** | **Range + default RAISED (2026-06-06): default `0.05→0.2`, range `[0,0.10]→[0.05,0.5]`.** Sensible per `plans/reward-densification`: at 0.05 the per-tick MTM gradient at the open decision is too weak vs value-function noise. NB the LIVE cohort default is still 0.0 (worker sets `reward.mark_to_market_weight=0.0`); the 0.2 only surfaces when the gene is enabled. Note the **floor is 0.05, not 0.0** — an enabled MTM gene can never sample "off". That's intentional (if you enabled it you want it on) but means MTM=0 is only reachable by NOT enabling the gene. |
| `naked_loss_scale` | `[0.0, 1.0]` uniform | 1.0 | Sensible (annealing knob; 1.0 = full cash cost). |
| `stop_loss_pnl_threshold` | `[0.0, 0.30]` uniform | 0.0 | Sensible — closes a pair when MTM crosses −X% of stake. Direct cut on naked bleed. |
| `naked_variance_penalty_beta` | `[0.0, 0.10]` uniform | 0.0 | Sensible; upper bound matches env `NAKED_VARIANCE_PENALTY_BETA_MAX`. **⚠️ minor doc drift:** the dataclass field comment says "Range [0, 0.005]" but the actual `_RANGE` constant and `_PHASE5_RANGES` use `[0.0, 0.10]`. The code (0.10) is authoritative; the inline comment is stale. |
| `reward_clip` | `[1.0, 10.0]` uniform | 10.0 | Sensible. |
| `alpha_lr` | `[1e-2, 1e-1]` **log-uniform** | 1e-2 | Sensible — entropy-controller SGD LR; CLAUDE.md promoted it to a gene with exactly this range. |

### 3b. Scalping / arb mechanism PHASE5 genes

| Gene | Range + dist | Default | Range sanity |
|---|---|---|---|
| `arb_spread_target_lock_pct` | `[0.005, 0.05]` uniform | 0.02 | Sensible — the phenotype handle (0.5% fill-seeker → 5% profit-seeker). Upper bound 0.05 because higher targets are commonly unreachable within `MAX_ARB_TICKS` (CLAUDE.md price-adaptive arb_spread). |

### 3c. Aux-head loss-weight PHASE5 genes

| Gene | Range + dist | Default | Range sanity |
|---|---|---|---|
| `fill_prob_loss_weight` | `[0.0, 0.30]` uniform | 0.0 | Sensible (CLAUDE.md fill-prob-in-actor). |
| `mature_prob_loss_weight` | `[1.0, 5.0]` uniform | 0.0 | Sensible — **range floor is 1.0 but default is 0.0** (intentional: disabled head gets no gradient → benign constant-0.5 column; enabled it trains hard). `assert_in_range` explicitly accepts the 0.0 default outside the range. CLAUDE.md raised this `[0,0.30]→[1.0,5.0]` (2026-05-04) on the "head not training under a small weight" diagnosis. |
| `risk_loss_weight` | `[0.0, 0.30]` uniform | 0.0 | Sensible (CLAUDE.md v2 aux-head port). |

### 3d. Direction-mechanism PHASE5 genes

| Gene | Range + dist | Default | Range sanity |
|---|---|---|---|
| `direction_gate_threshold` | `[0.20, 0.50]` uniform | 0.5 | Sensible — **recalibrated** `[0.5,0.95]→[0.20,0.50]` (2026-05-25) for the unweighted shared C11 head whose outputs mean ~0.26/max ~0.84. 0.20 ≈ "refuse ~95% of opens", 0.50 ≈ "refuse ~30%". Default 0.5 is the no-op floor (unread when gate off). Good. |
| `direction_prob_loss_weight` | `[0.1, 2.0]` uniform | 0.0 | Sensible — floor 0.1 = 2× the previously-broken 0.05 pin (lowest "actually trains"); ceiling 2.0 mirrors mature_prob mid-range. Default 0.0 outside range (benign). |
| `bc_direction_target_weight` | `[0.1, 0.5]` uniform | 0.0 | Sensible — BC blended loss `(1-w)·oracle + w·direction`; ceiling 0.5 keeps oracle dominant. Default 0.0 outside range. |
| `direction_gate_warmup_eps` | `[0, 20]` **uniform int** | 5 | Sensible (mirrors `bc_target_entropy_warmup_eps`). Promoted to PHASE5 2026-06-06. |

### 3e. BC-optimiser PHASE5 genes (promoted 2026-06-06)

| Gene | Range + dist | Default | Range sanity |
|---|---|---|---|
| `bc_learning_rate` | `[1e-5, 1e-3]` **log-uniform** | 3e-4 | Sensible — joins the LR family (log-uniform, like `learning_rate`/`alpha_lr`). Guarded against `--bc-learning-rate` (one source of truth). |
| `bc_target_entropy_warmup_eps` | `[0, 20]` **uniform int** | 5 | Sensible (BC→PPO entropy handshake window). |

### 3f. Direction-LABEL-STEM PHASE5 genes — sampleable but EXCLUDED from `--enable-all-genes`

These ARE PHASE5 genes (in `PHASE5_GENE_NAMES`, in `_PHASE5_RANGES`) and CAN be
enabled with an **explicit** `--enable-gene NAME`, but `--enable-all-genes`
deliberately excludes them via `_LABEL_STEM_PINNED` (§4).

| Gene | Range + dist | Default | Range sanity + why excluded from enable-all |
|---|---|---|---|
| `direction_horizon_ticks` | `[20, 120]` **uniform int** | 60 | Range OK. ⛔ from `--enable-all-genes`: defines the OFFLINE direction-label cache stem (`date\|horizon\|threshold\|force_close\|max_runners`); a per-agent draw needs a pre-scanned cache per distinct triple (combinatorial) and the trainer RAISES `FileNotFoundError` if `direction_prob_loss_weight>0` and the triple's cache is missing. 60 is the one triple that's pre-scanned. |
| `direction_threshold_ticks` | `[2, 10]` **uniform int** | 5 | Same — label-stem knob. ⛔ from enable-all. |
| `direction_force_close_seconds` | `[30.0, 180.0]` uniform | 60.0 | Same — label-stem knob (this is the *label*-definition force-close, distinct from the env's deploy-safety `force_close_before_off_seconds`). ⛔ from enable-all. |

### 3g. Selectivity-gate PHASE5 genes (promoted 2026-06-06, ex-category-D flags)

Previously cohort flags (`plans/feature-gene-census.md §D`); now per-agent PHASE5
genes — the mechanical cut on the campaign's 74% force-close bail. Each default
== the env/policy ctor default == disabled (byte-identical). **Four of the five
require the predictor**: `worker.py::_resolve_gate_genes` pins
`_GATE_GENE_PREDICTOR_REQUIRED = {race_confidence_threshold, lay_price_max,
predictor_p_win_back_threshold, predictor_p_win_lay_threshold}` to their disabled
default when the predictor is absent (the env raises otherwise), so a
predictor-less `--enable-all-genes` can't crash a worker on a non-disabled draw.

| Gene | Range + dist | Default (disabled) | Predictor-coupled? | Range sanity |
|---|---|---|---|---|
| `mature_prob_open_threshold` | `[0.0, 0.5]` uniform | 0.0 | **No** (POLICY-side — reads the agent's own `mature_prob_head`; always free to sample) | Sensible — the keystone selectivity cut. 0.0 = no gate; 0.5 = "refuse opens the head thinks are coin-flips-or-worse". Higher would starve PPO (cf. direction-gate `[0.20,0.50]` precedent). |
| `race_confidence_threshold` | `[0.0, 0.5]` uniform | 0.0 | **Yes** | Sensible — memory `project_race_confidence_gate`: 0=disabled, example live 0.30 ("skip races with no favourite"); a p_win=0.55 favourite still passes at 0.5. Higher = unreachable on even fields. |
| `lay_price_max` | sample `[10.0, 50.0]` uniform; **0.0 reachable** as disabled default | 0.0 (= NO CAP) | **Yes** | Sensible — memory `project_lay_ev_calibration_findings`: `lay_price_max=20` landed (dropped the 20-50 −£0.39/£ leverage trap). The `[10,50]` bracket centres on 20; **0.0 is the special "disabled/no-cap" value, only reachable by NOT enabling the gene** (the sample range starts at 10). Correctly handled — but note an enabled gene can never sample "no cap". |
| `predictor_p_win_back_threshold` | `[0.15, 0.40]` uniform | 0.0 (= no gate) | **Yes** | Sensible — same findings: back p_win 0.30-0.35 peaks +£9.49/pair, 0.40-0.50 is −£0.19/pair, so `[0.15,0.40]` admits the +EV back region. Disabled 0.0 outside range (reachable only when disabled). |
| `predictor_p_win_lay_threshold` | `[0.15, 0.45]` uniform | 1.0 (= no gate) | **Yes** | Sensible — findings moved it `0.40→0.20` (removes the p_win 0.20-0.30 calibration hole); bracket spans both. Disabled 1.0 outside range (reachable only when disabled). |

### 3h. Predictor-integration / value-betting PHASE5 genes (inert in arb mode)

Present so cross-mode breeding stays trivial; the env reads them only when
`strategy_mode` is the relevant `value_*`. In arb mode (the default focus) they
are inert.

| Gene | Range + dist | Default | Range sanity |
|---|---|---|---|
| `predictor_feature_gain` | `[0.0, 1.0]` uniform | 1.0 | Cross-mode (scales predictor obs columns into actor_input). Range fine. |
| `value_edge_threshold` | `[0.02, 0.10]` uniform | 0.05 | `value_win` only. Manifest recommends 0.05. Fine. |
| `value_kelly_fraction` | `[0.0, 1.0]` uniform | 0.25 | `value_win` only. Fine. |
| `each_way_edge_threshold` | `[0.02, 0.10]` uniform | 0.05 | `value_each_way` only. Fine. |
| `each_way_kelly_fraction` | `[0.0, 1.0]` uniform | 0.25 | `value_each_way` only. Fine. |

---

## 4. What the 2026-06-06 redesign changed (commit `8f48fdf`)

The "spray-and-bail redesign" attacks the full-gene campaign's failure mode:
the GA's composite score was BLIND to a 74% force-close bail (open ~249/race),
because force-closed pairs are excluded from the maturation bonus and net ~zero
raw P&L via the env's relaxed-matcher flatten — so a high-volume sprayer scored
as well as a selective scalper. Five coordinated changes:

1. **4 gates → 5 genes (category-D flags → PHASE5).** The selectivity gates
   (`mature_prob_open_threshold`, `race_confidence_threshold`, `lay_price_max`,
   and the back+lay `predictor_p_win_*_threshold` pair = "four gates, five
   names") were promoted from cohort flags to per-agent PHASE5 genes (§3g), so
   the gauntlet can evolve a selectivity phenotype. Predictor-coupled four are
   pinned off without the predictor (`worker.py::_resolve_gate_genes`).

2. **`force_close_before_off_seconds` → pinned `{0.0}` in training.**
   `FORCE_CLOSE_BEFORE_OFF_SAMPLE` was `{0.0, 60.0, 120.0}`, now `{0.0}`. Train
   NAKED so the agent FEELS the full naked-variance cost and learns SELECTIVITY
   (mature_prob gate, open_cost, close_signal) — per memory
   `project_force_close_train_vs_deploy`. A training-time safety net is exactly
   what let the campaign reach the spray-and-bail equilibrium (the env flattened
   the bail for free). **Deploy is unaffected**: `tools/reevaluate_cohort.py`
   sets `force_close_before_off_seconds=120` via `--reward-overrides`, threaded
   through `cohort_overrides` (NOT the gene) and `setdefault`-preserved.

3. **Force-close-rate composite penalty.** `_force_close_rate_penalty` subtracts
   `weight × (arbs_force_closed / max(1, pairs_opened))` from EVERY composite-
   score mode (additive, mode-agnostic), so spray-and-bail is DESELECTED at the
   GA step. Default weight **0.0 = byte-identical** (`FORCE_CLOSE_RATE_PENALTY_
   DEFAULT_WEIGHT`); set via `--force-close-rate-penalty-weight` (module-level
   `_FORCE_CLOSE_RATE_PENALTY_WEIGHT`, same plumbing pattern as `_SORTINO_LAMBDA`).

4. **`open_cost` range `[0,2]→[0,4]` + high-biased prior.** Median draw ≈3.0
   (`_sample_high_biased`, `power=2`, single rng draw so the stream isn't
   shifted). 0.0 stays reachable (u=0 ⇒ lo). Env clamp widened to `[0,4]`
   (`env/betfair_env.py:1734`) to match. Gives the toll headroom to actually
   bite now the composite deselects spray.

5. **`mark_to_market_weight` default `0.05→0.2`, range `[0,0.10]→[0.05,0.5]`.**
   Stronger per-tick gradient at the open decision so PPO can credit "this open
   will go bad" against value-function noise.

(The earlier 2026-06-06 commit `dcce671`/`4741931` promoted the 6 ex-hard-pinned
knobs — direction-label triple, gate warmup, BC lr + warmup — to PHASE5 and
added `--enable-all-genes` with the `_LABEL_STEM_PINNED` exclusion of the 3
label-stem knobs. That promotion is folded into this census's PHASE5 tallies.)

### `--enable-all-genes` semantics
Unions `frozenset(PHASE5_GENE_NAMES) − _LABEL_STEM_PINNED` onto the explicit
`--enable-gene` set. `_LABEL_STEM_PINNED = {direction_horizon_ticks,
direction_threshold_ticks, direction_force_close_seconds}` stays pinned to
60/5/60 (the one pre-scanned label triple). Per-knob one-source-of-truth guards
RAISE if a cohort flag (`--bc-learning-rate`, `--bc-target-entropy-warmup-eps`,
`--arb-spread-target-lock-pct`, and the five `_GATE_FLAG_GUARDS`) is set to a
non-disabled value while its gene is in `enabled_set`.

---

## 5. NOT genes — cohort-level flags that REMAIN flags (category-D survivors) ⛔

These are argparse flags on the runner, identical for every agent, **never
sampled**. They were category-D in the prior census and stay flags after the
redesign:

| Flag | Why it's NOT a gene |
|---|---|
| ⛔ `--strategy-mode` (`arb` / `value_win` / `value_each_way`) | Structural cohort-wide regime. Determines which genes the env even reads (arb vs the value/each-way PHASE5 genes). Not a `CohortGenes` field; arb is the focus. |
| ⛔ `--use-race-outcome-predictor` (+ `--predictor-bundle-manifests`) | Cohort-wide ON/OFF for the win-prob predictor in obs + the predictor bundle. Not a per-agent field. Gates whether the four predictor-coupled gate genes (§3g) can be active at all. |
| ‡ `--use-direction-predictor` | **Dual nature.** Exists BOTH as a cohort argparse flag AND as a fresh-blood-sampled architecture-structural gene (`use_direction_predictor`, §2). The flag turns it on cohort-wide; the gene lets fresh blood draw it ~50/50 per lineage. So unlike `--strategy-mode`/`--use-race-outcome-predictor`, this one IS a gene — it just also has a cohort-pin flag. |

### NOT a gene — hard-pinned in `_sample_field` ⛔

After this redesign, exactly **one** field is still a pure hard-pin on the base
sampler with no PHASE5/architecture sampling path:

| Field | Status |
|---|---|
| ⛔ `bc_pretrain_steps` (base sampler) | `_sample_field` returns `0`. BUT it IS fresh-blood-sampled `{0,500}` as an architecture-structural gene (§2), and the operator can pin it cohort-wide via `--bc-pretrain-steps`. So "hard-pinned" applies only to the base GA path; on the fresh-blood path it's a real gene. |

Net: there is **no longer any field that is dark on every path**. The prior
census's category-C (8 hard-pinned) and category-D gate flags have all been
promoted; the only true cohort-wide-only knobs left are `--strategy-mode` and
`--use-race-outcome-predictor` (both genuinely structural regime switches).

---

## 6. Tallies & flagged items

**Gene count by category** (the 50 `CohortGenes` fields partition cleanly —
7 + 12 + 31 = 50, no field in two categories):

- **legacy-7** (always sampled): **7** — learning_rate, entropy_coeff,
  clip_range, gae_lambda, value_coeff, mini_batch_size, hidden_size.
- **architecture-structural** (fresh-blood birth, frozen): **12** names in
  `ARCHITECTURE_GENE_NAMES` — architecture, transformer_depth, transformer_heads,
  transformer_ctx_ticks, transformer_ffn_mult, transformer_pos_encoding,
  predictor_lean_obs, use_direction_predictor, direction_gate_enabled,
  force_close_before_off_seconds, close_walk_ticks, bc_pretrain_steps.
  *(Note: it is `direction_gate_enabled` (the bool) that is structural-only; its
  companion `direction_gate_threshold` is the PHASE5 gene. `bc_pretrain_steps`
  is structural here AND the base-sampler hard-pin (§5) — one dataclass field,
  counted once in its structural home.)*
- **PHASE5** (`enabled_set`-gated): **31** names in `PHASE5_GENE_NAMES` /
  `PHASE5_GENE_DEFAULTS`. Of these, **3** (direction-label stem) are excluded
  from `--enable-all-genes` (explicit `--enable-gene` only), **5** are the new
  selectivity gates (4 predictor-coupled), and **5** are value-mode genes inert
  in arb.
- **hard-pinned-only**: **0** fields dark on every path (was 8 pre-redesign).
- **cohort-flag-not-a-gene**: **2** true survivors (`--strategy-mode`,
  `--use-race-outcome-predictor`) + 1 dual flag/gene (`--use-direction-predictor`).

`--enable-all-genes` therefore samples **31 − 3 = 28** PHASE5 genes plus the 7
legacy + the architecture-structural set drawn at fresh-blood birth.

**Ranges flagged as questionable / worth watching:**

1. **`open_cost` `[0.0, 4.0]` high-biased (median ≈3.0)** — the single most
   aggressive change. The "collapse above 2.0 → bet_count=0" risk noted in
   CLAUDE.md (selective-open-shaping) was under unrelated penalty genes; here
   matured-arb + MTM offset it, so it's defensible, but a high-biased prior
   centred at 3.0 on a knob with a known collapse mode warrants a guard /
   watch on bet_count in early generations.

2. **`naked_variance_penalty_beta` doc drift** — the dataclass field comment
   says "Range [0, 0.005]" but the actual range constant + `_PHASE5_RANGES` use
   `[0.0, 0.10]` (20× wider). The code (0.10) is authoritative and matches the
   env `NAKED_VARIANCE_PENALTY_BETA_MAX`; the inline comment is stale and should
   be corrected to avoid confusion. **Not a behavioural bug** — purely a
   comment/code mismatch.

3. **"Floors above the disabled default" pattern** (informational, all
   intentional): `mark_to_market_weight` (floor 0.05, off only via disable),
   `mature_prob_loss_weight` (floor 1.0, default 0.0), `lay_price_max` (sample
   `[10,50]`, no-cap 0.0 only via disable), `direction_prob_loss_weight`/
   `bc_direction_target_weight` (floor 0.1, default 0.0), and the two
   `predictor_p_win_*_threshold` (disabled defaults 0.0/1.0 sit outside their
   sample brackets). In every case the "off" value is reachable only by NOT
   enabling the gene; an *enabled* gene always samples an active value.
   `assert_in_range` explicitly whitelists the disabled default so these don't
   trip validation. Consistent and deliberate — noted so a future reader isn't
   surprised that enabling these genes removes the "off" setting from the draw.

All ranges otherwise track prior-experience evidence with cited sources
(force-close train/deploy asymmetry, lay_price_max≈20, pwin lay 0.20-0.45,
direction-gate recalibration, MTM densification). No range contradicts the
memory.
