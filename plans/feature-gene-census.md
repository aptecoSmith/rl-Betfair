# Feature / Knob Census — what's actually sampled vs dark (2026-06-05)

**Why this exists:** the `pbt_long` GPU-lane campaign (2026-06-04→05) trained 2
eras / ~7 generations and produced 6 champions, but a post-mortem showed it
explored almost none of the behavioural machinery we've built. The agents bled
naked (held-out naked −250 to −770) because every mechanism that would have
prevented it was at its default. This census enumerates **every knob, whether
it is currently sampleable as a gene, and what value the campaign actually
used** — so we stop burning compute on runs that can't reach the good configs.

## TL;DR

- `CohortGenes` has **42 fields**. The campaign's fresh-blood sampler explored
  **14** of them (6 hyperparams + `hidden_size` + 7 architecture/size genes).
- **20 "PHASE5" genes were pinned to defaults** — they ARE sampleable, but only
  when their name is in `enabled_set`, and the wrapper passed **no
  `--enable-gene` flags**, so `enabled_set` was empty. One-line fix (no code).
- **8 direction/BC knobs are HARD-PINNED** in `_sample_field` — never
  sampleable without code (incl. `direction_gate_enabled`).
- **The direction predictor was never in the obs** (`--use-direction-predictor`
  not passed) and several behavioural levers are **cohort-flags, not genes**
  (force-close, close-walk, the pwin/confidence gates).

Net: we explored hyperparameters + architecture and **nothing else**. Stop-loss,
naked-variance penalty, the direction gate, direction-prob training, all three
aux heads, force-close, close-walk, and value-betting were all dark.

---

## A. SAMPLED — the gauntlet explored these (14) ✅

| gene | range explored |
|---|---|
| learning_rate, entropy_coeff, clip_range, gae_lambda, value_coeff, mini_batch_size | full (legacy hyperparams) |
| architecture | {lstm, transformer} |
| hidden_size | lstm 64–1024 / tf 64–256 |
| transformer_depth, _heads, _ctx_ticks, _ffn_mult, _pos_encoding | full structural sets |
| predictor_lean_obs | {lean, full} |

## B. GENE-READY but PINNED OFF — sampleable with ONE CLI flag, no code (20) ⚠️

These live in `PHASE5_GENE_NAMES` with sampling ranges already wired. They were
pinned because `enabled_set` was empty. Enable per-run with `--enable-gene
<name>` (or enable-all). **Campaign value = the default in parentheses.**

| gene (campaign value) | what it does | matters because |
|---|---|---|
| **stop_loss_pnl_threshold (0.0 = OFF)** | closes a pair when its MTM crosses −X% of stake | direct cut on the naked bleed |
| **naked_variance_penalty_beta (0.0 = OFF)** | penalises per-leg naked variance in reward | teaches them to stop rolling naked dice |
| **direction_prob_loss_weight (0.0 = OFF)** | trains the per-runner direction head on the **direction predictor** | THE predictor "is this trade going bad" signal |
| **bc_direction_target_weight (0.0 = OFF)** | direction-targeted behavioural cloning | bootstraps acting on direction |
| direction_gate_threshold (0.5) | the gate's filter strength (only read when gate ON) | tunes the gate |
| **matured_arb_bonus_weight (0.0 = OFF)** | shaped reward per naturally-matured pair | rewards real scalping over naked |
| **open_cost (0.0 = OFF)** | selective-open shaping (charge per open, refund if it matures) | discourages speculative opens |
| **fill_prob_loss_weight (0.0)** / **mature_prob_loss_weight (0.0)** / **risk_loss_weight (0.0)** | train the three aux heads that feed the actor | all three heads were untrained → constant 0.5 columns |
| arb_spread_target_lock_pct (0.02) | the scalp's target locked-profit %, sets passive width | pinned at 2% — never explored tighter/wider phenotypes |
| mark_to_market_weight (0.05) | per-tick MTM reward densification | on at default; not explored |
| naked_loss_scale (1.0) | scales naked losses in raw reward (annealing) | not explored |
| alpha_lr (0.01) | entropy-controller LR | not explored |
| reward_clip (10.0) | reward clip | not explored |
| predictor_feature_gain (1.0), value_edge_threshold (0.05), value_kelly_fraction (0.25), each_way_edge_threshold (0.05), each_way_kelly_fraction (0.25) | value/each-way betting knobs | **inert in arb mode** — only matter if strategy_mode is value_* |

## C. HARD-PINNED — NOT sampleable without code (8) 🔴

Pinned inside `_sample_field`; only an operator flag/override can move them.
To make sampleable: add a sampling branch (bools → 50/50; floats → add to
`_PHASE5_RANGES`).

| knob (campaign value) | what it does | how to free it |
|---|---|---|
| **direction_gate_enabled (False)** | blocks/closes opens against the predicted price move — **the close signal you remembered** | needs a bool-sampling branch + must pair with `use_direction_predictor` |
| direction_horizon_ticks (60), direction_threshold_ticks (5), direction_force_close_seconds (60) | define the **direction-label** the head/BC learn (horizon, move size, label force-close) | add to `_PHASE5_RANGES` if we want the gauntlet to tune the label |
| direction_gate_warmup_eps (5) | ramps the gate in over N eps | add sampling branch |
| bc_pretrain_steps (gene 0; flag set **500**) | BC pretrain length | currently `--bc-pretrain-steps` cohort flag; make a gene to vary per-agent |
| bc_learning_rate (3e-4), bc_target_entropy_warmup_eps (5) | BC optimiser knobs | add to `_PHASE5_RANGES` |

## D. COHORT-LEVEL FLAGS — same for all agents, not genes at all 🔴

| flag (campaign value) | what it does | should it be a gene? |
|---|---|---|
| **use_direction_predictor (OFF)** | puts the direction predictor's signal **in the obs** | YES — make it structural like `predictor_lean_obs` (bakes both obs variants), OR turn ON cohort-wide. Without it the gate + direction head have no signal. |
| use_race_outcome_predictor (ON) | win-prob predictor in obs | was on; fine |
| **force_close_before_off_seconds (0 = OFF)** | env force-closes unfilled pairs at T−N (safety flatten) | YES — strong candidate gene; bounds naked at deploy |
| **close_walk_ticks (0 = OFF)** | lets a close walk N ticks to complete the hedge | YES — candidate gene; fixes under-hedged closes |
| strategy_mode (arb) | arb vs value_win vs value_each_way | structural; arb is the focus |
| race_confidence_threshold, lay_price_max, mature_prob_open_threshold, predictor_p_win_back/lay_threshold | per-race / per-runner gates | candidate genes (the value-mode ones inert in arb) |

---

## The fix (before the next campaign)

1. **One-line, no code — enable the PHASE5 genes for sampling.** Priority set
   for the naked bleed: `stop_loss_pnl_threshold`, `naked_variance_penalty_beta`,
   `direction_prob_loss_weight`, `bc_direction_target_weight`,
   `matured_arb_bonus_weight`, `open_cost`, the three aux-head weights,
   `arb_spread_target_lock_pct`. Pass via `--enable-gene` (verify the runner
   exposes "enable-all"; if not, add it).
2. **Turn the direction predictor ON** (`--use-direction-predictor`) so the
   direction head/gate actually have a signal — and ideally make it a
   structural gene so the gauntlet compares with/without.
3. **Code — promote the dark levers to sampleable genes:**
   `direction_gate_enabled` (bool), `force_close_before_off_seconds`,
   `close_walk_ticks`, `bc_pretrain_steps`. Add a regression test that a
   fresh-blood draw actually varies each new gene (the gap this census found
   had no such guard).
4. **Add a launch-time assertion / census check**: log, at cohort start, the
   set of genes that are actually being sampled vs pinned, so an empty
   `enabled_set` never silently ships again.
