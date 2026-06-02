# Session prompt — full-features scalper cohort handoff

Self-contained brief for a fresh agent picking up the scalper cohort
work where the previous session paused on 2026-05-23.

---

## TL;DR — what you're doing

1. **A pre-flight probe is currently running** (background, ~35 min wall):
   `registry/_predictor_SCALPING_full_features_probe2_1779576580/`
   — 3 agents × 1 generation with ALL the auxiliary head losses + BC
   pretrain + direction gate + matured-only bonus turned on. Confirms
   the integrated configuration doesn't crash before committing 13h
   to a full multi-generation cohort.

2. **If the probe succeeds**, scan oracle + direction caches for the
   remaining 8 training days (~5 min), then launch the full cohort
   (`6 agents × 5 generations`, ~13h wall).

3. **If the probe fails**, diagnose, fix, re-run probe before the
   real cohort.

The operator's goal: a multi-generation training run with every
feature we built actually turned on, to see whether the GA can push
natural-maturation rate (`mat%`) from the current ~5-10 % toward the
estimated ~15-25 % practical ceiling.

---

## Context: what happened today (2026-05-23)

The operator and the previous session spent ~12 hours debugging the
scalping pipeline. The high-level arc:

1. **Investigation** of why agents weren't scalping (`plans/force_close_and_arb_spread/findings.md`):
   - Original cohort had 76 % aggregate force-close rate
   - Discovered the auto-paired passive was always 20 ticks away
     regardless of price (hardcoded in `agents_v2/env_shim.py`)
2. **Price-adaptive arb_spread redesign** (commits `ae0d38d` → `438cc99`
   → `d2cc1a8`):
   - First: introduced `arb_spread_headroom_ticks` gene
   - Then: fixed `min_arb_ticks_for_profit` to use equal-profit sizing
     (was equal-exposure → over-stated commission floor by 2-5×)
   - Then: replaced headroom+scale pair with single
     `arb_spread_target_lock_pct` gene that directly expresses
     "what % of stake do I want locked per pair"
3. **Reward shape narrowing** (commit `5d57a91`):
   - `CLOSE_SIGNAL_BONUS` zeroed (was rewarding loss-closes)
   - `matured_arb_bonus_weight` narrowed to count natural maturation
     only (was counting agent_closed too — same conflation as the
     old `mr` metric)
4. **Whitelist bug audit** (commits `6db33db`, `cf82431`):
   - Found 3 reward override keys silently dropped at config merge:
     `matured_arb_expected_random`, `matured_arb_bonus_cap`,
     `naked_variance_penalty_beta`. The `naked_variance_penalty_beta`
     Phase-5 gene had been **silently inert since it landed**.
   - Added a regression test
     (`tests/test_forced_arbitrage.py::TestRewardOverridesWhitelistCovers_RewardCfgReads`)
     that parses env source for `reward_cfg.get(...)` keys and asserts
     each is whitelisted — catches future drift at test time.
5. **Oracle cache lean-obs fix** (commit `88c8dde`):
   - `oracle_cli scan` was building env with full 143-key obs
   - Cohorts use `--predictor-lean-obs` (23-key obs)
   - BC pretrain refused to load the wide-obs cache
   - Added `--predictor-lean-obs` flag to oracle_cli + passthrough
     into `arb_oracle.scan_day`
6. **Six 3-agent probes** with progressively more correct
   configurations. Best result so far:
   - `probe6_bc_1779572840`: BC pretrain on, 4-7 matured/agent (5-10 %
     mat rate), but force-close ballooned (50-58/agent, 70-78 % fc
     rate). Agent learned to OPEN more aggressively from BC but PPO
     hadn't had time to learn selectivity in 1 generation.
7. **Audit of "what features did we build and then leave off"**
   uncovered FOUR off-by-default aux head loss weights
   (`fill_prob_loss_weight`, `mature_prob_loss_weight`,
   `risk_loss_weight`, `direction_prob_loss_weight`). The corresponding
   prediction heads are wired into actor_head's input but only LEARN
   if their loss weight is non-zero — so the policy has been
   conditioning on near-constant 0.5 outputs from untrained heads.
   The operator was justifiably annoyed: "we built these features
   over days and only now find we aren't using them!"
8. **The pre-flight probe currently running** has all those weights
   enabled at small positive values, plus per-transition credit, plus
   BC direction-target weight. It's testing the integrated config.

---

## What the current probe is testing

`registry/_predictor_SCALPING_full_features_probe2_1779576580/`
launched at ~22:50 with this set of changes vs `probe6_bc_*`:

| Knob | Before | Now | Why |
|---|---|---|---|
| `fill_prob_loss_weight` | 0 | **0.1** | Trains the fill_prob head fed into actor_head |
| `mature_prob_loss_weight` | 0 | **0.1** | Same with strict label |
| `risk_loss_weight` | 0 | **0.1** | Trains LSTM backbone toward locked_pnl features |
| `direction_prob_loss_weight` | 0 | **0.05** | Trains direction head (needs direction_labels cache) |
| `bc_direction_target_weight` | 0 | **0.1** | BC pretrain also trains direction head |
| `per_transition_credit` | off | **on** | Phase 9 S02 cleaner BCE label assignment |
| `direction_gate_threshold` | 0.5 (gene fixed) | **GA evolves [0.5, 0.95]** | Let GA find selectivity sweet spot |

(Background ID: `b8dmlt1wx`. Monitor `bnwpgl3nn` armed for completion
events.)

**Gotcha just caught:** `direction_prob_loss_weight > 0` requires
`data/direction_labels/<date>/horizon60_thresh5_fc60.npz` to exist.
First probe attempt crashed with `FileNotFoundError`. Fix: scan
direction labels for all training days first. The probe-2 directory
has those scans complete for 2026-04-06..04-24 (12 days). For the
full cohort (16 training days), still need to scan:
- 2026-04-26 (already cached — pre-existing)
- 2026-05-02, 05-04, 05-05 (already cached — pre-existing)

So no extra direction-label scan needed for the full cohort.

For the full cohort's **oracle cache** (BC pretrain): currently have
2026-04-06..04-16 (8 days) scanned with `--predictor-lean-obs`. Still
need to scan:
- 2026-04-19, 04-20, 04-22, 04-24, 04-26, 05-02, 05-04, 05-05
- Command: `python -m training_v2.oracle_cli scan --dates
  2026-04-19,2026-04-20,2026-04-22,2026-04-24,2026-04-26,2026-05-02,2026-05-04,2026-05-05
  --predictor-lean-obs`
- ~3 min wall

---

## What to do when you start

### 1. Check probe status

```powershell
Get-Content registry\_predictor_SCALPING_full_features_probe2_1779576580.log -Tail 20
cat registry\_predictor_SCALPING_full_features_probe2_1779576580\status.txt   # if status watcher running
```

If probe completed successfully (Cohort complete in N seconds):
- Verify behaviour shifted. Compare to `probe6_bc_1779572840`:
  - `mat%` should be similar or higher (aux heads now active)
  - `bets` may be lower if the new direction_gate_threshold gene
    drew a high value for some agents (more selective)
  - `fc%` ideally lower (better targeting via aux heads)
- Move to step 2.

If probe failed: read the traceback. Most likely either another
silently-dropped override or another missing cache. Add the missing
piece and re-run.

### 2. Scan remaining oracle cache (~3 min)

```powershell
python -m training_v2.oracle_cli scan `
  --dates 2026-04-19,2026-04-20,2026-04-22,2026-04-24,2026-04-26,2026-05-02,2026-05-04,2026-05-05 `
  --predictor-lean-obs
```

Confirms generate `data/oracle_cache_v2/<date>/oracle_samples.npz`
with `obs_dim=574`.

### 3. Launch the full cohort

Same args as probe2, scaled up:
- `--n-agents 6` (instead of 3)
- `--generations 5` (instead of 1)
- `--cohort-eval-days 2026-04-07 2026-04-10 2026-04-14 2026-04-17 2026-04-21 2026-04-23 2026-04-25 2026-05-01 2026-05-03 2026-05-06` (full 10 eval days)
- `--training-days-explicit 2026-04-06 2026-04-08 2026-04-09 2026-04-11 2026-04-12 2026-04-13 2026-04-15 2026-04-16 2026-04-19 2026-04-20 2026-04-22 2026-04-24 2026-04-26 2026-05-02 2026-05-04 2026-05-05` (full 16 training days)
- `--monitor-days 2026-05-07 2026-05-08 2026-05-09 2026-05-10 2026-05-11 2026-05-12 2026-05-13 2026-05-14 2026-05-15 2026-05-16 2026-05-17 2026-05-18 2026-05-19 2026-05-20`
- `--rotating-eval-sample 7 --monitor-eval-top-k 3 --monitor-early-stop-patience 2`
- All the `--reward-overrides` from probe2 unchanged
- Both `--enable-gene` flags unchanged

Output dir convention: `registry/_predictor_SCALPING_full_features_cohort_<unix_ts>`

Expected wall: ~13h. Start the two watchers (`show_cohort_status` and
`tools/watch_race_actions`) so the operator can monitor progress
while away.

### 4. Notify the operator

Once the cohort is launched:
- Tell them the background IDs of cohort + status watcher +
  race_actions watcher
- Tell them what files to watch
- Reassure them the probe-verified config means no integration
  surprises mid-cohort

---

## Key infrastructure to know

### Tools

- `tools/watch_race_actions.py` — per-agent human-readable race
  timeline with per-pair outcome flags (`--MAT`, `--FCP`, `--FCL`,
  `--ECP`, `--ECL`, `--NAK`). Refreshes every 60s. Launch:
  `python -m tools.watch_race_actions <cohort_dir> --watch 60`
- `tools/show_cohort_status.py` — cohort-level status.txt with
  `mat%` / `cls%` columns (the `mr` column was renamed earlier today
  because it conflated natural-maturation with agent-closed).
  Launch: `python tools/show_cohort_status.py <cohort_dir>
  --target-rows 30 --watch 60`
- `C:/tmp/oracle_ceiling.py` — measures the theoretical mat-rate
  ceiling on raw market data (~31-37 % at target_lock_pct=0.02
  with no gates).

### Caches needed

- **Oracle cache** (`data/oracle_cache_v2/<date>/`): per-training-day
  arb-oracle samples for BC pretrain. Must be scanned with
  `--predictor-lean-obs` so obs_dim matches the cohort's shim
  (574 for our config; without lean flag scan produces 2254-dim
  cache and BC refuses to load).
- **Direction labels** (`data/direction_labels/<date>/horizon60_thresh5_fc60.npz`):
  per-training-day binary direction labels. Needed when
  `direction_prob_loss_weight > 0`. Defaults must match what's used
  at runtime (horizon=60 ticks, threshold=5 ticks, fc=60s).

### CLAUDE.md sections worth reading

- "Price-adaptive arb_spread (2026-05-23)" — the formula
- "CLOSE_SIGNAL_BONUS zeroed 0.5 → 0.0 + matured_arb_bonus_weight
  scope narrowed (2026-05-23)" — the reward narrowing
- "Floor function uses equal-profit sizing (2026-05-23 fix)" — the
  equal-profit fix to `min_arb_ticks_for_profit`
- "Equal-profit pair sizing (scalping)" — the env's placement formula

### Recent commits (all 2026-05-23, master)

- `cf82431` — fix(env): whitelist naked_variance_penalty_beta + guard test
- `6db33db` — fix(env): whitelist matured_arb_expected_random + cap
- `88c8dde` — fix(oracle_cli): support --predictor-lean-obs
- `f193e41` — fix(reward): halve CLOSE_SIGNAL_BONUS + visibility tools
- `5d57a91` — fix(reward): zero CLOSE_SIGNAL_BONUS + narrow matured_arb_bonus
- `d2cc1a8` — refactor(arb_spread): replace headroom+scale with target_lock_pct
- `438cc99` — fix(scalping_math): use equal-profit sizing in floor
- `ae0d38d` — feat: price-adaptive arb_spread + cohort training speedup

### Tests

- `tests/test_forced_arbitrage.py` — 246 tests
- `tests/test_scalping_math.py` — the equal-profit floor regression guards
- New guard: `TestRewardOverridesWhitelistCovers_RewardCfgReads`
  (catches whitelist-out-of-sync bugs)

---

## Strategic context

The operator wants a scalper agent that:
- Opens pairs (back+lay) on runners with directional support
- Has the second leg fill naturally (matured), not via close_signal
  or env force-close
- Locks ~2 % of aggressive stake per matured pair

Current state:
- Theoretical mat ceiling: ~30-37 % at target_lock_pct=0.02
  (back-first scalps on day 2026-04-11, no gates applied)
- Practical ceiling under gates: ~15-25 % estimated
- Best probe result so far (probe6 with BC, 1 generation): 5-10 %
  mat rate, dominated by ~70 % force-close

**What we believe the multi-gen cohort can deliver:** with all aux
heads now training (giving the policy real fill-prob and
mature-prob predictions to condition on) and direction-gate
threshold evolving per-agent, the GA + PPO over 5 generations
SHOULD be able to find phenotypes that approach the practical
ceiling. Not guaranteed — could plateau at 10-15 % if the
predictor signal isn't strong enough — but that's what the cohort
tests.

**What "success" looks like:** an agent that opens fewer pairs
(50-100/day vs current 110-150), with mat rate 15-25 %, fc rate
<30 %, and positive eval_day_pnl. Even one such agent in the
cohort is a "proof of concept" the operator wants.

**What "failure" looks like:** all agents stay at 5-10 % mat
across 5 generations. That would tell us we've hit a structural
ceiling and need to either improve the predictors, change the
strategy (drop pair-trade for directional value-betting via
`--strategy-mode value_win`), or revisit the simulator (queue
position, ladder depth, etc.).

---

## Final note

Pace yourself. There were ~50 task items completed today. Most of
them were debugging the same kind of bug repeatedly (overrides
silently dropped, wrong obs_dim, missing caches). The new
whitelist guard test should catch one whole class going forward;
the lean-obs flag fix should catch another. The operator is
tired and a little frustrated — communicate clearly, don't gloss
over what didn't work, and don't overpromise the GA's reach.
