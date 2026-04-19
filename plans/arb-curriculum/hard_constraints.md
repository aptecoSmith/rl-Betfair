# Hard constraints — Arb Curriculum

Non-negotiable rules. Anything that violates one gets
rejected in review.

## Scope

**§1** This plan makes these coordinated changes, in
session order:

1. Offline oracle scan producing an on-disk sample cache
   per training day; sample = (obs, runner_idx,
   arb_spread_ticks, expected_locked_pnl, tick_index).
2. Matured-arb bonus: small fixed shaped reward per pair
   whose second leg fills (natural or closed).
3. Naked-loss annealing: per-agent `naked_loss_scale` gene
   + env-side scaler inside `race_pnl`'s naked component;
   annealing curve driven by generation index.
4. Per-agent BC pretrainer on oracle cache; runs before
   the first PPO rollout.
5. Curriculum day ordering in the training worker using
   oracle density.
6. Registry reset + new training plan.
7. Validation launch (operator-gated).

Anything NOT in that list is out of scope. Explicit
out-of-scope examples: controller changes, matcher
changes, action/obs schema bumps, new aux heads, new
gene-range edits beyond the four this plan introduces,
live-ordering / wall-time budgeting changes.

**§2** No changes to matcher semantics.
`env/exchange_matcher.py` stays single-price, no-walking,
LTP-filtered. The oracle READS the matcher's filter
predicates (Session 01 exports them as pure functions if
they aren't already); it does not introduce new filtering
rules.

**§3** No changes to PPO numerical stability defences.
Ratio clamp ±5 stays. KL early-stop 0.03 stays. Per-arch
LR stays. 5-update LR warmup stays. Advantage normalisation
stays. Reward centering EMA stays.

**§4** The `entropy-control-v2` target-entropy controller
stays wired in. Session 04 (BC pretrainer) handles the
BC↔controller handshake explicitly — see §18.

**§5** MTM shaping from `reward-densification` stays at its
config.yaml default (0.05). No MTM changes in this plan.

## Oracle scan (Session 01)

**§6** Oracle is offline-only. Never runs inside the
training loop. CLI entrypoint
`python -m training.arb_oracle scan --date ...` writes to
`data/oracle_cache/{date}/oracle_samples.npz`. Cache is
gitignored.

**§7** Oracle profitability check uses
`env.scalping_math.locked_pnl_per_unit_stake(P_back, P_lay,
commission)` — the same helper the env uses. Do NOT
re-implement.

**§8** Oracle reachability check uses the matcher's filter
predicates. Exporting them as pure functions is in scope
if needed. A sample is emitted only if the env could
actually execute the paired placement at that tick.

**§9** Oracle cache is deterministic. Same input → same
bytes. Tagged with `obs_schema_version`,
`action_schema_version`, and `scalping_mode` in the `.npz`
header; hard error on load mismatch (no silent coercion).

## Matured-arb bonus (Session 02)

**§10** Matured-arb bonus lives in `shaped_bonus`, never
in `raw_pnl_reward`. The bonus is positive per pair whose
second leg fills — naturally or via `close_signal` — AND
is zero-mean corrected so a random policy's expected
bonus is zero. Correction: subtract `expected_random_pairs
× bonus` from shaped once per episode, where
`expected_random_pairs` is a running population-level
estimate (start with a fixed 2.0 for simplicity; can be
an EMA later).

**§11** New reward-config key
`matured_arb_bonus_weight: float`. Default `0.0` =
byte-identical to pre-change (no bonus). Whitelisted in
`_REWARD_OVERRIDE_KEYS` so per-agent genes can flow.

**§12** The bonus magnitude is capped per-episode at
`matured_arb_bonus_cap` (default `10.0` pounds) to prevent
the "complete tiny unprofitable arbs for the bonus"
failure mode. Cap is a config key, not a gene.

## Naked-loss annealing (Session 03)

**§13** New gene `naked_loss_scale: float`, range
`[0.0, 1.0]`. Multiplied into the per-pair naked-loss
component of `race_pnl` inside `_settle_current_race`:

```
naked_pnl_contribution = sum(
    min(0, per_pair_naked_pnl) * naked_loss_scale
    + max(0, per_pair_naked_pnl)  # winners unchanged
)
```

Asymmetry is deliberate — we anneal the LOSS side to
bootstrap the policy past the naked valley; winners stay
at full value (still clipped 95% in shaped as today).

**§14** Annealing schedule is configured at the plan level,
not per-gene. A new plan-JSON field
`naked_loss_anneal: {start_gen: int, end_gen: int}`
instructs the population manager to rescale the
`naked_loss_scale` gene across generations: at
`start_gen` the gene's effective value is whatever the GA
rolled; at `end_gen` and beyond it's forced to 1.0. Linear
interpolation between. Default `{start_gen: 0,
end_gen: 0}` = no annealing (byte-identical).

**§15** `naked_loss_scale < 1.0` runs are flagged in
`episodes.jsonl` via `naked_loss_scale_active` (new
optional field). Scoreboard rows with `scale < 1.0` are
NOT comparable to `scale=1.0` rows on `raw_pnl_reward`.

## BC pretrainer (Session 04)

**§16** **Per-agent BC, never shared.** Each agent
pretrains its own policy from scratch on its own share of
the oracle cache. Inherited from
`plans/arb-improvements/lessons_learnt.md` — a prior
footgun. Violating this is a correctness bug, not a style
issue.

**§17** BC trains only the `signal` head (cross-entropy
against oracle's "back" target on the right runner slot)
and the `arb_spread` head (MSE against oracle's tick
count). Stake, aggression, cancel, requote_signal, and
close_signal heads are NOT touched by BC.

**§18** BC-controller handshake: after BC completes, the
policy's forward-pass entropy is typically low. The
target-entropy controller (target 150) will push `alpha`
up aggressively on the first PPO update and undo BC's
shaping. A new gene `bc_target_entropy_warmup_eps`
(default 5, range `[0, 20]`) tells the controller to
anneal `target_entropy` from the measured post-BC entropy
up to 150 over that many episodes. Zero = no warmup
(legacy behaviour).

**§19** BC uses a separate optimiser from PPO's Adam.
When BC finishes, the first PPO mini-batch starts with
fresh PPO-Adam state (first/second moments both zero).
LR warmup still kicks in. No optimiser state bleed.

**§20** Empty oracle cache for a given date → BC skips
cleanly for that date, with a warning in the worker log.
If an agent's union of training-date samples is empty, BC
is skipped entirely and PPO starts as usual (equivalent
to `bc_pretrain_steps=0`).

## Curriculum day ordering (Session 05)

**§21** Curriculum ordering is opt-in via a new training
config key `training.curriculum_day_order: str`. Values:
- `"random"` (default, current behaviour) — shuffle per
  seed.
- `"density_desc"` — arb-rich days first.
- `"density_asc"` — arb-poor days first (for debugging
  the reverse hypothesis).

**§22** Curriculum ordering DOES NOT drop days. Every
training day is seen exactly once per epoch, same as
today. Only the order changes.

**§23** When `curriculum_day_order != "random"` and the
oracle cache for a given date is missing, that date is
treated as density-zero (slotted at the end of the
curriculum). Worker logs a warning so the operator knows
to re-run the oracle scan.

## Telemetry and invariant

**§24** Existing invariant test
`test_invariant_raw_plus_shaped_equals_total_reward` MUST
stay green, parametrised across every new knob:
`matured_arb_bonus_weight ∈ {0.0, 1.0}`,
`naked_loss_scale ∈ {0.5, 1.0}`. Any failure is a blocker.

**§25** Per-episode JSONL rows gain optional fields:
- `matured_arb_bonus_active: float` (Session 02)
- `naked_loss_scale_active: float` (Session 03)
- `bc_pretrain_steps: int` (Session 04)
- `bc_final_signal_loss: float` (Session 04)
- `bc_final_arb_spread_loss: float` (Session 04)
- `curriculum_day_order: str` (Session 05)

Pre-change rows lack them; downstream readers must tolerate
absence. Same backward-compat pattern as
`mtm_weight_active` / `alpha`.

## Testing

**§26** Each session commit ships with new tests. Full
`pytest tests/ -q` MUST be green on every session commit.

**§27** Session 01 tests (oracle scan): 8 tests per
existing design (synthetic profitable moment → sample;
filter compliance × 2; empty day; determinism;
round-trip; density metric; obs-dim matches env).

**§28** Session 02 tests (matured-arb bonus): invariant
preserved; bonus emitted only on pair maturation;
zero-mean correction present; cap enforced; JSONL field
emitted.

**§29** Session 03 tests (naked-loss annealing): env-side
scaler correctness for three scale values (0, 0.5, 1);
annealing interpolation across generations; invariant
preserved at `scale<1`; JSONL field emitted.

**§30** Session 04 tests (BC pretrainer): per-agent
independence (seed-divergent weights); only target heads
change; empty-cache skip; loss decreases on synthetic
samples; handshake gene respected; integration test
spying on `_update_reward_baseline` per the 2026-04-18
units-mismatch lesson.

**§31** Session 05 tests (curriculum): three ordering modes
produce the expected day sequence on synthetic density
inputs; missing-date fallback places date at end; a
`curriculum_day_order` round-trip through config preserves
the value.

**§32** Sessions 06 and 07 are operator-gated (registry
manipulation + UI launch). No new tests required.

## Reward-scale change protocol

**§33** This plan introduces TWO reward-scale changes:
(a) non-zero `matured_arb_bonus_weight`, and (b)
`naked_loss_scale < 1.0`. Runs with either active are NOT
byte-identical to pre-change. Per-race `raw_pnl_reward`
stays comparable ONLY when `naked_loss_scale = 1.0`.
`shaped_bonus` stays comparable ONLY when
`matured_arb_bonus_weight = 0.0`.

**§34** CLAUDE.md "Reward function: raw vs shaped" gets
TWO new dated paragraphs (one per reward-scale change).
Historical entries stay preserved.

## Cross-session

**§35** Sessions land as separate commits, in order
01 → 07. Session 06 and 07 are operator-gated (manual
registry archive + plan redraft + launch); same
precedent as `reward-densification` Session 03.

**§36** If Session 01 fails (oracle produces no samples
on real data, filter mismatch, determinism test red),
Sessions 04–07 don't start. BC is unbuildable without the
oracle.

**§37** If Session 02 OR Session 03 fails the invariant
test (existing or new), that session blocks. Do not bundle
"fix accounting" with the feature commit.

**§38** Do NOT bundle the re-launch into the Session 06
commit. Session 06 is "archive + redraft"; the launch is
a follow-on operator action that writes back into
`progress.md` as a Validation entry.

**§39** Archive artefacts from the 2026-04-19
`reward-densification-gene-sweep` stay available for
comparison — Session 06 moves them into
`registry/archive_<iso>/` without deletion.

**§40** When each session lands, append a dated entry to
`progress.md` mirroring the format used in
`reward-densification/progress.md`: What landed, Not
changed, Gotchas, Test suite, Next.
