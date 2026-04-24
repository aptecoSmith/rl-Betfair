# Session 01 prompt — Force-close at T−N + entropy-velocity gene + transformer ctx=256

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — especially the
  "Design sketch" subsections for (A) entropy controller
  velocity and (C) force-close at T−30s, and the
  "Failure modes" subsection (eats so much spread, gene
  collapses to floor, invariant breaks).
- [`../hard_constraints.md`](../hard_constraints.md) —
  §2 (no matcher changes), §4 (controller structure
  stays), §7 (force-closes NOT counted as matured), §9–
  §14 (force-close mechanics), §14a–§14d (transformer
  ctx widening), §15–§18 (entropy velocity gene), §28
  (invariant parametrised), §29 (telemetry fields),
  §30–§31 (testing).
- [`../master_todo.md`](../master_todo.md) — Session 01
  deliverables + exit criteria.
- `env/betfair_env.py` — the file being edited.
  Specific locations to find first:
  - The step loop's `time_to_off` calculation (around
    line 1458).
  - `_settle_current_race` (around line 2190) and the
    existing pair classification (arbs_completed,
    arbs_closed, arbs_naked).
  - `_attempt_close` (around line 1961) — execution
    mechanism for closes; force-close reuses this path.
  - `_REWARD_OVERRIDE_KEYS` frozenset.
  - `_get_info` / `_log_episode` telemetry paths.
- `env/bet_manager.py` — `Bet` dataclass
  (`close_leg: bool` already exists; add
  `force_close: bool = False` alongside). The scalping
  counters (`scalping_arbs_*`) — check whether
  BetManager owns these counts or whether BetfairEnv
  does.
- `env/exchange_matcher.py` — understand the junk
  filter and LTP guard that force-close inherits via
  `_attempt_close`. Do NOT modify.
- `env/scalping_math.py` — has
  `locked_pnl_per_unit_stake(P_back, P_lay, commission)`
  used by close-P&L calculations.
- `agents/ppo_trainer.py` — the file being edited for
  `alpha_lr`. Specific locations:
  - `self._log_alpha` and `self._alpha_optimizer`
    construction (grep for `SGD` to find it).
  - The controller step (grep `_log_alpha.exp`).
  - The trainer-side gene override path — whether a
    `_TRAINER_GENE_MAP` exists or whether the pattern is
    different.
- `CLAUDE.md` sections: "Order matching: single-price,
  no walking"; "Bet accounting: matched orders, not
  netted positions"; "Reward function: raw vs shaped";
  "Entropy control — target-entropy controller".
- `plans/arb-curriculum/session_prompts/02_matured_arb_bonus.md`
  — structural template for adding a new shaped-reward
  knob with invariant-test stacking. This session
  follows the same pattern.

## Why this is necessary

Per `purpose.md`, the 2026-04-21 `arb-curriculum-probe`
Validation found three failure mechanisms in the first
~10 post-BC episodes: entropy drift outruns the
controller's authority; shaped penalties dominate early
cash P&L; naked variance dominates the training signal.
This session attacks (A) and (C) — the entropy
controller's authority and the naked variance. (B) lands
in Session 02.

**Attacking both in one session** is deliberate. Force-
close changes the reward magnitude the controller is
trying to react to; entropy velocity changes how fast
the controller responds to that reward. Touching one
without the other means the second session's baseline
is a different reward landscape, and the tests have to
be re-derived.

## What to do

### 1. Scope confirmation (5 min)

Before any code changes, read `env/betfair_env.py`
around lines 1450–1470 (the step loop) and 2190–2560
(settlement). Verify:

- `time_to_off` is computed per-tick at line 1458.
- The pair classification loop at ~2225 uses
  `close_leg` on `Bet` to distinguish closed vs
  completed.
- `scalping_arbs_naked` is counted at ~2284 when a pair
  has an unfilled passive.
- `race_pnl` aggregation. Find where
  `scalping_locked_pnl`, `scalping_closed_pnl`, and the
  naked aggregation are summed. This is where
  `scalping_force_closed_pnl` adds a term.

If any of those don't match, PAUSE and consult the user
— the code may have drifted and this prompt needs
updating.

### 2. Env-side: add the config read

In `BetfairEnv.__init__`, near where other constraint
knobs land (grep for `_min_seconds_before_off` — right
next to that):

```python
# Force-close at T−N (plans/arb-signal-cleanup,
# Session 01, 2026-04-21). When > 0 and scalping_mode
# is on, any open pair with an unfilled second leg is
# force-closed via _attempt_close once time_to_off
# drops to or below the threshold. Best-effort: if the
# matcher can't find a priceable counter-leg, the
# position stays open and settles naked (subject to
# naked_loss_scale). Default 0 = disabled = byte-
# identical to pre-change. See hard_constraints.md
# s9-s14.
self._force_close_before_off_seconds: int = int(
    constraints.get("force_close_before_off_seconds", 0)
)
```

### 3. Env-side: add the force-close trigger

In the step loop at `env/betfair_env.py`, just after
`time_to_off` is computed (line 1458) and BEFORE the
per-runner action handling, add a dedicated force-close
pass. Conceptual shape (adapt to the actual control flow
— the step loop has nested runner iteration, so this
needs to happen once per tick, not per runner):

```python
# Force-close pass (plans/arb-signal-cleanup,
# Session 01). Runs before action handling so any
# force-closed leg is visible to downstream accounting
# the same way an agent-initiated close would be.
if (
    self.scalping_mode
    and self._force_close_before_off_seconds > 0
    and time_to_off <= self._force_close_before_off_seconds
):
    self._force_close_open_pairs(race, tick, time_to_off)
```

The `_force_close_open_pairs` helper (new method):

```python
def _force_close_open_pairs(
    self, race: Race, tick: Tick, time_to_off: float,
) -> None:
    """Force-close any pair with an unfilled second leg.

    Called once per tick when time_to_off drops to or
    below self._force_close_before_off_seconds. Best-
    effort: if the matcher rejects the close (no
    priceable opposite book, junk-filter trips, hard
    price cap), the pair stays open and settles naked.
    """
    bm = self.bet_manager
    assert bm is not None
    # Find pairs with one leg filled and one leg
    # unfilled (naked). Iterate bets grouped by pair_id
    # or use get_paired_positions + complete filter —
    # check which API exists. Mark only pairs where the
    # filled leg has no close_leg=True counter yet.
    ...
    for pair in naked_pairs:
        filled_leg = pair["filled"]  # or however the
                                      # API exposes it
        self._attempt_close(
            race=race,
            tick=tick,
            selection_id=filled_leg.selection_id,
            side_to_close=_opposite_side(filled_leg.side),
            stake=_sizing(filled_leg),  # see notes
            time_to_off=time_to_off,
            pair_id=filled_leg.pair_id,
            force_close=True,  # NEW flag, see step 4
        )
```

**Notes on stake sizing for the force-close leg:** look
at how `_attempt_close` currently sizes the agent-
initiated close (grep for its call sites around line
1684). Use the SAME sizing logic — we want the force-
close's execution semantics to match what
`close_signal` would have done. The equal-profit pair
sizing from CLAUDE.md governs it; don't re-derive.

**Notes on the API:** if `BetManager` doesn't currently
expose a "pairs with only one leg filled" query, you
may need to iterate `bm.bets` and group by `pair_id`
yourself, checking each pair has exactly one leg with
`status in (MATCHED, PARTIAL_MATCHED)` and no close leg
yet. Keep the helper inside `BetfairEnv` for now; don't
grow BetManager's API unless unavoidable.

### 4. Bet dataclass: new `force_close` attribute

`env/bet_manager.py`:

```python
@dataclass
class Bet:
    ... existing fields ...
    close_leg: bool = False
    force_close: bool = False  # NEW — plans/arb-signal-
                                # cleanup Session 01.
                                # Distinguishes env-
                                # initiated force-closes
                                # from agent-initiated
                                # close_signal closes.
                                # force_close=True implies
                                # close_leg=True.
```

`_attempt_close` gets a new keyword argument
`force_close: bool = False` and propagates it to the
constructed `Bet`.

### 5. Env-side: classify force-closed pairs in settlement

In `_settle_current_race`, the existing classification
loop around line 2225 currently classifies pairs as
closed (any leg has `close_leg=True`) or completed. Add
a third branch for force-closes:

```python
# Existing (around line 2242):
is_closed = (
    (agg is not None and agg.close_leg)
    or (pas is not None and pas.close_leg)
)
# NEW — force-closes are a subtype of closed pairs
# but accounted separately (hard_constraints.md s7,
# s12, s14). A pair is force-closed if EITHER leg has
# force_close=True.
is_force_closed = (
    (agg is not None and agg.force_close)
    or (pas is not None and pas.force_close)
)
```

Then in the dispatch:

```python
if is_force_closed:
    scalping_arbs_force_closed += 1
    # Compute force-close P&L exactly like the existing
    # closed-at-loss path. Reuse win_pnl / lose_pnl
    # computation from lines 2252-2260; take
    # min(win_pnl, lose_pnl) as the realised cost.
    force_close_realised = min(win_pnl, lose_pnl)
    scalping_force_closed_pnl += force_close_realised
    # Also record to close_events for the activity log
    # but MARK the entry as force_close so the UI /
    # bet explorer can display it differently.
    self._close_events.append({
        "selection_id": agg.selection_id,
        "back_price": back_bet.average_price,
        "lay_price": lay_bet.average_price,
        "realised_pnl": force_close_realised,
        "race_idx": self._race_idx,
        "force_close": True,  # NEW
    })
elif is_closed:
    # existing logic for agent-initiated close
    scalping_arbs_closed += 1
    ...
else:
    scalping_arbs_completed += 1
```

Initialise `scalping_arbs_force_closed = 0` and
`scalping_force_closed_pnl = 0.0` at the top of the
function alongside the other `scalping_*` counters.

### 6. Env-side: update `race_pnl` formula

Per `hard_constraints.md` §13, `race_pnl` becomes:

```python
race_pnl = (
    scalping_locked_pnl
    + scalping_closed_pnl
    + scalping_force_closed_pnl    # NEW
    + scaled_naked_sum
)
```

Find the existing `race_pnl =` assignment in
`_settle_current_race` (it's the line after all the
component sums). Add the force-closed term inline.

### 7. Env-side: matured-arb bonus excludes force-closes

Per `hard_constraints.md` §7, the existing matured count
stays `completed + closed` (agent-initiated only).
Verify this — the current code computes:

```python
n_matured = scalping_arbs_completed + scalping_arbs_closed
```

Do NOT add `scalping_arbs_force_closed` to this sum.
Write a comment at the site explaining why (so nobody
adds it later "for consistency"):

```python
# Matured-arb bonus counts ONLY pair maturations the
# agent caused — natural completions and close_signal-
# initiated closes. Force-closes at T-N are env-
# initiated and do NOT earn the agent credit.
# hard_constraints.md s7 (plans/arb-signal-cleanup).
n_matured = scalping_arbs_completed + scalping_arbs_closed
```

### 8. Env-side: `close_signal` success bonus excludes force-closes

Per `hard_constraints.md` §14. Find the `+£1 per
close_signal success` contribution (grep for
`close_signal_success_bonus` or equivalent — its
location has drifted across plans). It reads
`scalping_arbs_closed` currently; leave it reading
ONLY `scalping_arbs_closed`, not the sum with
`scalping_arbs_force_closed`.

### 9. Env-side: telemetry

In `_get_info`, add:

```python
"arbs_force_closed": scalping_arbs_force_closed,
"scalping_force_closed_pnl": scalping_force_closed_pnl,
"force_close_before_off_seconds":
    self._force_close_before_off_seconds,
```

In `_log_episode` (or wherever per-episode JSONL rows
are assembled — usually `agents/ppo_trainer.py`),
mirror the same three fields. Also add
`alpha_lr_active` from the trainer (see step 11).

### 10. Trainer-side: `alpha_lr` as gene

`agents/ppo_trainer.py`:

- Find `self._alpha_optimizer` construction (grep for
  `SGD(` and/or `self._log_alpha`). It's currently a
  hardcoded `lr=1e-2`.
- Accept `alpha_lr: float = 1e-2` on `PPOTrainer.__init__`
  (or `PPOConfig`, whichever holds the hyperparameters).
- Pass it through:

```python
self._alpha_lr: float = float(alpha_lr)
self._alpha_optimizer = torch.optim.SGD(
    [self._log_alpha],
    lr=self._alpha_lr,
    momentum=0,
)
```

- Record `self._alpha_lr` in `EpisodeStats` as
  `alpha_lr_active` (optional field; pre-change rows
  won't have it).

### 11. Trainer-side: gene plumbing

Look for how other trainer-side genes flow. Currently
the per-agent `entropy_coefficient`, `reward_clip`,
`bc_pretrain_steps`, etc. are set from the gene
dictionary. Follow the same pattern for `alpha_lr`.

If there's a `_TRAINER_GENE_MAP` dict analogous to
`_REWARD_GENE_MAP`, add `"alpha_lr": ("alpha_lr",)`.

If no such dict exists, find where the gene dict is
unpacked into trainer config (likely in `training/
run_training.py` or `agents/population_manager.py`
during agent construction) and plumb `alpha_lr`
through that path.

Default (no gene override) = `1e-2`. Runs without
`alpha_lr` in their gene schema stay byte-identical.

### 12. Transformer context-window widening

`agents/policy_network.py::PPOTransformerPolicy`:

Current state (as of 2026-04-21):
- Class docstring (around line ~1235) lists
  `transformer_ctx_ticks ∈ {32, 64, 128}`.
- `self.position_embedding = nn.Embedding(ctx_ticks,
  d_model)` already sizes off the gene value — no
  hardcoded ceiling.
- Causal mask built at `ctx × ctx` — ditto.

The change is almost entirely documentation + range
enumeration:

- Update the class docstring: "Three structural genes:
  ... `transformer_ctx_ticks` ∈ {32, 64, 128, 256}"
  (add 256).
- Grep the codebase for every site that enumerates
  ctx_ticks choices. Likely suspects:
  - `agents/architecture_registry.py` (if it defines a
    gene schema for the transformer)
  - `training/training_plan.py` (if it validates
    `transformer_ctx_ticks` values)
  - `tests/` anywhere that hard-codes the choice list
  - Any CLAUDE.md or plan-folder mention of the range
- At every site, add 256 as an allowed value.
  Strictly additive — do NOT remove 32, 64, or 128.

Then verify no architectural change is needed by
instantiating a policy locally:

```python
from agents.policy_network import PPOTransformerPolicy
# minimal hyperparams
hp = {
    "lstm_hidden_size": 256,
    "mlp_hidden_size": 128,
    "mlp_layers": 2,
    "transformer_heads": 4,
    "transformer_depth": 2,
    "transformer_ctx_ticks": 256,
}
policy = PPOTransformerPolicy(
    obs_dim=100, action_dim=30, max_runners=10,
    hyperparams=hp,
)
# Should build without error. position_embedding is
# nn.Embedding(256, 256); causal_mask is 256x256.
```

If any assert fires or shape mismatch appears, follow
the trace and fix inline — that goes in-scope as part
of this step per §14a–§14b.

### 13. Weight-file hash check (brief audit)

`registry/model_store.py` validates architecture
compatibility on weight load. Confirm (via a quick
read of the hashing logic, or a short test) that a
policy with `ctx_ticks=256` produces a different
architecture hash than one with `ctx_ticks=128`. We
don't need to ADD a check — we need to confirm the
existing check correctly treats these as distinct
variants. If the current hash hashes everything
except ctx_ticks, that's a latent bug worth flagging
but likely harmless in practice (no existing weight
files are at ctx=256). Note anything found in
`lessons_learnt.md`.

### 14. config.yaml

```yaml
constraints:
  ... existing ...
  # Force-close at T−N (plans/arb-signal-cleanup
  # Session 01, 2026-04-21). Triggers env-initiated
  # best-available close on any open pair with an
  # unfilled second leg once time_to_off drops to or
  # below this threshold (scalping_mode only). Default
  # 0 = disabled. Typical active value: 30. See
  # plans/arb-signal-cleanup/purpose.md.
  force_close_before_off_seconds: 0

agents:
  ppo:
    ... existing ...
    # Target-entropy controller learning rate
    # (plans/arb-signal-cleanup Session 01,
    # 2026-04-21). SGD momentum=0 on log_alpha.
    # Previously hardcoded 1e-2 in PPOTrainer; now a
    # per-agent gene with this default for runs without
    # a gene override. See CLAUDE.md "Entropy control".
    alpha_lr: 0.01
```

### 15. Tests — `tests/arb_signal_cleanup/test_force_close.py`

Per `hard_constraints.md` §31. Create the directory if
it doesn't exist (`tests/arb_signal_cleanup/__init__.py`).
Ten tests:

1. **Force-close fires at threshold.** Scripted race.
   Build a fixture where one pair is opened at T−60s
   and stays naked. At `threshold=30`, step through to
   T−31s: no force-close. Step to T−29s: pair is
   force-closed. Assert `arbs_force_closed=1`,
   `arbs_naked=0`.
2. **Force-close uses matcher.** Scripted race with a
   matchable opposite-side book at T−29s → close lands
   (`arbs_force_closed=1`, `scalping_force_closed_pnl`
   non-zero). Scripted race with an unpriceable runner
   (no LTP, or all levels outside the ±50 %
   deviation) at T−29s → close refused, pair stays
   naked (`arbs_force_closed=0`, `arbs_naked=1`).
3. **Force-close respects hard price cap.** Opposite-
   side best price after junk filter > `max_back_price`
   (or `max_lay_price` for the lay side) → close
   refused, pair stays naked.
4. **Force-close P&L in `race_pnl`.** Assert
   `race_pnl == scalping_locked_pnl +
   scalping_closed_pnl + scalping_force_closed_pnl +
   scaled_naked_sum` (spot-check the formula holds).
5. **Matured-arb bonus excludes force-closes.**
   Scripted race with bonus weight 1.0, cap 100,
   `expected_random=0`, 5 force-closes + 2 natural
   matures → `n_matured = 2`, bonus =
   `1.0 * (2 - 0) = 2.0`, not 7.0.
6. **Close_signal bonus excludes force-closes.**
   Scripted race with 0 agent closes + 5 force-closes
   → close_signal bonus contribution is 0.
7. **`alpha_lr` gene passthrough.** Construct
   PPOTrainer with a gene dict including
   `{"alpha_lr": 0.05}` → `trainer._alpha_optimizer.
   param_groups[0]['lr']` is 0.05.
8. **`alpha_lr` doesn't mutate.** Run 5 PPO updates
   with gene value 0.05; assert the optimiser LR is
   still 0.05 after each. (The controller steps
   `log_alpha`, it does NOT change the LR.)
9. **Invariant parametrised.** Add rows to the
   existing
   `test_invariant_raw_plus_shaped_equals_total_reward`
   (or a sibling test in
   `tests/arb_signal_cleanup/`) covering
   `force_close_before_off_seconds ∈ {0, 30}` and
   `alpha_lr ∈ {1e-2, 5e-2}`.
10. **Transformer builds and forwards at ctx=256**
    (per `hard_constraints.md` §14c). Instantiate
    `PPOTransformerPolicy` with
    `transformer_ctx_ticks=256`, run one forward
    pass on a synthetic `(1, obs_dim)` input,
    assert: (a) no exception raised, (b)
    `out.action_mean.shape == (1, action_dim)`,
    (c) `out.hidden_state[0].shape[1] == 256`
    (the rolling buffer is sized to ctx_ticks).
    Runs CPU-only (default pytest config). No GPU
    memory assertions — that's a manual smoke step
    during the session, not a regression guard.

### 16. CLAUDE.md updates

Under "Bet accounting: matched orders, not netted
positions", new subsection at the end:

```
### Force-close at T−N (2026-04-21)

When `constraints.force_close_before_off_seconds > 0`
and scalping_mode is on, the env force-closes any pair
with an unfilled second leg once `time_to_off` drops to
or below the threshold. Closes go through the existing
`_attempt_close` path — same matcher, same junk filter,
same LTP guard. Each closed leg is flagged
`force_close=True` on the Bet object so settlement
accounting classifies the pair into
`scalping_arbs_force_closed` (separate from
`scalping_arbs_closed` which stays agent-initiated
only).

`race_pnl` gains a `scalping_force_closed_pnl` term.
The matured-arb bonus and the `+£1 per close_signal`
shaped bonus BOTH exclude force-closes — the agent
didn't choose them. A force-close that the matcher
refuses (unpriceable runner, junk-filter trip, price
above cap) leaves the pair naked and settlement falls
back to the existing naked-term accounting.

Default `0` = disabled = byte-identical to pre-change.
See `plans/arb-signal-cleanup/purpose.md` for the
design rationale (naked variance dominates the training
signal; force-close converts ±£100s variance into
bounded ±£0.50–£3.00 spread cost).
```

Under "Entropy control — target-entropy controller
(2026-04-19)", new subsection at the end:

```
### alpha_lr as per-agent gene (2026-04-21)

`alpha_lr` (the SGD learning rate on `log_alpha`) is
now a per-agent gene, range `[1e-2, 1e-1]` in the
`arb-signal-cleanup-probe` plan. Previously hardcoded
at `1e-2` in PPOTrainer; the 2026-04-21
`arb-curriculum-probe` Validation observed entropy
drifting monotone 139 → 170–184 across ep1–ep10 on
17/66 agents, with `1e-2` unable to arrest drift once
entropy passed ~157. The gene range lets the GA find
the right velocity.

Default (no gene override) stays `1e-2` so reference
runs without `alpha_lr` in their schema are byte-
identical. The controller's structure (SGD, momentum
0, `log_alpha` clamp `[log(1e-5), log(0.1)]`, target
150, BC handshake via `bc_target_entropy_warmup_eps`)
is unchanged.
```

Third new subsection — under whatever section of
CLAUDE.md documents the architecture choices (look for
an existing "Transformer" mention; if none, add a new
subsection):

```
### Transformer context window — 256 available (2026-04-21)

`PPOTransformerPolicy.transformer_ctx_ticks` is a
structural gene with allowed values
`{32, 64, 128, 256}` (2026-04-21: 256 added; previous
max was 128).

Scale context: training-data races average ~150–250
ticks. At `ctx_ticks=32` (default) the transformer
attends to only the last ~13 % of a race; at 256 it
covers the full race for the median case. The LSTM
variants don't have this limitation — their hidden
state is initialised once per rollout and carries
across every tick of the day.

A transformer model trained at one `ctx_ticks` value
cannot cross-load weights from another (the
`position_embedding` matrix has different shape). The
`registry/model_store.py` architecture-hash check
treats each ctx_ticks value as a distinct variant.

See `plans/arb-signal-cleanup/purpose.md` for the
decision rationale (2026-04-21 transformer-memory
audit; pinned at 256 in the probe to remove a
systematic handicap).
```

### 17. Full-suite check

```
pytest tests/arb_signal_cleanup/ -x
```

Then — ONLY if no training is active —
`pytest tests/ -q --timeout=120`.

### 18. Commit

```
feat(env+arch): force-close before off + per-agent alpha_lr gene + transformer ctx 256 option

Three coordinated additions that together attack the
first-10-episode signal-noise problem identified in
the 2026-04-21 arb-curriculum-probe Validation, plus
one architectural ceiling raise surfaced during the
same audit:

1. Env-initiated force-close at T-N seconds before off.
   When constraints.force_close_before_off_seconds > 0
   and scalping_mode is on, any open pair with an
   unfilled second leg is force-closed via
   _attempt_close once time_to_off drops to the
   threshold. Reuses matcher, junk filter, LTP guard,
   hard price cap. Pairs the matcher rejects stay
   naked (existing accounting). New Bet.force_close
   flag + scalping_arbs_force_closed counter +
   scalping_force_closed_pnl in race_pnl.

2. Per-agent alpha_lr gene on the target-entropy
   controller. The SGD LR on log_alpha was hardcoded
   1e-2; the arb-curriculum-probe observed monotone
   entropy drift that 1e-2 couldn't arrest. Now a
   PPOTrainer init parameter plumbed through the gene
   path, default 1e-2 unchanged.

3. PPOTransformerPolicy allows transformer_ctx_ticks
   = 256 (previously max 128). Pure range widening —
   position_embedding and causal_mask already size off
   the gene value. Motivated by the 2026-04-21
   transformer-memory audit: ctx=128 covers only ~54%
   of a typical race while LSTM variants carry full-
   day memory; ctx=256 covers the median race end-to-
   end. Strictly additive (32/64/128 still valid).

None of the three affects pre-existing runs (new env
knobs default to 0, alpha_lr default 1e-2, ctx default
32). Matured-arb bonus and close_signal shaped bonus
both exclude force-closes (env-initiated, not agent-
chosen). Invariant raw+shaped~=total stays green
parametrised over force-close and alpha_lr values.

Tests: 10 in tests/arb_signal_cleanup/test_force_close.py
(adds a transformer-builds-at-ctx-256 forward test)
plus invariant parametrisation.

CLAUDE.md: three new dated subsections (Bet accounting,
Entropy control, Transformer architecture).

Per plans/arb-signal-cleanup/hard_constraints.md s9-s18,
s14a-s14d, s28-s31.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Do NOT

- Do NOT touch `env/exchange_matcher.py`. Force-close
  calls existing matcher code; the matcher itself is
  unchanged.
- Do NOT count force-closes as matured arbs. The bonus
  at `n_matured = completed + closed` is load-bearing.
- Do NOT credit `close_signal` bonus for force-closes.
  The agent didn't choose them.
- Do NOT change the controller's structure. Only
  `alpha_lr` (the LR value) becomes a gene. SGD vs
  Adam, momentum, clamp, target, handshake — all
  unchanged.
- Do NOT bundle Session 02 (shaped-penalty warmup) into
  this commit.
- Do NOT set `force_close_before_off_seconds` to
  non-zero in `config.yaml`. The default is disabled.
  The probe plan (Session 03) sets it to 30 on the
  relevant cohorts.
- Do NOT run the full pytest suite during active
  training.
- Do NOT let the force-close stake size diverge from
  what `close_signal` would have placed. The equal-
  profit pair sizing governs both.

## After Session 01

1. Append a progress entry to
   [`../progress.md`](../progress.md) with commit hash
   + test delta + any gotchas (e.g. API surface changes
   on BetManager, discovered issues with existing
   close accounting).
2. Note in `lessons_learnt.md` anything surprising
   about where force-close accounting had to hook in
   (the pair-classification loop in
   `_settle_current_race` is a busy place; document
   any drift from this prompt's assumptions).
3. Hand back for Session 02 (shaped-penalty warmup).
