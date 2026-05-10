# Session 04 — Each-way action surface

## Goal

Add the each-way action signal so `value_each_way` mode can
place EW bets. **NO data-pipeline work** — EW settlement is
already complete (`plans/ew-settlement/`, finished 2026-04-11)
and EW metadata is already in the parquet pipeline
(`Race.each_way_divisor`, `Race.number_of_each_way_places`).
This session adds: a per-runner `each_way` action signal in the
`value_each_way` mode action space; an `each_way: bool` kwarg
on `BetManager.place_back` / `place_lay`; non-EW race masking
(skip races with `each_way_divisor is None` in EW mode).

## Context to read

- `plans/ew-settlement/purpose.md` — what EW settlement does and
  the worked examples.
- `plans/ew-settlement/progress.md` — confirms completion (4
  sessions, 2026-04-11).
- `plans/ew-metadata-pipeline/purpose.md` — the `Bet` dataclass
  EW metadata fields (`is_each_way`, `each_way_divisor`,
  `number_of_places`, `settlement_type`, `effective_place_odds`).
- `plans/predictor-integration/strategy_modes.md` §"value_each_way".
- `plans/predictor-integration/integration_contract.md` §4 + §7.
- `plans/predictor-integration/hard_constraints.md` §6 — DON'T
  re-derive EW settlement.
- `env/bet_manager.py::place_back`, `place_lay`, `settle_race`
  — settle_race already handles EW correctly; place_* needs an
  `each_way` kwarg added.
- `env/betfair_env.py::_apply_action` — where action signals
  are routed to bet placement.
- `agents_v2/action_space.py` — discrete action space for
  non-scalping mode.
- `env/betfair_env.py:484` — race-data fields available
  (`each_way_divisor`, etc.).

## Deliverables

| File | Touch |
|---|---|
| `env/bet_manager.py` | MODIFY — `place_back` / `place_lay` accept `each_way: bool = False` kwarg; set `bet.is_each_way` and `bet.each_way_divisor` from the Race when True |
| `env/betfair_env.py` | MODIFY — in `value_each_way` mode: add `each_way` action dim; route flag through `_apply_action`; mask action space when `race.each_way_divisor is None` |
| `agents_v2/action_space.py` | MODIFY — define the each-way action dim for `value_each_way` mode |
| `tests/test_each_way_action.py` | NEW — settlement and masking tests |

## Implementation notes

### `BetManager.place_back` / `place_lay` extension

```python
def place_back(
    self,
    selection_id: int,
    stake: float,
    price: float,
    *,
    force_close: bool = False,
    each_way: bool = False,
    each_way_divisor: float | None = None,
    number_of_places: int | None = None,
) -> Bet | None:
    ...
    if each_way:
        if each_way_divisor is None or number_of_places is None:
            return None  # cannot place EW bet on non-EW race
        bet.is_each_way = True
        bet.each_way_divisor = each_way_divisor
        bet.number_of_places = number_of_places
        bet.effective_place_odds = (price - 1.0) / each_way_divisor + 1.0
    return bet
```

`bet.is_each_way` flips on the `Bet` object at placement; the
existing `settle_race` path reads it and applies doubled-stake
+ place-fraction settlement. Per hard_constraints §6, DO NOT
modify `settle_race` itself.

Default `each_way=False` means existing call sites (arb mode,
value_win mode, agent-initiated closes, env force-closes) are
byte-identical. Only `value_each_way` mode passes
`each_way=True`.

### Action surface for `value_each_way`

The non-scalping action space is currently 4-dim per runner
(`signal`, `stake`, `aggression`, `cancel`). For
`value_each_way` mode, add an `each_way` dim:

```python
# agents_v2/action_space.py
ACTIONS_PER_RUNNER = 4   # value_win mode (and current non-scalping)
EACH_WAY_ACTIONS_PER_RUNNER = 5  # value_each_way mode (+each_way dim)
```

The discrete-action policy reads the action dim at
construction. RUNNER_DIM stays 143 across modes (the action
space changes, not the obs space). The new dim is gene-mode-gated
inside the policy.

(Alternative considered: keep 4-dim and require the agent to
encode each_way through the existing `signal` dim with an extra
bit. Rejected — the architectural cost of a new dim is small,
and signal-encoding tricks make the policy harder to interpret.)

### `_apply_action` routing

```python
# env/betfair_env.py::_apply_action (excerpt)
if self._strategy_mode == StrategyMode.value_each_way:
    if race.each_way_divisor is None:
        return  # masked: non-EW race, no bet possible

    each_way_signal = action[runner_idx, each_way_dim] > 0
    bet = self.bet_manager.place_back(
        selection_id=...,
        stake=...,
        price=...,
        each_way=each_way_signal,
        each_way_divisor=race.each_way_divisor,
        number_of_places=race.number_of_each_way_places,
    )
```

### Non-EW race masking

When `strategy_mode == value_each_way` and a race has no EW
market (`each_way_divisor is None`), the env emits a
log-once-per-race "skipping non-EW race for value_each_way mode"
message and accepts no bet placements for that race. The day
loop continues; the agent observes the race state for cross-race
context but cannot act.

This is intentional — the policy still gets reward signal from
other (EW-eligible) races in the same day.

### Optional: derived edge feature

If `integration_contract.md` §"Edge calculation" is adopted,
add ONE more RUNNER_KEY (`champion_p_placed_implied_edge`) in
this session to make the EW-edge condition explicitly
observable to the policy. Recommend doing so (it's one line in
`data/feature_engineer.py` once `champion_p_placed` and the
back price are in scope) and bumping RUNNER_DIM to 144 for v8.
**Decide before coding** — the operator may prefer to keep the
edge implicit, letting the policy learn it from the inputs.

## Hard constraints

- §1 (byte-identical): default `each_way=False` keeps all
  non-EW-mode call sites byte-identical.
- §2 (no env mechanics changes): `settle_race` is NOT touched.
- §6 (don't re-derive EW settlement): reuse the existing path
  verbatim.
- §13 (don't expand scope): no new shaping for EW; no new data
  pipeline.

## Success bar

- `tests/test_each_way_action.py` runs:
  - `test_place_back_with_each_way_sets_bet_flag` — calling
    `place_back(each_way=True, each_way_divisor=4.0,
    number_of_places=3)` returns a `Bet` with `is_each_way == True`,
    `each_way_divisor == 4.0`, `effective_place_odds == (price-1)/4 + 1`.
  - `test_place_back_default_each_way_false` — old call sites
    still produce `is_each_way == False` (byte-identical).
  - `test_value_each_way_mode_skips_non_ew_race` — env in
    `value_each_way` mode against a race with
    `each_way_divisor is None` yields no bets, no crashes.
  - `test_value_each_way_settlement_winner` — full episode in
    `value_each_way` mode where the policy bets EW on the
    actual winner; assert P&L matches the
    `plans/ew-settlement/purpose.md` worked-example formula.
  - `test_value_each_way_settlement_placed` — same for a
    runner that placed but didn't win.
  - `test_value_each_way_settlement_unplaced` — same for an
    unplaced runner.
- Existing `BetManager` tests (including
  `TestEachWaySettlementCorrected` from `plans/ew-settlement/`)
  STILL PASS unchanged.
- The byte-identical regression test from Session 02 STILL
  PASSES.

## Out of scope for this session

- `value_each_way` smoke cohort (Session 06).
- Three-way comparison (Session 07).
- Any change to `settle_race`, `BetManager.update`, or env
  matching mechanics.
- Lay-side EW (lays on EW markets settle correctly per
  `plans/ew-settlement/` Session 01; verify with one test, but
  the policy doesn't need to lay EW for `value_each_way` mode
  in the smoke cohort — back-only is the simpler first cut).

## Operator decision before Session 06

After this session lands, decide: does the smoke cohort
(Session 06) allow LAY EW bets, or back-only? Recommend:
back-only for the smoke. Layered EW betting has a different
liability profile and is worth a separate experiment if the
back-only smoke proves the mechanism works.
