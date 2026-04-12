# Session 1: Environment mechanics for forced arbitrage

Read `CLAUDE.md` and `plans/issues-12-04-2026/05-forced-arbitrage/`
before starting. Follow session 1 of `master_todo.md`. Mark items done
as you go and update `progress.md` at the end.

## Context

The agent has back, lay, aggressive, and passive order capability but
has never discovered arbitrage. This feature forces it: when scalping
mode is on, every aggressive fill auto-generates a passive counter-order
on the opposite side of the same runner, N ticks away.

### Current action space

`env/betfair_env.py:79` — `ACTIONS_PER_RUNNER = 4`
Per runner: `[signal, stake, aggression, cancel]`
Action vector shape: `(max_runners × 4,)` = `(14 × 4,)` = `(56,)`

Signal: > +0.33 → back, < -0.33 → lay, else skip
Aggression: > 0 → aggressive (cross spread), ≤ 0 → passive (queue)

### Current passive order system

`env/bet_manager.py:130-509` — `PassiveOrderBook`
- Orders rest at own-side best price
- Fill when traded volume exceeds queue position
- Cancel action removes oldest order per runner
- Already supports back and lay passive placement

### Betfair tick ladder

Prices aren't linear. Tick increments vary:
- 1.01–2.00: 0.01 increments (100 ticks)
- 2.00–3.00: 0.02 increments (50 ticks)
- 3.00–4.00: 0.05 increments (20 ticks)
- 4.00–6.00: 0.10 increments (20 ticks)
- 6.00–10.0: 0.20 increments (20 ticks)
- 10.0–20.0: 0.50 increments (20 ticks)
- 20.0–30.0: 1.00 increments (10 ticks)
- 30.0–50.0: 2.00 increments (10 ticks)
- 50.0–100:  5.00 increments (10 ticks)
- 100–1000:  10.0 increments (90 ticks)

Need a `tick_offset(price, n_ticks, direction)` utility function.

### Commission

Betfair charges ~5% on net market profit. For a completed arb:
- Back at P_high, lay at P_low (same runner, same stake S)
- If horse wins: back profit = S×(P_high-1), lay loss = S×(P_low-1)
- Net = S×(P_high - P_low), commission = 0.05 × net
- If horse loses: back loss = S, lay profit = S, net = 0
- So completed arb always nets S×(P_high - P_low)×0.95

The agent must learn that the spread needs to exceed commission
breakeven — which is price-dependent because tick sizes vary.

## What to do

### 1. Tick ladder utility

Create a `tick_offset(price, n_ticks, direction)` function. Probably
in a new `env/tick_ladder.py` or inside `exchange_matcher.py`.
- `direction = +1` → move price up N ticks
- `direction = -1` → move price down N ticks
- Clamp to [1.01, 1000]

### 2. arb_spread action dimension

Add 5th dimension to action space: `arb_spread`
- `ACTIONS_PER_RUNNER = 5` (when scalping_mode=True)
- Map [-1, 1] → [1, MAX_ARB_TICKS] (e.g. 1–15 ticks)
- Only read when scalping_mode is on

For backward compatibility: when scalping_mode=False, the 5th
dimension is still present in the action space but ignored. This
avoids needing two different policy architectures. All architectures
output 5 dims per runner — scalping just activates the 5th.

### 3. Auto paired order

In `_process_action()` after a successful aggressive fill:
```python
if self.scalping_mode:
    arb_ticks = decode_arb_spread(raw_arb_spread)
    if side == BACK:
        passive_price = tick_offset(fill_price, arb_ticks, -1)
        passive_side = LAY
    else:
        passive_price = tick_offset(fill_price, arb_ticks, +1)
        passive_side = BACK
    pair_id = uuid4().hex[:12]
    aggressive_bet.pair_id = pair_id
    passive_book.place(runner, passive_side, stake,
                       price=passive_price, pair_id=pair_id)
```

### 4. Pair tracking

Add `pair_id` to `Bet` and `PassiveOrder` dataclasses. Add helpers
to BetManager for querying paired/naked positions.

### 5. Observation additions

Add features so the agent can see its arb state:
- Per runner: has_open_arb, passive_fill_proximity
- Agent state: locked_pnl_frac, naked_exposure_frac

## Key files

| File | What to change |
|------|----------------|
| `env/betfair_env.py` | Action space expansion, auto paired order, new obs features |
| `env/bet_manager.py` | pair_id on Bet/PassiveOrder, paired position helpers |
| `env/exchange_matcher.py` | Possibly add tick_offset utility |
| `env/tick_ladder.py` (new) | Betfair tick ladder + offset function |
| `config.yaml` | scalping_mode option |
| `agents/hyperparameters.py` | scalping_mode gene |

## Constraints

- Backward compatible: scalping_mode=False must produce identical
  behaviour to current code. The 5th action dimension exists but is
  ignored.
- Existing policies (saved weights) won't have the 5th output dim.
  Handle gracefully — pad with zeros on load, or only expand for
  new models.
- The passive counter-order must use the real tick ladder prices,
  not linear approximations.
- `python -m pytest tests/ --timeout=120 -q` must pass.

## Commit

Single commit: `feat: forced arbitrage mechanics — paired orders + arb_spread action`
Push: `git push all`
