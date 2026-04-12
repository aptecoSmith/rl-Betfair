# Hard Constraints

- **Backward compatible**: scalping_mode=False must produce identical
  behaviour to current code. No changes to existing model outputs,
  reward calculations, or settlement logic when scalping is off.
- **Real tick ladder**: passive counter-orders must use Betfair's
  actual tick ladder (non-linear increments), not linear price offsets.
  An arb placed at a non-existent price would never fill on a real
  exchange.
- **Commission-aware**: locked PnL calculations must deduct the ~5%
  Betfair commission. The agent must learn that tiny spreads don't
  cover commission — don't hide this from the reward signal.
- **Existing policies**: saved weights from non-scalping models won't
  have the 5th action output. Handle gracefully — either pad with
  zeros on load or only expand the action head for new models.
  Never crash on loading an old model.
- **One bet losing is expected**: in scalping mode, do NOT apply
  precision_bonus or any metric that penalises losing bets. One leg
  of every completed arb is a planned loss.
- **Naked exposure at the off**: unfilled passive arb legs must be
  cancelled at race-off and budget/liability returned. The remaining
  naked bet settles directionally as normal.
- **raw + shaped ≈ total_reward** invariant must hold. Scalping PnL
  is raw (real money). Any shaping terms must be zero-mean for random
  policies.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
