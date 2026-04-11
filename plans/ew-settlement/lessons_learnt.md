# Lessons Learnt — EW Settlement

Append-only. Date each entry.

---

## 2026-04-11 — Discovery

The EW settlement bug was discovered by inspecting live Betfair
screenshots and noticing that EW prices were *higher* than Win
prices. This prompted the question: "are we treating EW bets the
same as Win bets?" — and the answer was yes, across the entire
pipeline.

The comment "Betfair EACH_WAY markets already quote the
place-adjusted price" in `episode_builder.py` was written without
verification. The BetfairPoller queries `RUNNER_EXCHANGE_PRICES_BEST`
which returns **Win market prices** regardless of market type. The
EW divisor is captured separately by `EachWayHelper`. The place
fraction was never applied anywhere.

Impact: every EW race in the training data has an incorrect reward
signal. The magnitude depends on the race outcome and divisor, but
in the worst case (placed runner at long odds) the error is 10x+.

## 2026-04-11 — Historical P&L comparison results

Ran `scripts/ew_pnl_comparison.py` across 11 days of training data
(145 EW races, 290 simulated BACK+LAY bets on the favourite).

Key findings:
- **Mean absolute P&L delta per bet: £7,127** (at £10 stake).
  This is dominated by long-odds placed runners where the old method
  paid at full Win odds instead of the place fraction.
- **Total delta across all bets: +£52,944** — old method systematically
  overpaid placed-only back bets and over-charged placed-only lay bets.
- **0% of bets changed sign** — the direction was already correct
  (winners still win, losers still lose), but the magnitude was wildly
  wrong. This means the agent learned roughly correct *which* runners
  to bet on, but the *how much* signal was distorted.
- **PLACED runners account for 84% of the total delta** (£44,249 of
  £52,944). WINNER delta is smaller (£8,695) because both legs pay
  for winners — the difference is only the stake split.

The extreme magnitudes are partly due to late-market LTPs hitting
the 999.0 cap. Real training with the junk filter would not see
bets at those prices, but the distortion is still significant at
normal racing odds (e.g. 5.0 with divisor=4 gives a 32% delta for
placed-only runners).
