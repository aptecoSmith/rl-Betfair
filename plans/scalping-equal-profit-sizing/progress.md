# Progress — Scalping Equal-Profit Sizing

One entry per completed session. Most recent at the top.

---

_(No completed sessions yet. Plan folder created 2026-04-18 in
response to the operator spotting that an activity-log line
"Arb completed: Back £8.20 / Lay £6.00 → locked £+0.08" was
reporting a wildly under-locked balance for what should have been
a healthy scalp. Diagnosis: the sizing formula
`S_lay = S_back × P_back / P_lay` is correct only at zero
commission; at Betfair's 5 % it equalises *exposure*, not P&L,
producing pairs whose worst-case floor (which `locked_pnl`
reports) collapses near zero. The correct formula derives from
setting `total_win == total_lose` after commission and produces
genuinely balanced pairs. See `purpose.md` for the full
derivation, worked example, and reward-scale-change protocol.)_
