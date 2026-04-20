# Progress — Arb Curriculum

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows
`plans/reward-densification/progress.md` — "What landed",
"Not changed", "Gotchas", "Test suite", "Next".

---

## Session 01 — 2026-04-20

**Commit:** (see git log)

**What landed:**

- `env/exchange_matcher.py` — exported two pure functions:
  `passes_junk_filter(price, reference_price, max_dev_pct) -> bool` and
  `passes_price_cap(price, max_price) -> bool`. Used internally by the
  oracle. Existing class behaviour and all 33 matcher tests unchanged.
- `training/arb_oracle.py` — new module:
  - `OracleSample` dataclass (tick_index, runner_idx, obs, arb_spread_ticks,
    expected_locked_pnl).
  - `scan_day(date, data_dir, config) -> list[OracleSample]`: scans every
    pre-race tick via back-first arb check (junk filter → price cap →
    min_arb_ticks_for_profit → passive lay junk filter → lay price cap →
    freed-budget reservation → locked_pnl > 0). Uses a BetfairEnv for
    static obs (scalping obs v6). Samples sorted by (tick_index, runner_idx)
    for determinism.
  - `save_samples(...)`: writes `data/oracle_cache/{date}/oracle_samples.npz`
    + `header.json`. Atomic write (temp rename). Also writes
    `unique_arb_ticks_density` in the header.
  - `load_samples(date, data_dir, strict=True)`: loads cache; hard error
    on obs/action schema version mismatch (§9).
  - CLI: `python -m training.arb_oracle scan --date D [--dates D1,D2,...]`.
    Prints `samples=X ticks=Y density=X/Y unique_arb_ticks=A unique_arb_density=B`.
- `data/oracle_cache/` — added to `.gitignore`.
- `tests/arb_curriculum/test_arb_oracle.py` — 19 tests across the 8
  required categories (§27 of hard_constraints). 19 pass, 1 skipped
  (real-data obs-dim test skips cleanly when no processed data present).

**Not changed:** matcher behaviour, env schemas, reward path, PPO,
  controller, BetfairEnv observation_space, any training loop code.

**Gotchas:**
- `np.savez(path, ...)` appends `.npz` automatically. The temp file must
  be named with a `_tmp` stem, not a `.tmp` suffix — `with_suffix(".npz")`
  replaces the suffix, so `oracle_samples.tmp` → `.with_suffix(".npz")`
  → `oracle_samples.npz`, not `oracle_samples.tmp.npz`. Fixed by naming
  the stem `oracle_samples_tmp` and the temp file `oracle_samples_tmp.npz`.
- At price ~5.0 with 5% commission, `min_arb_ticks_for_profit` returns 9
  ticks (need to get from 5.0 to ~4.1). Tests at typical horse prices
  naturally produce samples without needing a crossed book.
- Pre-existing test failure: `test_session_4_9.py::TestStartEndpoint::
  test_start_returns_run_config` — confirmed pre-existing, unrelated.

**Test suite:** `pytest tests/arb_curriculum/ -v` → 19 passed, 1 skipped.
  Full suite → 1875 passed, 67 skipped, 1 pre-existing fail (session 4.9).

**Per-day densities (CLI output pending actual data run):**
  Oracle CLI not yet run against the training-date window because the
  training run may be active. Operator to run:
  `python -m training.arb_oracle scan --dates <dates>` after confirming
  no active training. Append results here. Flag any day with density < 0.001.

**Next:** Session 02 — Matured-arb bonus (knob at 0 default).

---

_Plan folder created 2026-04-19. See `purpose.md` for the
structural diagnosis flowing from the
`reward-densification-probe` 2026-04-19 failure and the
gene-sweep currently running: the policy finds "arb less"
before "arb better" because random arbing is expected-
negative. This plan attacks the local minimum via four
coordinated interventions: offline oracle scan, matured-arb
bonus, naked-loss annealing, BC pretrain + curriculum day
ordering._

_Operator observation 2026-04-19: one training day had
only 3 possible arbs across the whole day. That's bad
curriculum material at agent init — nothing for the policy
to imitate or exploit, but still ambient cost on any
random arbing. Curriculum day ordering (Session 05) uses
Session 01's oracle density to front-load arb-rich days._
