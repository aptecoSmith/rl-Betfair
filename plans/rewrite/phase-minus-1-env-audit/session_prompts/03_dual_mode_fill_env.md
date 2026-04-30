# Session prompt — dual-mode passive-fill env (volume / pragmatic)

Use this prompt to open a new session in a fresh context. The
prompt is self-contained — it briefs you on the question, the
context, the design, and the constraints. Do not require any
context from the session that scaffolded it.

---

## The question

**Make `PassiveOrderBook` fill-rate mechanics flexible to the
data the day was captured with: spec-faithful volume gating
when per-runner cumulative `total_matched` is populated,
pragmatic market-level prorated volume gating when it isn't.
Auto-detect per day; never mix modes mid-day; surface the mode
in telemetry.**

## Why

Phase −1 env audit finding F7
(`plans/rewrite/phase-minus-1-env-audit/audit_findings.md`)
discovered that historical parquets in `data/processed/*.parquet`
have `RunnerSnap.total_matched == 0` on every active runner of
every tick. The upstream `StreamRecorder1\BetfairPoller` poller
never captured per-runner cumulative volume, so the data was
lost at capture time. **Reprocessing existing parquets cannot
recover this field — the value was never written to MySQL in
the first place.**

A separate session (`02_streamrecorder_per_runner_volume.md`)
fixes the upstream poller. Days polled AFTER that fix lands
will carry valid per-runner volume; historical days never
will.

This creates a curriculum problem. We have ~200+ days of
historical data the rewrite needs to train on. Three options
exist (per the audit's RED-verdict triage):

- **A. Wait for fresh data.** 4–6 weeks of slip before Phase 0
  has a usable curriculum.
- **B. PRO Historical backfill.** Engineering project to
  download and parse Betfair's archived stream tapes.
- **C. Pragmatic fill-model fallback.** Adapt the simulator to
  the data that IS available on historical days.

The operator chose **C, but flexibly**: when post-fix data is
present, use the spec-faithful volume model; when it isn't,
fall back to a pragmatic model that uses signals already in
the data. That's this session.

## Design

### Mode selection — per-day, at load, deterministic

Decision point is `Day` / episode load. Once chosen, the mode
is fixed for the entire day. No mid-race mode-switching.

```python
# At Day load time (data/episode_builder.py):
day.fill_mode: Literal["volume", "pragmatic"] = (
    "volume"
    if any(
        r.total_matched > 0
        for race in day.races
        for tick in race.ticks
        for r in tick.runners
        if r.status == "ACTIVE"
    )
    else "pragmatic"
)
```

### Mode dispatch — `PassiveOrderBook.on_tick`

```python
def on_tick(self, tick, tick_index):
    if self._fill_mode == "volume":
        self._volume_phase_1(tick)        # current code, unchanged
    else:
        self._pragmatic_phase_1(tick)     # new code path
    self._phase_2_fill_check(tick, tick_index)  # shared, unchanged
```

Phase 2 (the actual fill check — junk-band filter +
threshold check) is **unchanged**. Both modes feed
`order.traded_volume_since_placement`; only the source of the
delta differs.

### Pragmatic Phase 1 — what the new code looks like

```python
def _pragmatic_phase_1(self, tick):
    # Market-level traded_volume IS populated on historical data
    # (£100k–£5M per race). Sum across runners, prorate by ladder
    # size weight, apply the same crossability gate as volume mode.
    market_tv = tick.traded_volume
    market_delta = max(0.0, market_tv - self._prev_market_tv)
    self._prev_market_tv = market_tv
    if market_delta <= 0.0:
        return

    # Build runner weights by total visible book size (back + lay).
    # Runners with thicker books trade proportionally more.
    runner_weights: dict[int, float] = {}
    total_visible = 0.0
    for r in tick.runners:
        if r.status != "ACTIVE":
            continue
        size = (
            sum(lv.size for lv in r.available_to_back)
            + sum(lv.size for lv in r.available_to_lay)
        )
        runner_weights[r.selection_id] = size
        total_visible += size
    if total_visible <= 0.0:
        return

    runner_by_sid = {r.selection_id: r for r in tick.runners}
    for sid, sid_orders in self._orders_by_sid.items():
        if not sid_orders:
            continue
        weight = runner_weights.get(sid, 0.0) / total_visible
        synth_delta = market_delta * weight
        if synth_delta <= 0.0:
            continue
        snap = runner_by_sid.get(sid)
        if snap is None:
            continue
        ltp = snap.last_traded_price
        for order in sid_orders:
            # Crossability gate — same logic as volume mode (commit 4ee9fb5).
            if ltp is None or ltp <= 0.0:
                continue
            if order.side is BetSide.LAY and ltp > order.price:
                continue
            if order.side is BetSide.BACK and ltp < order.price:
                continue
            order.traded_volume_since_placement += synth_delta
```

Same crossability gate, same threshold semantics, same junk
filter — only the source of `delta` differs. Pragmatic mode
shares the spec's structural form, just with synthetic
attribution rather than per-runner truth.

### Telemetry — load-bearing, do not skip

The mode the day used MUST be visible everywhere a downstream
reader could compare runs across modes. Without this, cohort
metrics blend silently and nobody can tell why force-close
rates differ between two runs.

Per-tick `info` dict:
```python
info["fill_mode_active"] = self._fill_mode  # "volume" | "pragmatic"
```

Per-episode JSONL row (`logs/training/episodes.jsonl`):
```python
{"fill_mode": "pragmatic", ...}
```

`RaceRecord` (`env/betfair_env.py`):
```python
@dataclass
class RaceRecord:
    ...
    fill_mode: str = "volume"  # set per-race from day.fill_mode
```

Operator log line — extend per-episode summary to include the
mode:
```
Episode 3/9 [2026-04-02] reward=+12.345 (raw=+4.567 shaped=+7.778)
  pnl=+4.57 mode=pragmatic ...
```

### `Day` object change

Add `fill_mode: Literal["volume", "pragmatic"]` to the `Day`
dataclass at `data/episode_builder.py`. Computed in
`_build_day` after all ticks are loaded. Wire through
`BetfairEnv.__init__` so `PassiveOrderBook` reads the right
mode.

## What to do

### 1. Read the surrounding code (~15 min)

- `env/bet_manager.py::PassiveOrderBook.on_tick` — the
  current volume-mode code path (lines ~643–828).
- `data/episode_builder.py::_build_day` — where `Day` objects
  are constructed.
- `env/betfair_env.py::reset` — where `BetManager` /
  `PassiveOrderBook` are instantiated per race.
- `tests/test_per_runner_total_matched_data.py` — the F7
  regression guards. These must continue to FAIL on historical
  data (volume-mode requires real data; the regression guards
  document that requirement).

### 2. Implement (~2 hours)

In order:

a. **`Day.fill_mode` auto-detect.** Update `Day` dataclass +
   `_build_day` in `data/episode_builder.py`. Default
   `"pragmatic"`; flip to `"volume"` iff any active runner on
   any tick of any race has `total_matched > 0`.

b. **Plumb through to `PassiveOrderBook`.** Pass the day's
   `fill_mode` into `BetfairEnv` (constructor or per-race),
   thread into `BetManager` → `PassiveOrderBook`. Stash on
   `self._fill_mode`. Default `"volume"` for stub /
   synthetic tests so existing tests don't regress.

c. **Refactor existing Phase 1** into
   `PassiveOrderBook._volume_phase_1`. No logic change — just
   move the existing accumulation block into a method.

d. **Implement `PassiveOrderBook._pragmatic_phase_1`.** Per
   the design above. Cache `_prev_market_tv` (per-runner
   `_last_total_matched` is already cached for volume mode;
   pragmatic mode caches the market-level scalar instead).

e. **Mode dispatch in `on_tick`.** Top of method, dispatch to
   one Phase 1 path or the other. Phase 2 unchanged.

f. **Telemetry.** Per-tick `info["fill_mode_active"]`,
   `RaceRecord.fill_mode`, episode JSONL `fill_mode`,
   operator log `mode=...`.

### 3. Tests (~60 min)

In `tests/test_passive_order_book_dual_mode.py` (new):

- `test_volume_mode_unchanged_on_synthetic_tick` — feed a
  synthetic tick with non-zero per-runner `total_matched` AND
  set `fill_mode="volume"`. Assert exactly the same fill
  behaviour as before this plan. Locks the contract that
  spec-faithful mode is byte-identical to pre-plan.
- `test_pragmatic_mode_attributes_market_volume_to_runners` —
  feed a synthetic tick with `total_matched=0` per runner but
  `tick.traded_volume_delta=£10000` and a `fill_mode="pragmatic"`
  passive resting at the visible top. Assert the order's
  `traded_volume_since_placement` advances by the prorated
  share.
- `test_pragmatic_mode_respects_crossability_gate` — same
  setup but lay resting at price below LTP. Assert no
  accumulation despite market delta.
- `test_pragmatic_mode_zero_total_visible_no_accumulation` —
  edge case: empty books on every runner. Assert no
  accumulation, no crash.
- `test_day_fill_mode_auto_detects_volume_when_any_runner_nonzero` —
  build a `Day` with one tick where one runner has
  `total_matched > 0`. Assert `day.fill_mode == "volume"`.
- `test_day_fill_mode_auto_detects_pragmatic_when_all_zero` —
  same but all `total_matched == 0`. Assert
  `day.fill_mode == "pragmatic"`.
- `test_telemetry_surfaces_fill_mode_in_info` — drive one
  step in each mode, assert `info["fill_mode_active"]` matches.

In `tests/test_episode_builder.py` (extend existing):

- `test_build_day_sets_fill_mode_from_synthetic_data` — round-
  trip a synthetic ticks DataFrame with non-zero
  `TradedVolume` and assert the loaded `Day.fill_mode ==
  "volume"`.

The pre-existing F7 regression tests in
`tests/test_per_runner_total_matched_data.py` MUST continue to
fail on real historical data — don't change them. They
document the data limitation, not the env behaviour.

### 4. Validate against the existing suite (~10 min)

```
python -m pytest tests/ -q
```

All pre-existing tests must still pass. The two F7 regression
guards still fail on real data (correctly — they're testing
data, not env). Total expected: 348 + new pragmatic-mode tests,
2 still failing F7 guards.

### 5. Write up (~15 min)

`plans/rewrite/phase-minus-1-env-audit/session_03_findings.md`:

- Implementation summary (file:line of each change).
- Test results (count of pre-existing pass / new pass / F7
  guards still failing).
- Verification: pick one historical day's parquet, load via
  `episode_builder`, confirm `day.fill_mode == "pragmatic"`.
  Run a stub policy through one race in pragmatic mode,
  confirm fills happen and `info["fill_mode_active"]` is
  surfaced correctly.
- Hand-off note: Session 04 (UI surfacing) reads from this
  telemetry. Don't rename or remove the field names without
  coordinating.

## Hard constraints

- **Don't mix modes mid-day.** Mode is fixed per-`Day` at load.
  No per-race or per-tick switching. Mid-mode changes corrupt
  training signal.
- **Don't change Phase 2.** The junk-band filter + threshold
  check is shared. Both modes feed the same downstream
  mechanic.
- **Don't change the F7 regression test.** Those tests
  document the data gap; their failures are intentional. If
  they start passing on historical data after this plan, the
  test is wrong and the audit's RED verdict was wrong — neither
  should change quietly.
- **Default to `"volume"` for synthetic / stub tests.**
  Existing `tests/test_*.py` build `Tick` / `RunnerSnap` with
  hardcoded non-zero `total_matched`. They expect volume-mode
  behaviour. Don't break them.
- **Telemetry is mandatory.** `info["fill_mode_active"]`,
  episode JSONL `fill_mode`, `RaceRecord.fill_mode` —
  Session 04 (UI surfacing) and any future cohort analysis
  depend on these fields. If you skip telemetry the work is
  not done.
- **Reward magnitudes change.** Pragmatic-mode passive fill
  timing differs from volume-mode by construction. Cohort
  scoreboards comparing pre-plan rows to post-plan rows are
  ONLY valid within-mode. Document this in the lessons-learnt
  write-up, do not silently break comparability.

## Out of scope

- **UI surfacing** — Session 04 (`04_dual_mode_fill_ui.md`)
  handles this. Don't touch the frontend or API in this
  session.
- **Changing the trainer to filter curriculum days by mode**
  — the env supports both; the trainer's curriculum-selection
  logic is a separate concern and a separate session.
- **PRO Historical backfill** — Option B from the audit; not
  this session.
- **Spec rewrites** — `docs/betfair_market_model.md` is fine
  as-is. The pragmatic mode is a documented approximation
  with telemetry; it doesn't require a spec change.
- **Anything in `StreamRecorder1`** — that's Session 02's
  scope.

## Useful pointers

- **F7 root analysis:** `plans/rewrite/phase-minus-1-env-audit/audit_findings.md`
  finding F7.
- **F7 regression test:** `tests/test_per_runner_total_matched_data.py`.
- **Current volume-mode code:** `env/bet_manager.py::PassiveOrderBook.on_tick`
  lines ~643–828.
- **Current `Day` / `Race` / `Tick` dataclasses:**
  `data/episode_builder.py` lines ~30–230.
- **Current env wire-up:** `env/betfair_env.py::reset` and
  the `BetManager` instantiation around line 1644.
- **CLAUDE.md "Order matching"** — load-bearing rules that
  apply to BOTH modes (no walking, junk filter, hard cap).
  These are unchanged.
- **Sample historical parquet for verification:**
  `data/processed/2026-04-11.parquet`, market `1.256488956`
  (33 active runners, £4.9M market traded volume, 222 ticks
  — heavy enough to exercise pragmatic-mode prorating).

## Estimate

Single session, 3.5–4 hours.

- 15 min: read surrounding code.
- 2 hours: implement (a–f above).
- 60 min: tests.
- 10 min: validate full suite.
- 15 min: write up.

If you're heading toward 5+ hours, stop and write up where
you are. Don't try to also tackle Session 04's UI work in
this session — they're explicitly separated for a reason.
