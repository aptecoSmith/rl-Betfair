# Session prompt — throughput-fix Session 01: rollout sync removal

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Phase 3's GPU pathway is wired correctly but does not saturate the
device — single-agent CUDA runs at **~145 s/episode** vs **~113 s/ep**
on CPU, a **1.28× regression**. Phase 3 Session 01b traced the cause
(`plans/rewrite/phase-3-cohort/findings.md` §"Bar 4 — Speed"): the
rollout loop hits ~5+ CUDA→CPU sync barriers per tick, and at
LSTM(h=128, batch=1) those barriers dominate compute. The hidden-
state CUDA→CPU round-trip was already removed in Session 01b; the
remaining syncs sit in `RolloutCollector._collect`.

Of those remaining syncs, two are **structurally required** (the
CPU env consumes `action_idx: int` and `stake_pounds: float` every
tick) and three are **deferrable** (the values land in `Transition`
fields that the PPO update consumes in batch at end-of-rollout).
This session cuts the deferrable three.

End-of-session bar:

1. **Self-parity holds.** Two CUDA runs at the same seed produce
   bit-identical `total_reward` and `value_loss_mean` — same shape
   as Phase 3 Session 01b's load-bearing Bar 1. CUDA↔CUDA bit-parity
   is the correctness guard; if it breaks, the speed work is
   shelved.
2. **CPU/CUDA action-histogram band stays ≤ ±5 %** on the same
   12-day eval slice as Phase 3 Session 01b. The CPU code path is
   unchanged in this session, so the band shouldn't move; verifying
   it didn't catches accidental cross-device behavioural drift.
3. **CUDA wall < 116 s/episode** (≥ 20 % speedup vs the 145 s/ep
   AMBER v2 single-agent baseline) on the same day as Phase 2 /
   Phase 3 Session 01b (2026-04-23). Honest target — the upper
   bound on what removing 3 of 5 sync barriers can deliver is
   ~50 % reduction in sync overhead, but the LSTM forward itself
   is also kernel-launch-bound, so 20–40 % is the realistic band.
4. **All pre-existing v2 trainer / rollout / collector tests pass**
   on CPU (byte-identical) and CUDA. Inherited from Sessions 01/01b
   — every change in `training_v2/` runs the existing 14+ tests.
5. **Verdict logged** as one of:
   - **GREEN**: speed bar (3) hit AND self-parity holds. Plan moves
     to Session 03 (verdict writeup) — Session 02 (batched cohort)
     becomes optional, not required.
   - **PARTIAL**: self-parity holds but speedup is < 20 %. Document;
     Session 02 is the critical path. Plan continues.
   - **FAIL**: self-parity breaks. Do not ship the change. Stop and
     triage. The speed measurement is not relevant if parity broke.

## What you need to read first

1. `plans/rewrite/phase-3-followups/throughput-fix/purpose.md` —
   this plan's goal, success bar, hard constraints, and the
   sync-barrier inventory it inherits from Phase 3.
2. `plans/rewrite/phase-3-cohort/findings.md` §"Session 01" and
   §"Session 01b" — the GPU pathway as it stands today, the
   parity-test design (CUDA↔CUDA bit-identical + CPU/CUDA
   action-histogram band), and the per-tick sync inventory that
   names the 5 sync sites you're targeting.
3. `training_v2/discrete_ppo/rollout.py` — the file you're modifying.
   Read all of `RolloutCollector._collect` (~line 100 to ~line 266).
   The five sync sites are concentrated in lines 190–225.
4. `training_v2/discrete_ppo/transition.py` — the `Transition`
   definition. You'll either widen its field types to accept torch
   tensors or introduce a sibling type for deferred fields. Read
   the existing field types and which ones the PPO update consumes
   (`log_prob_action`, `log_prob_stake`, `value_per_runner`).
5. `training_v2/discrete_ppo/trainer.py::_ppo_update` — search for
   how it stacks per-transition log-probs and value tensors into
   the batched forms it feeds the surrogate loss. Your refactor
   either preserves or consolidates that batching; either is fine
   as long as the PPO update consumes the same shapes.
6. `tests/test_v2_gpu_parity.py` — the existing parity test. Read
   the module-scoped fixture and the three tests; you'll either
   reuse it verbatim or extend it.
7. CLAUDE.md §"Recurrent PPO: hidden-state protocol on update" —
   the hidden-state contract you must NOT break. The state lives
   on-device as of Session 01b; this session does not touch it.
8. Phase 1 hidden-state contract (`plans/rewrite/phase-1-policy-
   and-env-wiring/findings.md` §"Hidden-state protocol") for
   reference on why batching anything that touches hidden state
   is hazardous (it's not in scope here, but a regression is
   easy to introduce).

## What to do

### 1. Pre-flight (~30 min)

- Confirm the AMBER v2 baseline single-agent CUDA wall by reading
  `plans/rewrite/phase-3-cohort/findings.md` Session 01b's table
  (5 episodes total → per-episode mean). Document the exact
  baseline numbers in the session note you're going to write.
- Run `pytest tests/test_v2_gpu_parity.py -v --runslow` (or the
  equivalent marker incantation that the repo uses) BEFORE making
  any changes. The three tests should PASS on the current code.
  This is your "known-good" reference. If they fail before you
  start, stop — there's a different bug to triage.
- Run `pytest tests/test_discrete_ppo_rollout.py
  tests/test_discrete_ppo_trainer.py
  tests/test_discrete_ppo_transition.py -v` and confirm all 14+
  tests pass on CPU. Same reason: known-good baseline.
- Re-read [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
  lines 190–225. Confirm the five sync sites (two structural +
  three deferrable) match Phase 3's inventory:

  | Line | Call | Goes to | Status |
  |---|---|---|---|
  | 191 | `int(action.item())` | env.step | STRUCTURAL |
  | 204 | `float(stake_unit_t.item())` | env.step (via budget × unit) | STRUCTURAL |
  | 193 | `out.action_dist.log_prob(action).item()` | Transition.log_prob_action | DEFERRABLE |
  | 217 | `stake_dist.log_prob(stake_unit_t).item()` | Transition.log_prob_stake | DEFERRABLE |
  | 222–225 | `out.value_per_runner.detach().squeeze(0).cpu().numpy()` | Transition.value_per_runner | DEFERRABLE |

  If your read disagrees with this table, **stop and reconcile
  with the operator**. The plan's hypothesis is built on this
  inventory; a wrong inventory invalidates the session.

### 2. Decide the buffering shape (~15 min)

Two viable shapes:

**(a) Per-tick device tensors held on `Transition`.** Extend
`log_prob_action`, `log_prob_stake`, and `value_per_runner` to
accept torch tensors (CUDA or CPU) in addition to the existing
float / numpy types. The PPO update materialises them by stacking
into batched tensors at the start of `_ppo_update` (it already
does this for the per-transition fields; this just changes the
input type from "scalar float" to "0-d torch tensor").

**(b) Sidecar buffers on the collector.** Keep `Transition`'s
fields as-is (CPU types). Add `_pending_log_prob_action`,
`_pending_log_prob_stake`, `_pending_value_per_runner` lists on
the collector that hold the device tensors. At end-of-episode,
do **one** batched `.cpu()` transfer per buffer, then write the
results back into the per-transition CPU fields.

Pick **(b)**. Reasons:

- `Transition` is consumed by `_ppo_update` and several tests;
  widening its field types ripples into every test
  (`test_discrete_ppo_transition.py` builds Transitions with
  explicit float fields and is the load-bearing schema test).
  Sidecar shape isolates the change to the rollout loop.
- The end-of-episode batched transfer is structurally cleaner —
  one CUDA→CPU sync per episode instead of three per tick × T
  ticks. At T = ~12 k that's 12 k × 3 = 36 k syncs collapsing
  to 3 syncs.
- (b) is also the shape Session 02 wants: the batched-rollout
  collector will hold per-agent-batched tensors and emit
  per-agent transitions at end-of-episode anyway. (a) would
  have to be partly undone for Session 02.

If during implementation (b) turns out infeasible (e.g. memory
spike from holding ~12 k device tensors), fall back to (a) but
flag this in the session note as a Session 02 risk.

### 3. Implement sync removal (~1.5 h)

In `training_v2/discrete_ppo/rollout.py::_collect`:

```python
# Pre-loop: allocate sidecar buffers.
pending_log_prob_action: list[torch.Tensor] = []  # 0-d device tensors
pending_log_prob_stake: list[torch.Tensor] = []
pending_value_per_runner: list[torch.Tensor] = []  # (N_runners,) device tensors

# Inside the loop, replace the three deferrable .item() / .cpu() calls
# with device-resident appends:

# action_dist.log_prob — keep on device
log_prob_action_t = out.action_dist.log_prob(action).detach()  # 0-d, device
pending_log_prob_action.append(log_prob_action_t)

# stake log-prob — keep on device, masked by uses_stake later
if action_uses_stake(self.action_space, action_idx):
    log_prob_stake_t = stake_dist.log_prob(stake_unit_t).detach()  # 0-d, device
else:
    log_prob_stake_t = torch.zeros((), device=self.device)
pending_log_prob_stake.append(log_prob_stake_t)

# value_per_runner — keep on device
value_per_runner_t = out.value_per_runner.detach().squeeze(0)  # (N_runners,), device
pending_value_per_runner.append(value_per_runner_t)

# action.item() and stake_unit_t.item() STAY — env.step needs them on CPU.
action_idx = int(action.item())          # STRUCTURAL — keep
stake_unit = float(stake_unit_t.item())  # STRUCTURAL — keep
```

Then at end-of-episode (after the `while not done` loop, before
the `return transitions` line), do the batched materialisation:

```python
# One batched CUDA→CPU transfer per buffer. Three syncs total
# instead of 3 × T per-tick syncs.
if pending_log_prob_action:
    log_prob_action_arr = torch.stack(pending_log_prob_action).cpu().numpy()
    log_prob_stake_arr = torch.stack(pending_log_prob_stake).cpu().numpy()
    value_per_runner_arr = torch.stack(pending_value_per_runner).cpu().numpy().astype(np.float32)
else:
    # Empty episode — defensive; shouldn't happen but the env technically allows it.
    log_prob_action_arr = np.zeros((0,), dtype=np.float32)
    log_prob_stake_arr = np.zeros((0,), dtype=np.float32)
    value_per_runner_arr = np.zeros((0, 0), dtype=np.float32)

# Backfill the Transition objects.
for i, t in enumerate(transitions):
    t.log_prob_action = float(log_prob_action_arr[i])
    t.log_prob_stake = float(log_prob_stake_arr[i])
    t.value_per_runner = value_per_runner_arr[i]
```

The two structural `.item()` calls (action_idx, stake_unit) stay
exactly where they are. The hidden-state capture path is
unchanged from Session 01b (`hidden_in_t = tuple(t.detach().clone()
for t in hidden_state)` stays).

**Note on `Transition` mutability.** If `Transition` is a frozen
dataclass, the backfill loop fails. Two options:

- Unfreeze it (least disruptive — the trainer already mutates
  some fields elsewhere; check first).
- Build the Transition objects AFTER the backfill (rearrange the
  rollout loop to defer Transition construction until end-of-
  episode). This is the cleaner shape long-term and is what
  Session 02 will need anyway. Pick this if Transition is frozen.

### 4. Tests (~30 min)

In `tests/test_v2_rollout_sync.py` (NEW), three tests:

```python
@pytest.mark.gpu
@pytest.mark.slow
def test_cuda_self_parity_after_sync_removal():
    """Two CUDA runs at the same seed produce bit-identical
    total_reward and value_loss_mean."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device")
    # Run main(...) twice with --device cuda --seed 42, 5 episodes,
    # one day. Assert per-episode total_reward and value_loss_mean
    # are exactly equal (diff = 0.0).

def test_action_idx_and_stake_unit_still_materialise_per_tick():
    """The two structural .item() calls remain — they're load-
    bearing for the CPU env. A regression where a refactor
    accidentally batches the env step would defer these too,
    breaking the env contract.

    Spy on int() and float() inside the rollout loop OR check the
    env step logs for non-int / non-float types. The simplest
    realisation: assert action_idx is `int` and stake_pounds is
    `float` at the env.step call site by patching env.step with
    a wrapper that asserts types and re-dispatches."""

def test_transition_fields_match_pre_plan_within_fp32_eps():
    """Run a 1-episode rollout on CPU with the pre-plan code path
    (manually invoked by reverting the change in a fork OR by
    holding a reference output) and the post-plan code path. Assert
    transition.log_prob_action, .log_prob_stake, .value_per_runner
    match within 1e-6 absolute. CPU code is unchanged so this
    should be exact, not just close."""
```

The third test is the safety net: if the refactor accidentally
changes which tick the log_prob is read at (e.g. before vs after
hidden-state update), this catches it. Make the assertion EXACT
(`assert a == b`) on CPU, not "close" — the CPU code path doesn't
change in this session.

Run:

```
pytest tests/test_v2_rollout_sync.py -v --runslow
pytest tests/test_v2_gpu_parity.py -v --runslow
pytest tests/test_discrete_ppo_rollout.py
        tests/test_discrete_ppo_trainer.py
        tests/test_discrete_ppo_transition.py -v
```

All three command groups should pass. If `test_v2_gpu_parity.py`'s
existing CUDA↔CUDA self-parity test fails, **stop**. The refactor
broke parity; speed-bar measurement is irrelevant.

### 5. Speed measurement (~1 h, ~10 min × 2 runs + analysis)

Run two 5-episode CUDA jobs on the same day used by Session 01b:

```
python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --n-episodes 5 \
    --seed 42 \
    --device cuda \
    --out logs/discrete_ppo_v2/throughput_session01_run_a.jsonl

python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --n-episodes 5 \
    --seed 42 \
    --device cuda \
    --out logs/discrete_ppo_v2/throughput_session01_run_b.jsonl
```

Capture wall time per episode (the existing log lines emit
"Episode N/5 ... wall=X.Xs" in the JSONL or stdout; check the
post-Session-01b emitter).

Comparison vs Session 01b's CUDA mean (145.5 s/ep):

| Run | Per-episode wall | Δ vs AMBER v2 (145 s/ep) |
|---|---|---|
| AMBER v2 (Session 01b) | 145.5 s | — |
| Session 01 CUDA-a | ? | ? |
| Session 01 CUDA-b | ? | ? |

The two CUDA runs at the same seed should be bit-identical on
total_reward (per Bar 1) and within wall-time noise (~5 % run-to-
run on a quiet GPU). If wall times disagree by > 20 % between A
and B, something is off (background load, thermal throttling,
non-deterministic kernel hitting the new path) — investigate
before drawing conclusions.

### 6. Optional but recommended — `cProfile` snapshot (~30 min)

If the speed bar is a near-miss (CUDA wall in the 116–130 s/ep
range — partial speedup but not enough), profile a 1-episode
run to identify the next biggest cost:

```
python -m cProfile -o /tmp/v2_rollout.prof -m training_v2.discrete_ppo.train \
    --day 2026-04-23 --n-episodes 1 --seed 42 --device cuda \
    --out /tmp/v2_rollout_cprof.jsonl

python -c "import pstats; p = pstats.Stats('/tmp/v2_rollout.prof'); p.sort_stats('cumulative').print_stats(40)"
```

Top-40 cumulative time. Look for:

- `cuda` kernel launch overhead (irreducible at batch=1; needs
  Session 02 batching).
- env.step (CPU, irreducible without env changes — out of scope).
- Anything in `rollout.py` that wasn't on the original 5-syncs
  list (might be a new sync introduced by the refactor —
  triage).

Capture the top-10 in the session findings note. This becomes
the entry point for Session 02's prompt.

### 7. Findings note (~15 min)

Append to
`plans/rewrite/phase-3-followups/throughput-fix/findings.md`
(create if missing — empty file with `# Throughput-fix —
findings` header):

```markdown
## Session 01 — rollout sync removal

**Status:** GREEN | PARTIAL | FAIL

### Speed table

| Run | Per-episode wall | Δ vs AMBER v2 |
|---|---|---|
| AMBER v2 baseline | 145.5 s | — |
| Session 01 CUDA-a | X.X s | -Y% |
| Session 01 CUDA-b | X.X s | -Y% |

### Self-parity

CUDA↔CUDA self-parity: PASS (diff = 0.0 on all 5 episodes).
CPU/CUDA action-histogram band: PASS (max drift < 5 %).

### What changed
- (One-paragraph summary of the refactor.)

### What's next
- (GREEN: jump to Session 03 writeup. PARTIAL: open Session 02
  prompt with the cProfile top-10 as the gap-to-close.
  FAIL: triage parity break.)
```

## Stop conditions

- **CUDA↔CUDA self-parity test FAILS** → stop, triage. The most
  likely cause is the refactor accidentally changing the order of
  RNG draws (e.g. if `stake_dist.log_prob` got moved relative to
  `stake_dist.sample` and now consumes RNG state in a different
  order). Re-read the rollout loop; the sample()s and log_prob()s
  must execute in the same order pre- and post-refactor.
- **Pre-existing v2 trainer / rollout / transition tests fail** →
  the refactor leaked into the schema. Roll back to a state where
  pre-existing tests pass before pushing on the speed measurement.
- **Speed measurement shows CUDA wall > 145 s/ep** (regression) →
  the refactor introduced a NEW per-tick sync. Likely candidate:
  a `torch.stack` or `torch.cat` inside the loop instead of at
  end-of-episode. Profile and fix.
- **`Transition` is frozen and unfreezing breaks 3+ unrelated
  tests** → switch to "build Transition at end-of-episode" path
  (option (b) of step 3 — defer Transition construction). Don't
  unfreeze a frozen dataclass that has explicit downstream
  immutability invariants.
- **Past 5 h excluding speed-measurement wall** → stop and check
  scope. The refactor is small; if it's taking that long, there's
  a hidden-state contract bug (unlikely if you didn't touch the
  hidden-state path) or a Transition-shape ripple that needs a
  cleaner cut.

## Hard constraints

Inherited from
`plans/rewrite/phase-3-followups/throughput-fix/purpose.md`
§"Hard constraints" plus:

1. **No env edits.** `env/betfair_env.py`, `env/bet_manager.py`,
   `env/exchange_matcher.py` are off-limits. If a perf opportunity
   sits inside the env, file a follow-on plan; do not bundle it
   into this session.
2. **No reward-shaping changes.** Untouched: `reward.*` config,
   matured-arb bonus, naked-loss handling, mark-to-market,
   open-cost shaping.
3. **Hidden-state contract is unchanged.** The `hidden_state_in`
   capture pattern from Session 01b
   (`tuple(t.detach().clone() for t in hidden_state)` BEFORE the
   forward pass) stays exactly as it is. Don't move it, batch it,
   or refactor it. CLAUDE.md §"Recurrent PPO: hidden-state protocol
   on update" is the contract; breaking it inflates `approx_kl`
   by orders of magnitude on the first PPO update after the change.
4. **The two structural `.item()` calls stay.** `int(action.item())`
   and `float(stake_unit_t.item())` are required by the CPU env.
   Removing them silently is a correctness break (env receives a
   tensor and either errors loudly or — worse — coerces it badly).
5. **Same `--seed 42` for every measurement.** Cross-run
   comparisons against Session 01b's baseline rest on this.
6. **`cudnn.deterministic = True` stays on.** Plan-level constraint
   inherited from purpose.md.
7. **No GA gene additions.** This session ships zero new genes.
8. **No re-import of v1 trainer / policy / rollout.** Phase 2/3
   hard constraint inherited verbatim.
9. **Self-parity is the load-bearing correctness guard.** A 30 %
   speedup that breaks it is not shipped.

## Out of scope

- Multi-agent batched forward (Session 02).
- AMP / autocast.
- Env vectorisation.
- `cudnn.benchmark = True`.
- Multi-GPU training.
- 12-agent cohort wall measurement (Session 02 owns this; this
  session measures single-agent wall only).
- 66-agent scale-up.
- v1 deletion.
- Reward-shape iteration.

## Useful pointers

- Per-tick sync sites:
  [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
  lines 190–225.
- Transition definition:
  [`training_v2/discrete_ppo/transition.py`](../../../../training_v2/discrete_ppo/transition.py).
- PPO update consumer:
  [`training_v2/discrete_ppo/trainer.py`](../../../../training_v2/discrete_ppo/trainer.py)
  search for `_ppo_update`.
- Existing parity test:
  [`tests/test_v2_gpu_parity.py`](../../../../tests/test_v2_gpu_parity.py).
- Pre-existing v2 tests:
  [`tests/test_discrete_ppo_rollout.py`](../../../../tests/test_discrete_ppo_rollout.py),
  [`tests/test_discrete_ppo_trainer.py`](../../../../tests/test_discrete_ppo_trainer.py),
  [`tests/test_discrete_ppo_transition.py`](../../../../tests/test_discrete_ppo_transition.py).
- Phase 3 GPU pathway findings:
  `plans/rewrite/phase-3-cohort/findings.md` §"Session 01" and
  §"Session 01b".
- Hidden-state contract: CLAUDE.md §"Recurrent PPO: hidden-state
  protocol on update".
- v1 GPU pinning reference (read, do not import):
  [`agents/ppo_trainer.py`](../../../../agents/ppo_trainer.py)
  lines 2131–2174.

## Estimate

3 h:

- 30 min: pre-flight (baseline test runs, sync inventory check).
- 15 min: buffering-shape decision.
- 1.5 h: refactor + tests.
- 1 h: speed measurement (2 × 5-episode CUDA runs at ~12 min each,
  plus analysis).
- 30 min: optional cProfile + findings note.

If past 5 h excluding the cohort-wall measurement, stop and check
scope.
