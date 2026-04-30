# Session prompt — Phase 3, Session 01b: parity reformulation + per-tick GPU sync fix

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Session 01 wired the v2 trainer to CUDA but **failed two of three
bars**:

- Bar 2 (CPU/CUDA parity ≤ 1e-5): FAIL with diff 62.07 at episode 0.
- Bar 3 (CUDA < 57 s/episode): FAIL — CUDA at 142.7 s/episode is
  *slower* than CPU at 108.8 s/episode.

`plans/rewrite/phase-3-cohort/findings.md` "Session 01 — GPU
pathway" has the full diagnostic. The two failures have independent
root causes:

1. **Parity is unachievable bit-wise.** torch's CPU and CUDA RNG
   streams diverge from tick 0; FP-epsilon matmul drift compounds
   over ~12 k forward passes / episode. The session-01 prompt's
   1e-5 target was wrong — it modelled parameter-update drift, not
   stochastic-action sampling on different RNG streams.
2. **CUDA is slower than CPU due to per-tick sync.**
   `training_v2/discrete_ppo/rollout.py:160` does
   `t.detach().clone().cpu().numpy()` on each LSTM hidden-state
   tensor every tick — ~24 k forced CUDA→CPU sync barriers /
   episode. Tiny LSTM (h=128, batch=1) forwards are
   GPU-overhead-bound regardless; the buffer-reuse savings the
   session targeted are dwarfed by these syncs.

This session does both fixes together because the new parity bar is
needed to validate the speed fix.

End-of-session bar (revised — see "Why these bars" below):

1. CUDA↔CUDA same-seed reproducibility: two CUDA runs with the
   same seed produce bit-identical `total_reward` and
   `value_loss_mean` per episode (1e-7 abs tolerance — float32
   epsilon, not 1e-5 stochastic drift).
2. CPU↔CUDA distributional band: action histogram per type
   within ±5 % of total ticks; `total_reward` within ±15 % of CPU
   value per episode. Honest about what's achievable across RNG
   streams.
3. CUDA wall < 30 s / episode (≤ 28 % of CPU's 108.8 s/episode —
   tightened from session-01's 57 s because the per-tick sync was
   the dominant cost; with it removed, the kernel-launch ceiling
   on this LSTM size is closer to 10–20 s/episode).

## Why these bars

The original prompt's parity bar conflated two different drift
sources:

- **Parameter-update drift** between CPU and CUDA matmul: this is
  ~1e-7 per matmul, ~1e-5 cumulative over ~744 mini-batches × 5
  episodes. Achievable.
- **Stochastic-action drift** from divergent RNG streams: this is
  100 % from tick 0. Every action sampled is a categorical /
  Beta draw; CPU and CUDA generators are independent. Even moving
  sampling to CPU only delays the divergence — matmul drift still
  tips near-tied probabilities to different buckets after a few
  thousand ticks.

Same-seed CUDA↔CUDA reproducibility is the contract that *actually
matters* for cohort runs (does seed=42 produce the same answer
twice on this GPU). The CPU↔CUDA band check catches catastrophic
device-handshake bugs (e.g. wrong-device hidden state, dtype
mismatch) — those would show up as +50 % shift in
action histogram, not as 1e-5 drift in total_reward.

## What you need to read first

1. `plans/rewrite/phase-3-cohort/findings.md` "Session 01 — GPU
   pathway" — the full diagnostic for what failed and why.
2. `plans/rewrite/phase-3-cohort/purpose.md` — phase-level success
   bar. Note that the speed bar in purpose.md still references
   the old "< 57 s/episode" number; this session lives with that
   for now and the cohort-run target stays correct (5–10× speed-up
   over CPU).
3. `training_v2/discrete_ppo/rollout.py:140-180` — the per-tick
   hidden-state capture path that needs to defer its sync.
4. `training_v2/discrete_ppo/transition.py` — `Transition`
   dataclass; `hidden_state_in` is a `tuple[np.ndarray, ...]`
   today and will become `tuple[torch.Tensor, ...]` (device-
   resident).
5. `training_v2/discrete_ppo/trainer.py:400-430` — where the
   trainer reads `hidden_state_in` and converts back through
   `torch.from_numpy`. After this session that round-trip
   disappears.
6. `tests/test_v2_gpu_parity.py` — the parity test. Replace the
   1e-5 assertion with the two new bars.
7. `agents_v2/discrete_policy.py:253-272` — `pack_hidden_states`
   already takes torch tensors as input, so no change there.
8. `CLAUDE.md` §"Recurrent PPO: hidden-state protocol on update" —
   the contract is "the state passed INTO the forward pass that
   produced this transition's log-prob". Storage location (CPU
   numpy vs device tensor) is implementation detail; the contract
   is unchanged.

## What to do

### 1. Defer the hidden-state sync (~45 min)

Change `Transition.hidden_state_in` from
`tuple[np.ndarray, np.ndarray]` to
`tuple[torch.Tensor, torch.Tensor]`. The tensors are device-
resident clones captured before the forward pass.

In `rollout.py`'s `_collect`, replace:

```python
hidden_in_np = tuple(
    t.detach().clone().cpu().numpy() for t in hidden_state
)
```

with:

```python
hidden_in_t = tuple(
    t.detach().clone() for t in hidden_state  # stays on device
)
```

The `.clone()` is still load-bearing — without it the tensor would
mutate as the LSTM rolls forward across ticks. `.detach()` ensures
no autograd tape leaks into rollout-time tensors.

In `trainer.py::_ppo_update`, the existing block

```python
hidden_pairs = [
    tuple(torch.from_numpy(arr) for arr in tr.hidden_state_in)
    for tr in transitions
]
packed_hidden_cpu = self.policy.pack_hidden_states(hidden_pairs)
packed_hidden = tuple(_move_to_device(t, device) for t in packed_hidden_cpu)
```

becomes simply:

```python
hidden_pairs = [tr.hidden_state_in for tr in transitions]
packed_hidden = self.policy.pack_hidden_states(hidden_pairs)
# Already on device — no transfer needed.
```

`_bootstrap_value` similarly stops `torch.from_numpy(arr).to(device)`
on `final_transition.hidden_state_in` and just uses the tensors.

**Memory budget check.** ~12 k transitions × 2 tensors × num_layers
(=1) × hidden (=128) × float32 = ~12 MB GPU memory. Trivially
within budget.

**Test impact.** Anywhere that constructs `Transition` directly
(`tests/test_discrete_ppo_trainer.py`,
`tests/test_discrete_ppo_rollout.py`,
`tests/test_discrete_ppo_transition.py`) needs to pass torch
tensors instead of numpy arrays. Update the helpers, not the
contract.

### 2. Reformulate the parity test (~30 min)

Rewrite `tests/test_v2_gpu_parity.py` with three tests, all
`@pytest.mark.gpu @pytest.mark.slow`:

```python
def test_cuda_self_parity_5_episodes(tmp_path):
    """Two CUDA runs with same seed produce bit-identical results."""
    out_a = tmp_path / "cuda_a.jsonl"
    out_b = tmp_path / "cuda_b.jsonl"
    main(day_str="2026-04-23", ..., seed=42, out_path=out_a, device="cuda")
    main(day_str="2026-04-23", ..., seed=42, out_path=out_b, device="cuda")
    a_rows = ...; b_rows = ...
    for ra, rb in zip(a_rows, b_rows):
        assert abs(ra["total_reward"] - rb["total_reward"]) < 1e-7
        assert abs(ra["value_loss_mean"] - rb["value_loss_mean"]) < 1e-7

def test_cpu_cuda_action_histogram_band(tmp_path):
    """CPU and CUDA action histograms within ±5% of total ticks."""
    # Reuse cpu_out and cuda_out from the existing helper or
    # generate fresh.
    for cpu_row, cuda_row in zip(cpu_rows, cuda_rows):
        n = cpu_row["n_steps"]
        for action in ("NOOP", "OPEN_BACK", "OPEN_LAY", "CLOSE"):
            cpu_count = cpu_row["action_histogram"].get(action, 0)
            cuda_count = cuda_row["action_histogram"].get(action, 0)
            assert abs(cpu_count - cuda_count) / n < 0.05, (
                f"action {action} drift {abs(cpu_count - cuda_count) / n:.1%}"
            )

def test_cpu_cuda_total_reward_band(tmp_path):
    """CPU and CUDA total_reward within ±15% per episode."""
    for cpu_row, cuda_row in zip(cpu_rows, cuda_rows):
        cpu = cpu_row["total_reward"]
        cuda = cuda_row["total_reward"]
        # Use abs(cpu) as the denominator; episode rewards are
        # comfortably non-zero in this regime so no zero-division
        # guard needed, but assert it just in case.
        assert abs(cpu) > 1.0
        rel_diff = abs(cpu - cuda) / abs(cpu)
        assert rel_diff < 0.15, (
            f"total_reward rel diff {rel_diff:.1%} (cpu={cpu:.2f} cuda={cuda:.2f})"
        )
```

Share the run-the-trainer setup across the three tests via a
session-scoped fixture (compute once, assert thrice) — three
fresh runs at ~10 minutes each is wasteful.

The CPU run is needed only for tests 2 and 3. CUDA-vs-CUDA
(test 1) doesn't need it. Skip CPU when only test 1 is selected.

### 3. Re-run the GPU baseline (~30 min)

Run after the sync fix lands:

```
python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --n-episodes 5 \
    --seed 42 \
    --device cuda \
    --out logs/discrete_ppo_v2/run_cuda_post_sync_fix.jsonl
```

Capture wall time per episode. Expectation: 10–25 s / episode
(versus 142.7 s pre-fix). If it's still > 30 s, the sync wasn't
the bottleneck and you need to profile (`torch.profiler`); see
"Stop conditions" below.

### 4. Update findings.md (~15 min)

Append a "Session 01b — parity reformulation + per-tick sync fix"
section to `plans/rewrite/phase-3-cohort/findings.md`. Required
content:

- Bar 1 (CUDA↔CUDA): PASS / FAIL.
- Bar 2 (CPU↔CUDA band): PASS / FAIL with the worst-case action-
  histogram drift and total_reward rel diff.
- Bar 3 (speed): wall time per episode; speed-up vs CPU.
- One paragraph on whatever turned up during the wiring that wasn't
  predicted here.

If any bar fails, hard-stop per "Stop conditions".

## Stop conditions

- Bar 1 (CUDA self-parity) fails → **stop and triage**. This is the
  load-bearing bar. The only way it can fail is a true device-
  handshake bug — the same code on the same GPU with the same seed
  must produce the same answer twice. If it doesn't, look for:
  non-deterministic kernels (set
  `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var), un-seeded RNG calls
  outside the rollout (e.g. data-loader workers, numpy-side
  shuffling), or a `.cuda()` call that's allocating from a
  different stream.
- Bar 2 (CPU↔CUDA band) fails by a wide margin (action histogram
  > 20 %, or total_reward > 50 %) → **stop**. Either device is
  computing genuinely different things. Likely a dtype mismatch
  or a tensor that's silently CPU on one path and GPU on the
  other. Bar 1 should also fail in this case; if Bar 1 passes
  but Bar 2 fails wide, it's a CPU-only bug introduced by the
  sync refactor.
- Bar 3 (speed) fails (CUDA still > 30 s/episode) → **document
  in findings.md, do not block Session 02**. The cohort-run
  speed-up over CPU is real even at 50 s/episode (2× over CPU,
  enough for the cohort to fit in the timetable). Stretch goals
  on speed belong in a follow-on, not here.
- Bar 2 fails narrowly (action histogram between 5 % and 20 %, or
  total_reward between 15 % and 50 %) → loosen the band, document
  the new band as the production contract, proceed. The band's
  job is "catch catastrophic bugs", not "guarantee tight
  match" — that ship sailed when we accepted CPU/CUDA RNG
  divergence.

## Hard constraints

- **No env edits.** Same as all rewrite phases.
- **No re-import of v1 trainer / policy / worker.** Read for
  pattern, re-implement in `training_v2/`.
- **No changes to Phase 1 / Phase 2 numerical defaults.** Same
  hyperparameters, same hidden size, same scorer. Only what's
  in scope here changes.
- **No new tests outside `tests/test_v2_gpu_parity.py` and the
  files this session necessarily touches** (`Transition`
  contract change cascades through three test files; that's
  fine).
- **No loosening Bar 1.** CUDA self-parity within 1e-7 is
  non-negotiable — it's the only honest reproducibility bar
  available. If you can't hit 1e-7, find the source of non-
  determinism, don't widen the tolerance.

## Out of scope

- Multi-day training (Session 02).
- GA cohort scaffolding (Session 03).
- Frontend events (Session 04).
- AMP / autocast.
- Multi-GPU.
- CUDA streams / async kernel launches beyond `non_blocking=True`.
- Moving action sampling to CPU. Tempting as a parity fix, but
  (a) doesn't help — matmul drift still tips probabilities, and
  (b) reintroduces CUDA→CPU sync per tick, which is what we're
  removing.

## Useful pointers

- Session 01 wiring still in place: `--device` flag,
  `_move_to_device` helper, pre-allocated obs/mask buffers. Do
  not undo.
- Existing CPU regression bar: 12 v2 tests in
  `tests/test_discrete_ppo_trainer.py` and
  `tests/test_discrete_ppo_rollout.py`. Re-run after the
  Transition refactor; they must still pass on CPU.
- v1 reference for hidden-state on-device storage:
  `agents/ppo_trainer.py:1393-1450` captures hidden state as
  device tensors during rollout, packs end-of-rollout. Same
  pattern.

## Estimate

2 hours.

- 45 min: Transition refactor + sync removal.
- 30 min: rewrite parity test (three tests, shared fixture).
- 30 min: re-run GPU baseline + measure speed.
- 15 min: findings.md write-up.

If past 3 hours, stop and check scope. Most likely overrun: the
Transition refactor cascading into more test surgery than
expected. If that happens, audit which tests genuinely care
about the storage type vs which were just constructing fixtures
the easy way; the latter just need a one-line helper update.
