# Session prompt — Phase 3, Session 01: GPU pathway + parity

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Wire the v2 trainer to a CUDA device cleanly. Phase 2 hardcodes
`device="cpu"`; Phase 3's cohort run is unworkable without GPU
(estimated 37 hours on CPU). v1's GPU pathway works and is the
canonical reference for pinned memory + non-blocking transfers.

End-of-session bar:

1. `python -m training_v2.discrete_ppo.train --day 2026-04-23 --device cuda --n-episodes 5 --seed 42` runs end-to-end on a CUDA device.
2. The CUDA run produces **numerically identical** total_reward and value_loss_mean as the CPU run (same seed, same day) within 1e-5 absolute tolerance per episode.
3. The CUDA run is < 57 s / episode (50 % of Phase 2's 113 s/episode CPU baseline).

## What you need to read first

1. `plans/rewrite/phase-3-cohort/purpose.md` — success bar table,
   GPU contract, hard constraints.
2. `plans/rewrite/phase-2-trainer/findings.md` — Phase 2 baseline
   numbers (113 s/episode CPU, 744 mini-batches, approx_kl 0.017–
   0.036). Your CUDA run replicates these.
3. `agents/ppo_trainer.py:2131-2174` — v1's pinned-memory + non-
   blocking transfer pattern. **Read, don't import.** This is the
   canonical reference for the transfer shape.
4. `agents/ppo_trainer.py:1384-1390` — v1's pre-allocated GPU
   `obs_buffer` reused across rollout steps. Same pattern, same
   reason: avoid per-step CUDA malloc.
5. `training_v2/discrete_ppo/{trainer.py,rollout.py,train.py}` —
   the files you're modifying.
6. `agents_v2/discrete_policy.py:280-310` — Phase 1's policy
   class. The hidden-state pack/slice and forward pass already
   call `.to(obs.device)` for the LSTM hidden state, so device-
   agnostic; no changes needed if your transfers are correct.
7. `CLAUDE.md` §"Recurrent PPO: hidden-state protocol on update"
   — the rollout-time hidden-state capture rule. The `hidden_state_in`
   stored on every Transition is currently a numpy tuple; on GPU
   it stays as numpy in the Transition (CPU-side storage), then is
   moved to device at update time via `pack_hidden_states`. Don't
   change the storage layout.

## What to do

### 1. `train.py` — `--device` flag (~15 min)

Add an argparse flag:

```
p.add_argument(
    "--device", default="cpu", choices=["cpu", "cuda"],
    help="Torch device. Default cpu (Phase 2 baseline). "
         "Use cuda for the GPU pathway. "
         "Specific GPUs (cuda:1, cuda:2) supported via raw string.",
)
```

Allow specific device strings (`cuda:0`, `cuda:1`) past the choices
list — strip the `:N` suffix for validation, accept the full
string verbatim into `torch.device`. Forward to
`DiscretePPOTrainer(device=...)` already at the trainer ctor.

If `cuda` requested but `torch.cuda.is_available() == False`,
fail loud with a clear error message — don't silently fall back to
CPU (that's the worst kind of bug to debug from JSONL output).

### 2. `trainer.py` — pinned-memory + non-blocking transfers in `_ppo_update` (~45 min)

Phase 2's `_ppo_update` does direct `.to(device)` for every tensor.
The v1 pattern is:

```python
# CPU side: build the tensor with pin_memory=True.
obs_t = torch.from_numpy(obs_np).pin_memory()
# Transfer side: non_blocking=True overlaps with downstream compute.
obs_t = obs_t.to(device, non_blocking=True)
```

Apply this to: `obs`, `masks`, `action_idx`, `stake_unit`,
`uses_stake`, `chosen_adv`, `joint_old_lp`, `returns_t`. The
hidden-state pack already runs through `policy.pack_hidden_states`
which returns torch tensors; pin those on the CPU side first
(loop the tuple, pin each element), then `.to(device,
non_blocking=True)`.

`pin_memory` is a CPU-only no-op when the device is CPU, so the
same code path works for both backends. **Verify this** —
`tensor.pin_memory()` on a CPU-only build raises; the code path
must short-circuit when `device.type == "cpu"`. Implement a
trivial helper:

```python
def _move_to_device(tensor, device, non_blocking=True):
    if device.type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=non_blocking)
    return tensor.to(device)
```

### 3. `rollout.py` — pre-allocated GPU obs buffer (~30 min)

v1 `agents/ppo_trainer.py:1384-1390` allocates a single `obs_buffer`
of shape `(1, obs_dim)` on the device once per episode and reuses
it. v2's collector currently does:

```python
obs_t = torch.from_numpy(obs).to(self.device, dtype=torch.float32).unsqueeze(0)
```

per-step, which mallocs on every tick. Replace with a pre-
allocated buffer + in-place `copy_`:

```python
# Once per episode, before the while-not-done loop:
obs_buffer = torch.empty(
    (1, shim.obs_dim), dtype=torch.float32, device=self.device,
)
mask_buffer = torch.empty(
    (1, self.action_space.n), dtype=torch.bool, device=self.device,
)

# Inside the loop:
obs_buffer.copy_(torch.from_numpy(obs).unsqueeze(0))
mask_buffer.copy_(torch.from_numpy(mask_np).unsqueeze(0))
out = policy(obs_buffer, hidden_state=hidden_state, mask=mask_buffer)
```

This is a CPU→GPU per-step transfer (same as before) but without
the malloc. The GPU saturation win is small per-step but
compounds across 11872 ticks/episode.

### 4. Parity test (~30 min)

`tests/test_v2_gpu_parity.py`:

```python
@pytest.mark.gpu
def test_cpu_cuda_parity_5_episodes():
    # Skip if no CUDA available — but if available, MUST match.
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device")

    from training_v2.discrete_ppo.train import main
    cpu_out = tmp_path / "cpu.jsonl"
    cuda_out = tmp_path / "cuda.jsonl"

    main(day_str="2026-04-23", data_dir=Path(...), n_episodes=5,
         seed=42, out_path=cpu_out, ...)  # device defaults to cpu

    main(day_str="2026-04-23", data_dir=Path(...), n_episodes=5,
         seed=42, out_path=cuda_out, ..., device="cuda")

    cpu_rows = [json.loads(l) for l in cpu_out.read_text().splitlines()]
    cuda_rows = [json.loads(l) for l in cuda_out.read_text().splitlines()]
    assert len(cpu_rows) == len(cuda_rows) == 5
    for cpu_row, cuda_row in zip(cpu_rows, cuda_rows):
        assert abs(cpu_row["total_reward"] - cuda_row["total_reward"]) < 1e-5
        assert abs(cpu_row["value_loss_mean"] - cuda_row["value_loss_mean"]) < 1e-5
```

Note: `main()` doesn't yet take a `device` parameter; thread it
through from the CLI flag. Pass `device=device` to the trainer.

The 1e-5 tolerance is tight on purpose — anything looser admits
silent device-handshake bugs (see Phase 1 findings §"Hidden-state
protocol"). PyTorch's CUDA RNG seeded from `torch.manual_seed`
matches CPU RNG to FP determinism within `1e-6` for the matmul
shapes we use, so 1e-5 cumulative across 744 mini-batches × 5
episodes is the right margin.

If the parity test fails, **stop and triage**. Likely causes (in
order):
- Hidden state not on the right device at update time.
- `pack_hidden_states` returning CPU tensors instead of device
  tensors.
- The pre-allocated obs buffer's dtype mismatch (`float32` vs
  `float64`).
- Non-deterministic CUDA kernels enabled (set
  `torch.backends.cudnn.deterministic = True`).

### 5. Run the GPU baseline + write a session note (~30 min)

Run on the same day as Phase 2:

```
python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --n-episodes 5 \
    --seed 42 \
    --device cuda \
    --out logs/discrete_ppo_v2/run_cuda.jsonl
```

Capture wall time per episode. Compare to Phase 2's 113 s/episode
CPU baseline. The bar is < 57 s/episode (2× speedup); 5–10× is
expected on a real GPU.

Append to `plans/rewrite/phase-3-cohort/findings.md` (create the
file with a Session-01 section, or stub an empty file with a
"Session 01 — GPU pathway" header):

- Bar 1 (parity): PASS / FAIL.
- Bar 2 (speed): wall time per episode CPU vs CUDA, ratio.
- Anything surprising during the wiring.

## Stop conditions

- Parity bar fails → **stop**. Triage device handshake before
  proceeding to Session 02. Multi-day GPU work amplifies any
  parity bug into 7× the noise.
- Speed bar fails (CUDA wall < 50 % of CPU but > 0 %) → **document
  in findings.md, proceed to Session 02**. The phase ships with
  the speed bar at 50 % being a stretch goal; the parity bar is
  the load-bearing one.
- CUDA available but `torch.backends.cudnn.deterministic = True`
  causes wall time to balloon → **leave deterministic mode on for
  the parity test, off for the cohort run**. Phase 3 Session 04's
  cohort doesn't need bit-identical CPU/CUDA output; Session 01's
  test does.

## Hard constraints

- **No env edits.** Same as all rewrite phases.
- **No re-import of v1 trainer / policy / worker.** Read v1's
  pinned-memory pattern; re-implement in `training_v2/`.
- **No changes to Phase 1 / Phase 2 numerical defaults.** Same
  hyperparameters, same hidden size, same scorer. The only thing
  this session changes is where the tensors live.
- **No new tests outside `tests/test_v2_gpu_parity.py`.** This
  session's deliverable is the GPU pathway, not a test sweep.

## Out of scope

- Multi-day training (Session 02).
- GA cohort scaffolding (Session 03).
- Frontend events (Session 04).
- AMP / autocast (follow-on; Phase 3 uses fp32 throughout to keep
  the parity bar honest).
- Multi-GPU (follow-on; Phase 3's cohort runs on one GPU).
- CUDA streams / async kernel launches beyond `non_blocking=True`.

## Useful pointers

- v1 GPU pinning: `agents/ppo_trainer.py:2131-2174` (read for
  pattern, do not import).
- v1 obs buffer reuse: `agents/ppo_trainer.py:1384-1390`.
- v1 device auto-detect + log: `training/run_training.py:183-188`.
- Phase 2 baseline: `plans/rewrite/phase-2-trainer/findings.md`
  §"Reproducibility".
- Phase 1 hidden-state contract: `plans/rewrite/phase-1-policy-
  and-env-wiring/findings.md` §"Hidden-state protocol".

## Estimate

2.5 hours.

- 15 min: `--device` flag.
- 45 min: pinned-memory transfers in `_ppo_update`.
- 30 min: pre-allocated obs buffer in rollout.
- 30 min: parity test.
- 30 min: GPU baseline run + session note in findings.md.

If past 4 hours, stop and check scope. Most likely overrun: the
parity bar — a 1e-5 tolerance is tight and any device-handshake
bug surfaces here. If parity is hard to hit, that's the finding;
don't loosen the tolerance.
