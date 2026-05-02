# Session prompt — throughput-fix Session 02: batched cohort forward

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Session 01 (commit `38dd3a5`) cut the three deferrable per-tick
CUDA→CPU sync barriers in `RolloutCollector._collect`. Self-parity
held bit-identical, but the speedup was **+0.0% combined / -5.3%
steady-state** (10-episode mean 145.5 s/ep vs the AMBER v2 baseline
of 145.5 s/ep; run-B steady-state ~138 s/ep). Verdict: PARTIAL. See
`plans/rewrite/phase-3-followups/throughput-fix/findings.md` §"Session
01" for the trace.

The remaining cost is **architectural**: at LSTM(h=128, batch=1) the
forward pass is dominated by CUDA kernel-launch overhead, not actual
matmul work. Cutting syncs doesn't help if the GPU spends most of
its time launching kernels for tiny tensors. The only path to the
plan-level GREEN bar (≤ 90 s/ep single-agent or ≤ 90 min for a
12-agent × 7-day cohort) is to amortise launch overhead across N
agents per forward.

This session ships **batched cohort forward**: run N agents through
**one** policy forward per tick instead of N separate forwards.
Each agent keeps its own independent env (the env is not thread-safe
and per-tick env steps stay sequential in Python). The forward pass
batches agents as `(N_agents, obs_dim)`. The PPO update path stays
**per-agent** (each agent has its own optimiser + trajectory + KL
early-stop budget). Only the rollout loop is batched.

End-of-session bar:

1. **Per-agent self-parity holds.** Running agent A inside a batch
   of N produces a transition list bit-identical to running agent A
   alone (same seed, same env, same arch). This is the load-bearing
   correctness guard for the whole batched-forward pattern. Same
   shape as Session 01's CUDA↔CUDA self-parity bar, just one level
   up: instead of "CUDA twice = same answer", it's "CUDA-batched =
   CUDA-solo".
2. **CPU/CUDA action-histogram band stays ≤ ±5 %** on the same
   12-day eval slice as Phase 3 Session 01b. The CPU code path is
   structurally unchanged; this just confirms the batched CUDA path
   stays in the band.
3. **Single-agent CUDA wall ≤ 90 s/episode** at N=1
   (degenerate-batch case must not regress vs Session 01's
   ~138 s/ep). The plan-level GREEN bar in this slot.
4. **12-agent cohort wall ≤ 90 min** on the 12 × 7-day protocol
   from `plans/rewrite/phase-3-cohort/` Session 04. AMBER v2
   reference is ~3.1 h / ~186 min. This is the GREEN-with-stretch
   bar; honest band is 90–120 min depending on what fraction of
   per-tick wall is forward vs env.
5. **All pre-existing v2 trainer / rollout / collector tests pass**
   on CPU (byte-identical) and CUDA. Inherited from Sessions 01/01b.
6. **All cohort-runner / worker tests pass** at N=1 and N=12. The
   sequential N=1 path is what the existing tests exercise; the
   N=12 batched path is new.
7. **Verdict logged** as one of:
   - **GREEN**: bars (1)–(3) hit AND self-parity holds AND cohort
     wall ≤ 90 min. Plan moves to Session 03 (verdict writeup).
   - **GREEN-with-stretch**: (1)–(3) hit, but cohort wall is in
     90–120 min. Document; Phase-4 scale-up is unblocked but a
     follow-on session may revisit if 66-agent extrapolation is
     too slow.
   - **PARTIAL**: per-agent self-parity holds, single-agent ≤ 90 s
     but cohort wall stays > 120 min. Document; either ship
     anyway (Phase-4 ahead-of-schedule) or open a follow-on for
     env-side work.
   - **FAIL**: per-agent self-parity breaks. Do not ship the
     change. Stop and triage. Cross-agent leakage in the batched
     forward is the most likely cause.

## What you need to read first

1. `plans/rewrite/phase-3-followups/throughput-fix/purpose.md` —
   this plan's goal, success bar, hard constraints, the locked
   correctness foundation (CUDA↔CUDA self-parity), and the
   §"Session 02" sketch.
2. `plans/rewrite/phase-3-followups/throughput-fix/findings.md`
   §"Session 01" — what got cut, what the residual cost looks like,
   and the gap-to-close (Session 01 steady-state ~138 s/ep → GREEN
   bar 90 s/ep is a 35 % gap that needs ~3× speedup at single-agent
   wall, OR ~2× at cohort wall).
3. `training_v2/discrete_ppo/rollout.py::RolloutCollector._collect`
   — Session 01's deferred-sync pattern. Read it end-to-end (~280
   lines). The batched collector copies its high-level shape
   (sidecar buffers + end-of-episode batched cpu transfer + late
   Transition construction) up one level: per-agent sidecar buffers
   instead of per-tick.
4. `agents_v2/discrete_policy.py::DiscreteLSTMPolicy` — read
   `forward` (~line 276), `init_hidden`, `pack_hidden_states`,
   `slice_hidden_states`. The forward already accepts arbitrary
   batch sizes — at `(N, obs_dim)` it broadcasts cleanly across N.
   The pack/slice helpers exist for the per-mini-batch PPO update
   path; you'll reuse them for per-agent slicing inside the batched
   rollout.
5. `training_v2/cohort/runner.py` — current sequential cohort
   driver. Lines 180–227 hold the per-agent loop you'll either
   replace (one batched call) or keep (around a new
   `train_cohort_batched` entry point).
6. `training_v2/cohort/worker.py::train_one_agent` — current
   single-agent worker. The batched version's per-agent update
   logic is mostly the same; the rollout collection moves to a
   shared collector.
7. `training_v2/discrete_ppo/trainer.py::DiscretePPOTrainer` —
   especially `_collect_rollout` callsite (~line 270) and
   `_ppo_update`. The PPO update is per-agent and stays per-agent.
   Only the rollout-collection input changes.
8. `tests/test_v2_gpu_parity.py` — Phase 3 Session 01b's parity
   test. The new batched parity test borrows its structure
   (5 episodes, fixed seed, JSONL diff).
9. `tests/test_v2_rollout_sync.py` — Session 01's regression
   guards. Read all three; the new batched-rollout tests follow
   the same pattern (gpu+slow self-parity, structural-call
   guards, CPU-twice byte-identity).
10. CLAUDE.md §"Recurrent PPO: hidden-state protocol on update" —
    the hidden-state contract is per-agent; batching doesn't change
    it. Each agent's `hidden_state_in` for tick t is captured BEFORE
    that agent's slice of the batched forward.
11. `plans/rewrite/phase-3-cohort/findings.md` §"Session 01b" — the
    GPU pathway + per-tick sync inventory you inherited.
12. CLAUDE.md §"PPO update stability — advantage normalisation" and
    §"Per-mini-batch KL check" — both are per-agent and stay
    per-agent. The batched rollout doesn't change the surrogate
    loss path.

## The hard correctness problem

**Multiple agents in a cohort do NOT share weights.** Each agent has
its own randomly-initialised parameters that diverge across training.
A naive `policy.forward(stacked_obs, ...)` call uses ONE policy's
weights and would silently mis-attribute outputs across agents.

Three viable shapes for "batched forward across N different policies":

**(a) `torch.vmap` over per-agent parameters.** Use
`torch.func.functional_call` with `in_dims=(None, 0, ...)` to vectorise
the policy forward over a leading "agent" dim of stacked parameters.
Cleanest semantically — the per-agent forward is mathematically
identical to running each agent alone. Real refactor; requires
stacking each agent's `state_dict()` into a per-tensor `(N, ...)`
batched buffer before the rollout and unstacking at end-of-episode.

**(b) Cluster agents by architecture, batch within clusters.**
Different `hidden_size` values ({64, 128, 256}) can't share a forward
even with vmap. Group agents by exact (hidden_size, num_layers,
arch) tuple; each cluster gets one batched forward. Within a cluster,
you still need (a)'s vmap to handle different weights — clustering
just bounds the cluster size to ≤ N_agents.

**(c) Per-agent forward in a Python loop, one big optimiser-step.**
No batched forward at all — N forwards per tick. The "batching" is
just removing the env-step pause between them. Easiest to implement,
gives almost no GPU speedup (Session 01 already did most of this
implicitly). NOT what this session is for; included for completeness
because (a) might prove infeasible in implementation and you might
have to fall back.

**Pick (b) — cluster, then vmap.** Reasons:

- Real cohorts in the v2 GA mix architectures (cohort-A had hidden
  size variation across agents). A pure-vmap solution that doesn't
  cluster will mis-batch at the first generation that breeds
  hidden-size diversity.
- Within a cluster, vmap is the only correct path. PyTorch 2.0+
  `torch.func.vmap` + `functional_call` is mature.
- The clustering layer is small (a `defaultdict[arch_key, list[agent_idx]]`
  + a per-cluster collector instance).
- Falls back gracefully: a "cluster" of 1 agent is just a single-
  agent forward, so the degenerate case (N_clusters = N_agents = 1)
  reduces to Session 01's path bit-identically.

If during implementation vmap proves to have a fatal pitfall (e.g.
LSTMCell doesn't vmap cleanly under autograd at the version pinned
in this repo), fall back to (c) and document the speedup gap.

## Inherited optimisations — do not regress

V1 (`agents/ppo_trainer.py`) is **single-agent only** — there is no
multi-agent GPU-batching pattern to crib from. The batched-forward
shape in this session is novel work. What v1 *did* figure out for
single-agent CUDA throughput is already lifted into v2 and must
survive the batched refactor:

1. **Pre-allocated obs / mask buffers, reused across ticks.** V1
   pattern at [`agents/ppo_trainer.py:1384-1390`](../../../../agents/ppo_trainer.py).
   Session 01 already mirrors it at batch=1 in
   [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
   lines ~148–153 (`obs_buffer`, `mask_buffer` allocated once, then
   `.copy_(...)` per tick). The batched collector should do the
   **same at batch=N_active**: allocate `(N_max, obs_dim)` and
   `(N_max, action_n)` buffers once, slice the active subset each
   tick, `.copy_(...)` the per-agent obs into the corresponding
   row. Avoids per-tick `torch.empty` × N_active allocations.
2. **Pinned-memory + non-blocking H2D transfers.** V1 pattern at
   [`agents/ppo_trainer.py:2131-2174`](../../../../agents/ppo_trainer.py).
   The v2 trainer already wraps this in `_move_to_device`
   ([`training_v2/discrete_ppo/trainer.py`](../../../../training_v2/discrete_ppo/trainer.py)
   ~line 72) — use it for any obs / mask H2D the batched collector
   does, not raw `.to(device)`.
3. **`hidden_state_in.detach().clone()` BEFORE the forward pass.**
   Phase 3 Session 01b's load-bearing fix. CLAUDE.md §"Recurrent
   PPO: hidden-state protocol on update" is the contract. The
   batched collector captures per agent: slice the packed hidden
   state per-agent, then `detach().clone()` on the slice. Don't
   `detach().clone()` the packed-batch tensor and slice afterward —
   that mutates the rolling state.
4. **`cudnn.deterministic = True`, `cudnn.benchmark = False`.**
   Already set in `train.py` at device-resolve time. Plan-level
   constraint inherited from purpose.md.

What v1 did **not** figure out and this session is NOT trying to
inherit:

- **No per-agent RNG isolation in v1.** V1 used the global
  `torch.manual_seed` for everything; correct for single-agent
  but a leak hazard at batch=N. This session's per-agent
  `torch.Generator` requirement (§2 above) is novel — there is
  no v1 reference implementation to copy.
- **No vmap / functional_call usage in v1.** Carrying anything
  from v1 here is impossible; the forward path is being rebuilt
  one level up, not extended.
- **No mixed precision / AMP in v1.** Out of scope this session
  per purpose.md.

## What to do

### 1. Pre-flight (~45 min)

- Read all 12 files in §"What you need to read first". Before
  writing any code, make sure you can sketch on paper:
  - Where the rollout-loop's per-tick forward call lives today.
  - Which buffers in Session 01 hold per-agent vs shared state.
  - How the trainer's `_ppo_update` consumes a `list[Transition]`.
  - The shape of `policy.state_dict()` for `DiscreteLSTMPolicy` —
    which tensors will get a leading `N_cluster` dim under vmap.
- Run the Session 01 baseline tests as known-good:

  ```
  pytest tests/test_v2_rollout_sync.py -v --runslow
  pytest tests/test_discrete_ppo_rollout.py
        tests/test_discrete_ppo_trainer.py
        tests/test_discrete_ppo_transition.py -v
  pytest tests/test_v2_gpu_parity.py -v --runslow  # ~25 min, CUDA only
  ```

  All should pass at the Session 01 commit. If any fail before you
  start, stop — there's a different bug to triage.
- Verify `torch.func.vmap` and `torch.func.functional_call` are
  importable in the repo's pinned torch version:

  ```python
  import torch
  from torch.func import vmap, functional_call
  print(torch.__version__)  # should be 2.x
  ```

  If the version doesn't have `torch.func`, stop and reconcile with
  the operator before proceeding.
- Confirm the AMBER v2 cohort-wall baseline by reading
  `plans/rewrite/phase-3-cohort/findings.md` Session 04's table
  (or `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`
  timestamps if the table is incomplete). Document the exact
  baseline cohort wall in the session note.

### 2. Decide and lock the architecture-clustering shape (~30 min)

Before any code lands, write the clustering policy down in the
session note. The minimum decision set:

- **Cluster key.** What identifies a "compatible-for-vmap" agent
  group? Recommend: `(policy_class, hidden_size, num_layers,
  obs_dim, action_dim)`. The `policy_class` matters for future
  GRU / Transformer support; today only LSTM exists, so this
  collapses to `hidden_size` in practice.
- **Within-cluster ordering.** Stable agent-index order. Use the
  agent's index in the parent cohort's gene list — this lets the
  RNG-seeding logic (next decision) remain deterministic.
- **Cross-cluster scheduling.** Sequential. Cluster 1 trains for
  one generation, then cluster 2, etc. Parallelising across
  clusters competes for GPU and re-introduces the worker-pool
  fragility purpose.md §"Out of scope" excludes.
- **Per-agent RNG.** Each agent gets its own `torch.Generator`
  seeded by `(cohort_seed, generation, agent_idx)`. **Do not**
  rely on `torch.manual_seed`. The batched `Categorical.sample()`
  / `Beta.sample()` calls must accept a per-agent generator OR be
  re-implemented in terms of `torch.rand` with per-agent generator
  (since `Distribution.sample` doesn't accept a `generator`
  kwarg today). The simplest concrete shape:

  ```python
  # Per-agent generator, seeded once at agent build time
  generators = [
      torch.Generator(device=self.device).manual_seed(
          (cohort_seed * 1_000_003 + generation * 10_000 + i) & 0x7FFFFFFF
      )
      for i in range(n_agents)
  ]

  # Replace `dist.sample()` with manual reparameterised sampling:
  # Categorical: torch.multinomial(probs, num_samples=1, generator=g_i)
  # Beta: stake = a / (a + b * exp(-x)) where x is from N(0, 1)
  #       OR keep dist.sample() and seed the global generator before
  #       each agent's sample slice — works but fragile.
  ```

  Pick the multinomial / explicit-sample path. It's a small amount
  of code and isolates per-agent RNG from any global state.

### 3. Implement `BatchedRolloutCollector` (~3 h)

New file: `training_v2/discrete_ppo/batched_rollout.py`. Keep
Session 01's `RolloutCollector` untouched — the new collector is
its sibling.

```python
class BatchedRolloutCollector:
    """Drive N (shim, policy) pairs through one episode in lockstep,
    with one batched policy forward per tick.

    Constraints:
    - All N policies must share architecture (cluster key).
    - Each agent has its own env, its own seed, its own generator.
    - PPO update consumes per-agent transition lists from this
      collector exactly the same as the single-agent collector.
    """
```

Key methods:

- `__init__(shims, policies, device, generators)`. Validate:
  same architecture across all `policies`, same `obs_dim`,
  same `action_space.n`. `len(shims) == len(policies) == len(generators)`.
- `collect_episode_batch() -> list[list[Transition]]`. Returns
  per-agent transition lists. Per-agent ordering preserved.
- Internal: stack params via `torch.stack` over each
  `state_dict()` key, run the rollout loop with the vmapped
  `functional_call`-based forward, slice outputs per-agent,
  step each env in a Python loop.

The rollout loop's shape:

```python
# Pre-loop: stack per-agent params into (N, ...) buffers.
stacked_params = stack_params([p.state_dict() for p in policies])
hidden_states = [p.init_hidden(batch=1) for p in policies]
# Concat along agent dim using the policy's pack_hidden_states helper
# (LSTM packs along dim 1; transformer packs along dim 0 — see
# CLAUDE.md "Recurrent PPO: hidden-state protocol on update"
# §"Architecture-specific batching axis").
packed_hidden = template_policy.pack_hidden_states(hidden_states)

# Per-agent sidecar buffers (Session 01 pattern, lifted to per-agent):
pending_log_prob_action = [[] for _ in range(N)]
pending_log_prob_stake = [[] for _ in range(N)]
pending_value_per_runner = [[] for _ in range(N)]
# Per-tick CPU-side bookkeeping, also per-agent:
per_tick_obs = [[] for _ in range(N)]
per_tick_hidden_in = [[] for _ in range(N)]
# ... etc, mirroring Session 01's lists

active = list(range(N))  # agent indices still mid-episode

while active:
    # Stack obs from active agents.
    obs_batch = torch.stack([
        torch.from_numpy(latest_obs[i]).to(device, dtype=torch.float32)
        for i in active
    ], dim=0)
    mask_batch = torch.stack([...], dim=0)
    active_hidden = template_policy.slice_hidden_states(
        packed_hidden, torch.tensor(active, device=device),
    )
    active_params = slice_params(stacked_params, active)

    # Capture hidden_in BEFORE forward (per agent).
    hidden_in_per_agent = [
        # detach + clone the slice for agent i
        ...
        for i in active
    ]

    # Vmapped forward: (N_active, obs_dim) → DiscretePolicyOutput
    # whose tensors are (N_active, ...).
    out = vmapped_forward(active_params, obs_batch, active_hidden, mask_batch)

    # Per-agent sample using per-agent generator.
    for j, i in enumerate(active):
        action_i = sample_categorical(out.logits[j], generators[i])  # 0-d
        # ...
        # Stash deferred device tensors on per-agent buffers.
        pending_log_prob_action[i].append(out.action_dist_log_prob[j].detach())
        # ...

        # Structural sync: env consumes int + float per tick.
        action_idx = int(action_i.item())
        stake_unit = float(stake_unit_t.item())
        next_obs_i, ... = shims[i].step(action_idx, stake=stake_pounds)
        if done_i:
            active.remove(i)
            terminated[i] = (info_i, ...)

    # Update packed_hidden with out.new_hidden_state for active agents.
    packed_hidden = scatter_update_hidden(packed_hidden, active, out.new_hidden_state)
```

End-of-episode: per-agent `torch.stack` + `.cpu()` per buffer
(same as Session 01). Build per-agent transition lists. Return
`list[list[Transition]]`.

**Critical correctness invariants:**

1. **Per-agent hidden state never bleeds across agents.** The
   slicing via `slice_hidden_states` is the load-bearing primitive.
   Test: at tick 0, slice agent 0's hidden state out of a packed
   hidden where agent 1 was initialised non-zero — agent 0's slice
   must still be zero.
2. **Per-agent RNG never bleeds.** Two batched runs at the same
   per-agent seeds must produce identical actions per agent.
   Switching agent A's seed must NOT change agent B's actions.
3. **Active-set bookkeeping.** When an agent terminates mid-batch,
   subsequent batched forwards must NOT include it. Shrinking
   `active` rebuilds the param/hidden/obs slices. The unused agent's
   final hidden state is captured exactly at termination.

### 4. Wire `BatchedRolloutCollector` into the cohort layer (~1.5 h)

Two reasonable shapes; pick (a):

- **(a) New `train_cohort_batched` entry point.** Lives in
  `training_v2/cohort/runner.py` alongside the existing
  per-agent loop. Default cohort-runner CLI flag `--batched`
  (default off) flips between paths. Sequential per-cluster, with
  one `BatchedRolloutCollector` per cluster per training day.
- **(b) Refactor `train_one_agent` to optionally accept a shared
  collector.** More invasive; risks coupling per-agent worker
  state to cross-agent state.

Pick (a). Reasons: keeps the worker.py per-agent contract
unchanged (PPO update + registry write per agent), confines the
batching change to the runner, makes the rollback path one CLI
flag away.

The new path:

```python
def train_cohort_batched(...):
    clusters = group_by_arch(genes_list)
    for cluster_key, agent_indices in clusters.items():
        per_day_results = train_cluster_batched(
            agent_indices, training_days, eval_day, ...
        )
        for i, result in zip(agent_indices, per_day_results):
            scoreboard_rows[i] = result_to_row(result)
```

### 5. Tests (~1 h)

New file: `tests/test_v2_batched_rollout.py`. Five tests:

```python
@pytest.mark.gpu
@pytest.mark.slow
def test_per_agent_self_parity_batched_vs_solo():
    """Running agent A inside a batch of N produces transitions
    bit-identical to running A alone.

    Build the same arch + same seed twice. Run once via
    BatchedRolloutCollector at N=4 (4 agents, all same arch),
    keep agent 0's transition list. Run again via the Session 01
    RolloutCollector at N=1 with the same agent-0 seed and policy
    weights. Compare action_idx, stake_unit, log_prob_action,
    log_prob_stake, value_per_runner element-wise. Strict equality."""

@pytest.mark.gpu
@pytest.mark.slow
def test_per_agent_rng_independence_in_batch():
    """Switching agent A's seed does not change agent B's actions.

    Two batched runs of N=4. Run 1: seeds [42, 43, 44, 45].
    Run 2: seeds [99, 43, 44, 45]. Assert agents 1, 2, 3 produce
    bit-identical transition lists across the two runs. Agent 0
    diverges (by construction)."""

def test_active_set_shrinks_when_agent_terminates_mid_batch():
    """When agents have different episode lengths, the batched
    forward correctly excludes terminated agents.

    Construct a synthetic 2-agent setup where agent 0 terminates
    at tick T and agent 1 at tick T+5. Verify the forward pass
    on ticks T+1..T+5 has batch size 1 (agent 1 only), and
    agent 0's transition list ends at tick T."""

def test_cluster_key_groups_compatible_archs():
    """Agents with different hidden_size land in different
    clusters. Within a cluster all archs match exactly."""

def test_batched_collector_falls_back_to_n1_session01_path():
    """N=1 batched collector produces transitions byte-identical
    to Session 01's RolloutCollector on CPU at the same seed.

    The degenerate-batch case must reduce to Session 01's path —
    no semantic drift introduced by the batched code path."""
```

The first test is the load-bearing correctness guard, mirroring
Session 01's `test_cuda_self_parity_after_sync_removal`.

Run:

```
pytest tests/test_v2_batched_rollout.py -v --runslow
pytest tests/test_v2_rollout_sync.py -v --runslow  # Session 01 guards still pass
pytest tests/test_v2_gpu_parity.py -v --runslow  # ~25 min — sanity
pytest tests/test_discrete_ppo_rollout.py
       tests/test_discrete_ppo_trainer.py
       tests/test_discrete_ppo_transition.py -v
```

If `test_per_agent_self_parity_batched_vs_solo` fails, **stop**.
The batched forward has cross-agent leakage; do not measure speed
until parity holds.

### 6. Speed measurement (~2 h)

Two probes:

**Probe A — single-agent degenerate batch (~10 min).** Same shape
as Session 01: 5 episodes, seed 42, day 2026-04-23, CUDA, n_agents=1
through the batched path. Bar: ≤ 90 s/ep AND no regression vs
Session 01's ~138 s/ep.

```
python -m training_v2.cohort.runner \
    --n-agents 1 --generations 1 --days 1 \
    --device cuda --seed 42 --batched \
    --output-dir registry/throughput_session02_n1_$(date +%s)
```

**Probe B — 12-agent cohort (~90–180 min).** Same shape as Phase 3
Session 04 (12 agents × 7 days × 1 gen). Bar: ≤ 90 min wall to
GREEN-with-stretch, ≤ 120 min to PARTIAL.

```
python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 7 \
    --device cuda --seed 42 --batched \
    --output-dir registry/throughput_session02_n12_$(date +%s)
```

Capture wall time per episode + total cohort wall. Compare:

| Configuration | s/ep CUDA (single agent) | Cohort wall (12×7) | Self-parity |
|---|---|---|---|
| AMBER v2 (Phase 3 S01b) | 145.5 s | ~3.1 h / ~186 min | PASS |
| Session 01 | ~138 s | (not measured) | PASS |
| Session 02 N=1 (this) | ? | (n/a) | ? |
| Session 02 N=12 (this) | ? | ? | ? |

If Probe B's cohort wall is in the 90–180 min range, fine. If it's
>180 min something is wrong (the batched forward is slower than
sequential — likely a bad cluster shape or a sync that re-appeared
under vmap).

### 7. Optional — `cProfile` snapshot (~30 min)

If Probe A is GREEN but Probe B disappoints, profile a 1-episode
N=12 batched run:

```
python -m cProfile -o /tmp/v2_batched.prof -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 1 --batched ...

python -c "import pstats; p = pstats.Stats('/tmp/v2_batched.prof'); p.sort_stats('cumulative').print_stats(40)"
```

Look for:
- `vmap` / `functional_call` overhead at the per-tick forward.
- `shim.step` time × N — sequential env stepping is structural; if
  it dominates, env vectorisation (separate plan, out of scope) is
  the next lever.
- `torch.stack` / `torch.cat` inside the rollout — those should be
  end-of-episode only.

Capture the top-10 in the session findings note.

### 8. Findings note (~15 min)

Append to `plans/rewrite/phase-3-followups/throughput-fix/findings.md`:

```markdown
## Session 02 — batched cohort forward

**Status:** GREEN | GREEN-with-stretch | PARTIAL | FAIL

### Speed table

| Configuration | s/ep CUDA (N=1) | Cohort wall (12×7) | Δ vs AMBER v2 |
|---|---|---|---|
| AMBER v2 baseline | 145.5 s | ~186 min | — |
| Session 01 | ~138 s | (not measured) | -5.3% |
| Session 02 N=1 | X.X s | (n/a) | -Y% |
| Session 02 N=12 | X.X s | Z min | -W% |

### Self-parity

Batched-vs-solo per-agent self-parity at N=4: PASS (diff = 0.0).
Per-agent RNG independence: PASS (agents 1-3 unchanged when
agent 0's seed changes).

### What changed
- (One-paragraph summary of the new collector + cohort-runner
  wiring.)

### What's next
- (GREEN: Session 03 verdict writeup + Phase-4 unblocked.
  GREEN-with-stretch: same, but document residual gap.
  PARTIAL: open follow-on — env-side vectorisation or
  multi-process workers.
  FAIL: triage parity break.)
```

## Stop conditions

- **Per-agent self-parity test FAILS** → stop, triage. The most
  likely cause is a cross-agent leak in the vmapped forward (e.g.
  `state_dict()` keys not stacked along the right dim, or
  `slice_hidden_states` indexing the wrong axis on LSTM hidden
  state — LSTM packs on dim 1, not dim 0). Re-read
  `plans/rewrite/ppo-kl-fix/lessons_learnt.md` for the analogous
  per-mini-batch hidden-state bug; same shape, one level up.
- **Pre-existing v2 trainer / rollout / transition tests fail** →
  the new collector code path leaked into the schema. Roll back
  to the Session 01 commit and check for shared imports. The
  batched collector is a SIBLING file; if changes spill into
  `rollout.py`, that's a regression.
- **N=1 batched wall > 145 s/ep** → vmap + functional_call has
  per-tick overhead that exceeds the sync savings. Profile, and
  if the overhead is unavoidable at this torch version, fall
  back to design (c) (per-agent forward in a Python loop with no
  vmap) and document the speedup gap.
- **N=12 cohort wall > 180 min** → the batched path is slower
  than sequential. Almost certainly a bad cluster shape (e.g.
  forcing N=12 through one cluster when half the agents have
  hidden_size=64 and half have hidden_size=128). Verify
  clustering output before re-measuring.
- **Past 8 h excluding speed-measurement wall** → stop and check
  scope. The batched-forward refactor is real work (purpose.md
  estimates 6 h) but if it's stretching to 10 h there's a hidden
  vmap/autograd compatibility issue. Consider falling back to
  design (c) and shipping a PARTIAL verdict.
- **vmap doesn't compose with the policy's `LSTM` autograd path**
  → check `torch.func.functional_call` against `nn.LSTM` at the
  pinned torch version. If it errors on backward, the cleanest
  fallback is design (c) (per-agent loop, no vmap) — accept the
  speedup gap and document it.

## Hard constraints

Inherited from
`plans/rewrite/phase-3-followups/throughput-fix/purpose.md`
§"Hard constraints" plus:

1. **No env edits.** `env/betfair_env.py`, `env/bet_manager.py`,
   `env/exchange_matcher.py` are off-limits. If a perf opportunity
   sits inside the env, file a follow-on plan.
2. **No reward-shaping changes.** Untouched: `reward.*` config,
   matured-arb bonus, naked-loss handling, mark-to-market,
   open-cost shaping.
3. **Hidden-state contract is unchanged per agent.** The
   `hidden_state_in` capture pattern from Session 01b
   (`tuple(t.detach().clone() for t in hidden_state)` BEFORE the
   forward pass) stays exactly as it is — just per-agent, sliced
   from the packed-batch tensor. Don't move it, don't batch
   across the capture, don't refactor it. CLAUDE.md §"Recurrent
   PPO: hidden-state protocol on update" is the contract.
4. **The two structural `.item()` calls stay per agent.**
   `int(action.item())` and `float(stake_unit_t.item())` are
   required by the CPU env. Removing them silently is a
   correctness break.
5. **Per-agent self-parity is the load-bearing correctness
   guard.** A 5× cohort-wall speedup that breaks it is not
   shipped.
6. **Per-agent RNG independence.** Each agent's
   `Categorical.sample()` and `Beta.sample()` must draw from a
   generator seeded by its `(cohort_seed, generation, agent_idx)`,
   not from the shared default generator.
7. **Same `--seed 42` for every measurement.** Cross-cohort
   comparison invariant.
8. **`cudnn.deterministic = True` stays on.** Plan-level
   constraint inherited from purpose.md.
9. **No GA gene additions.** This session ships zero new genes.
   Clustering reads existing arch genes; doesn't write new ones.
10. **No re-import of v1 trainer / policy / rollout / worker
    pool.** Phase 2/3 hard constraint inherited verbatim.
11. **Transition shape unchanged.** `Transition` stays a frozen
    dataclass with the same fields. The PPO update consumes
    per-agent transition lists exactly the same shape as
    Session 01.
12. **`--batched` defaults OFF.** The sequential path stays the
    default until at least one cohort run validates the batched
    one. Deleting the sequential path is a Session 03 question.

## Out of scope

- Multi-GPU training (one machine, one GPU).
- AMP / autocast.
- Env vectorisation (would touch `env/` — separate plan).
- `cudnn.benchmark = True`.
- 66-agent scale-up (Phase-4 question, gated on this plan's
  verdict).
- v1 deletion.
- Reward-shape iteration.
- BC pretrain.
- Moving the PPO update to batched (per-agent stays per-agent;
  only rollout collection batches).
- Frontend-event throughput.

## Useful pointers

- Session 01 collector:
  [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py).
- Policy with batchable forward:
  [`agents_v2/discrete_policy.py`](../../../../agents_v2/discrete_policy.py)
  (`forward`, `pack_hidden_states`, `slice_hidden_states`).
- Cohort runner:
  [`training_v2/cohort/runner.py`](../../../../training_v2/cohort/runner.py).
- Single-agent worker:
  [`training_v2/cohort/worker.py`](../../../../training_v2/cohort/worker.py).
- PPO trainer:
  [`training_v2/discrete_ppo/trainer.py`](../../../../training_v2/discrete_ppo/trainer.py)
  (`_collect_rollout`, `_ppo_update`).
- Existing parity test:
  [`tests/test_v2_gpu_parity.py`](../../../../tests/test_v2_gpu_parity.py).
- Session 01 sync-removal tests:
  [`tests/test_v2_rollout_sync.py`](../../../../tests/test_v2_rollout_sync.py).
- Pre-existing v2 trainer/rollout/transition tests:
  [`tests/test_discrete_ppo_rollout.py`](../../../../tests/test_discrete_ppo_rollout.py),
  [`tests/test_discrete_ppo_trainer.py`](../../../../tests/test_discrete_ppo_trainer.py),
  [`tests/test_discrete_ppo_transition.py`](../../../../tests/test_discrete_ppo_transition.py).
- Session 01 findings:
  [`plans/rewrite/phase-3-followups/throughput-fix/findings.md`](../findings.md)
  §"Session 01".
- Phase 3 cohort baseline:
  `plans/rewrite/phase-3-cohort/findings.md` §"Session 01b" + §"Session 04".
- AMBER v2 cohort dir:
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- Hidden-state contract: CLAUDE.md §"Recurrent PPO: hidden-state
  protocol on update".
- PPO KL stability: CLAUDE.md §"Per-mini-batch KL check".
- v1 GPU pinning reference (read, do not import):
  [`agents/ppo_trainer.py`](../../../../agents/ppo_trainer.py)
  lines 2131–2174.

## Estimate

8 h:

- 45 min: pre-flight (read, baseline tests, vmap availability check,
  cohort baseline doc).
- 30 min: clustering-shape + RNG-independence design lock.
- 3 h: BatchedRolloutCollector implementation (the bulk of the
  refactor; vmap + per-agent generators + active-set bookkeeping).
- 1.5 h: cohort-runner wiring (`train_cohort_batched` entry point +
  CLI flag).
- 1 h: tests (5 new in `tests/test_v2_batched_rollout.py`).
- 2 h: speed measurement (10 min N=1 + 90–180 min N=12 cohort + analysis).
- 30 min: optional cProfile + findings note.

If past 8 h excluding the cohort-wall measurement, stop and check
scope. The batched forward is the highest-risk session in this plan;
a fallback to design (c) (per-agent forward, no vmap) is acceptable
if vmap proves intractable, but costs the GPU-saturation goal.

If past 12 h total (refactor + cohort run), the cohort run itself
is hung — kill it, check for a deadlock in the active-set
bookkeeping or a Python-side memory blowup from per-agent buffers.
