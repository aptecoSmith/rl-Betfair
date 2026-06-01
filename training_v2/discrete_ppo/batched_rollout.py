"""Batched rollout collector for the v2 discrete-PPO trainer.

Throughput-fix Session 02 deliverable. Drives ``N`` ``(shim, policy)``
pairs through one episode each, with per-tick forwards run in a tight
sequential loop (no env-step pause between them) and per-agent
sidecar buffers that defer the device→CPU transfer to end-of-episode.
The active set shrinks as agents terminate so terminated agents do
not consume forward / env work for the rest of the episode.

What this is and is not:

- This is the **fallback design (c)** from the Session 02 prompt:
  per-agent forward in a Python loop. The prompt's design (b) (vmap
  over stacked per-agent params via ``torch.func.functional_call``)
  is currently blocked at PyTorch 2.11 because there is no batching
  rule for ``aten::lstm.input`` — vmap raises
  ``RuntimeError: Batching rule not implemented for aten::lstm``.
  This implementation keeps the batched-collector interface so a
  vmap-based forward can swap in once the LSTM batching rule lands;
  the active-set bookkeeping, per-agent generators, sidecar buffers
  and deferred end-of-episode CPU transfer all carry forward.
- All N policies in one collector instance share architecture (same
  cluster key — see :func:`cluster_agents_by_arch`). Cohorts that
  mix architectures must run multiple collectors in sequence.
- Per-agent self-parity is preserved by construction: each agent's
  forward is the same call (same obs, same hidden, same mask) as
  the single-agent :class:`RolloutCollector`. RNG independence
  comes from save / restore of the global RNG state around each
  agent's tick; cross-agent leakage cannot occur because no
  randomness is consumed outside the per-agent restore window.

Session 01's deferred-sync pattern is preserved per agent: the
three deferrable per-tick CUDA→CPU syncs (action log-prob, stake
log-prob, per-runner value) are stashed as 0-d / 1-d device
tensors in per-agent sidecar buffers and materialised in one
batched ``.cpu()`` per agent at end-of-episode. The two structural
``.item()`` calls (``action_idx``, ``stake_unit``) stay — the CPU
env consumes ``int`` and ``float`` every tick.

Hard constraints (Session 02 prompt §"Hard constraints"):

* No env edits.
* No re-import of v1 trainer / policy / rollout / worker pool.
* Hidden-state contract is unchanged per agent.
* The two structural ``.item()`` calls stay per agent.
* Per-agent self-parity is the load-bearing correctness guard.
* Per-agent RNG independence.
* ``Transition`` shape unchanged.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, NamedTuple

import numpy as np
import torch

from agents_v2.discrete_policy import BaseDiscretePolicy
from agents_v2.env_shim import DiscreteActionShim
from env.bet_manager import MIN_BET_STAKE
from training_v2.discrete_ppo.transition import Transition, action_uses_stake


logger = logging.getLogger(__name__)


__all__ = [
    "BatchedForwardOut",
    "BatchedRolloutCollector",
    "batched_forward_core",
    "cluster_agents_by_arch",
    "stack_policies",
]


# ── R1: GPU batch=N forward ──────────────────────────────────────────────


class BatchedForwardOut(NamedTuple):
    """Tensors-only result of :func:`batched_forward_core` (leading dim N).

    Excludes the fc-head fields (``fc_prob``/``fc_prob_logit``) — they are
    ``None`` when the fc head is disabled (the default) and ``vmap`` cannot
    return ``None``. The batched collector does not stamp fc onto bets, so
    dropping them is benign (documented batched-path limitation; if the fc
    head is ever enabled in a batched cohort, wire it here).
    """

    logits: torch.Tensor
    masked_logits: torch.Tensor
    stake_alpha: torch.Tensor
    stake_beta: torch.Tensor
    value_per_runner: torch.Tensor
    new_h: torch.Tensor
    new_c: torch.Tensor
    fill_prob: torch.Tensor
    mature_prob: torch.Tensor
    risk_mean: torch.Tensor
    risk_log_var: torch.Tensor
    direction_back_prob: torch.Tensor
    direction_lay_prob: torch.Tensor
    direction_back_logits: torch.Tensor
    direction_lay_logits: torch.Tensor


def stack_policies(policies: list):
    """Stack a cluster's per-agent params/buffers for the batched forward.

    Call ONCE per active-set change (weights are frozen during rollout —
    no gradient steps — so re-stacking every tick is pure waste; that was
    the dominant overhead in the first R1 measurement). Returns
    ``(base, params, buffers)`` ready for :func:`batched_forward_core`.
    """
    from torch.func import stack_module_state
    params, buffers = stack_module_state(policies)
    return policies[0], params, buffers


def batched_forward_core(
    base,
    params: dict,
    buffers: dict,
    obs: torch.Tensor,
    hidden_h: torch.Tensor,
    hidden_c: torch.Tensor,
    mask: torch.Tensor,
):
    """One stacked GPU forward over a cluster of N same-arch agents.

    training-speedup-v2 R1. Replaces the per-agent batch=1 forward loop
    (launch-bound on CUDA) with a single ``vmap(functional_call(...))``
    over the agents' stacked weights — the idle GPU finally does real
    work. Uses each agent's manual-LSTM path (``_manual_lstm_step``)
    because the fused ``nn.LSTM`` has no vmap batching rule.

    Parameters
    ----------
    base : a reference DiscreteLSTMPolicy (``policies[0]``) — functional_call
        runs its ``forward`` with the stacked params swapped in.
    params, buffers : from :func:`stack_policies` (stacked once per
        active-set change; weights are frozen during rollout).
    obs : ``(N, 1, obs_dim)`` — one obs row per agent (the inner batch dim
        of 1 is kept so each agent's ``forward`` sees a 2-D obs).
    hidden_h, hidden_c : ``(N, num_layers, 1, hidden)`` — per-agent LSTM
        state going INTO this tick.
    mask : ``(N, action_n)`` bool — per-agent legality mask.

    Returns
    -------
    :class:`BatchedForwardOut` (leading dim N). Each per-agent field carries
    the inner batch=1 dim, e.g. ``masked_logits`` is ``(N, 1, action_n)``,
    ``value_per_runner`` ``(N, 1, max_runners)``, ``new_h``/``new_c``
    ``(N, 1, 1, hidden)``. The caller squeezes dim 1.
    """
    from torch.func import functional_call, vmap

    prev_flag = base._manual_lstm_step
    base._manual_lstm_step = True
    try:
        def core_fn(p, b, o, h, c, m):
            t = functional_call(
                base, (p, b), (o,),
                {"hidden_state": (h, c), "mask": m, "_return_core": True},
            )
            # Tensors-only subset (drop the None-able fc fields so vmap
            # can map the output pytree).
            return BatchedForwardOut(
                logits=t.logits, masked_logits=t.masked_logits,
                stake_alpha=t.stake_alpha, stake_beta=t.stake_beta,
                value_per_runner=t.value_per_runner,
                new_h=t.new_h, new_c=t.new_c,
                fill_prob=t.fill_prob, mature_prob=t.mature_prob,
                risk_mean=t.risk_mean, risk_log_var=t.risk_log_var,
                direction_back_prob=t.direction_back_prob,
                direction_lay_prob=t.direction_lay_prob,
                direction_back_logits=t.direction_back_logits,
                direction_lay_logits=t.direction_lay_logits,
            )

        with torch.no_grad():
            out = vmap(core_fn, in_dims=(0, 0, 0, 0, 0, 0))(
                params, buffers, obs, hidden_h, hidden_c, mask,
            )
    finally:
        base._manual_lstm_step = prev_flag
    return out


_ATTRIBUTION_TOLERANCE = 1e-4


# ── Architecture clustering ──────────────────────────────────────────────


def cluster_agents_by_arch(
    policies: list[BaseDiscretePolicy],
) -> dict[tuple, list[int]]:
    """Group agent indices by ``(policy_class, hidden_size, obs_dim, action_n)``.

    Returns a mapping from the cluster-key tuple to the list of agent
    indices in that cluster. Within-cluster ordering is the agent's
    index in the parent cohort (stable, deterministic). Cross-cluster
    scheduling is the caller's choice — see ``train_cohort_batched``
    in ``training_v2/cohort/runner.py`` for sequential-per-cluster.

    Cluster key shape:

    * ``policy_class``: ``type(policy).__name__``. Today only
      ``DiscreteLSTMPolicy`` exists; future GRU / Transformer
      siblings would land in their own clusters.
    * ``hidden_size``: each architecture variant in the GA gene
      schema (e.g. {64, 128, 256}) is its own cluster.
    * ``obs_dim``: cohorts using the same scorer share this; here
      for safety against future obs-schema variants.
    * ``action_n``: ditto for action-space variants.
    """
    clusters: dict[tuple, list[int]] = defaultdict(list)
    for i, p in enumerate(policies):
        key = (
            type(p).__name__,
            int(p.hidden_size),
            int(p.obs_dim),
            int(p.action_space.n),
        )
        clusters[key].append(i)
    return dict(clusters)


# ── Collector ───────────────────────────────────────────────────────────


class BatchedRolloutCollector:
    """Drive N (shim, policy) pairs through one episode each in lockstep.

    Per tick:

    1. For each active agent ``i``: restore agent ``i``'s saved global
       RNG state, run the policy forward, sample action + stake (which
       consumes RNG), save the new RNG state back into agent ``i``'s
       slot. This makes per-agent sampling bit-identical to running
       agent ``i`` solo with the same seed (see
       :class:`RolloutCollector`).
    2. The two structural ``.item()`` calls per agent stay (env wants
       ``int`` + ``float``); the three deferrable syncs are stashed
       on per-agent sidecar buffers.
    3. Step each active agent's env. An agent that terminates on this
       tick is removed from ``active`` — subsequent ticks skip its
       forward entirely.

    Returns ``list[list[Transition]]``: one inner list per agent in
    construction order. Per-agent transition lists are byte-identical
    to those produced by :class:`RolloutCollector` for the same agent
    at the same seed (load-bearing correctness guard, see
    ``tests/test_v2_batched_rollout.py``).

    Parameters
    ----------
    shims:
        ``len(shims) == N``. Each must be a constructed
        :class:`DiscreteActionShim`. Callers should NOT pre-reset.
    policies:
        ``len(policies) == N``. All must share the same architecture
        cluster key (see :func:`cluster_agents_by_arch`). Different
        per-agent weights are fine — the per-agent forward uses each
        agent's own ``policy`` instance.
    device:
        Device for the policies' forward passes. ``"cpu"`` or
        ``"cuda"``. All policies must already be on this device.
    seeds:
        Optional ``len(seeds) == N`` list of integers. Used to seed
        each agent's private RNG state via the save/restore mechanism.
        At ``N == 1`` and ``seeds is None`` the collector takes the
        Session 01 path (single global RNG state) so its output is
        bit-identical to :class:`RolloutCollector` at the same global
        seed.
    """

    def __init__(
        self,
        shims: list[DiscreteActionShim],
        policies: list[BaseDiscretePolicy],
        device: str = "cpu",
        seeds: list[int] | None = None,
    ) -> None:
        if len(shims) != len(policies):
            raise ValueError(
                f"shims and policies length mismatch: "
                f"{len(shims)} vs {len(policies)}",
            )
        n = len(shims)
        if n == 0:
            raise ValueError("BatchedRolloutCollector needs at least one agent")
        if seeds is not None and len(seeds) != n:
            raise ValueError(
                f"seeds length {len(seeds)} does not match n_agents {n}",
            )

        # Validate cluster-key match across all policies.
        keys = cluster_agents_by_arch(policies)
        if len(keys) != 1:
            raise ValueError(
                f"BatchedRolloutCollector requires all policies to share "
                f"an architecture cluster key; got {list(keys.keys())!r}. "
                f"Use cluster_agents_by_arch() to split agents into "
                f"per-cluster collector instances.",
            )

        self.shims = list(shims)
        self.policies = list(policies)
        self.n = n
        self.device = torch.device(device)
        self.seeds = list(seeds) if seeds is not None else None

        first = policies[0]
        self.max_runners = first.max_runners
        self.action_space = first.action_space

        # Last terminal info dict per agent. Trainer / eval reads
        # ``day_pnl`` etc. out of these.
        self.last_infos: list[dict] = [{} for _ in range(n)]

    # ── Public API ────────────────────────────────────────────────────────

    def collect_episode_batch(self) -> list[list[Transition]]:
        """Run one full episode end-to-end for every agent.

        Returns a list of length ``N`` where entry ``i`` is the
        transition list for agent ``i``. Each agent's final
        transition has ``done = True``.
        """
        was_training = [bool(p.training) for p in self.policies]
        for p in self.policies:
            p.eval()
        try:
            return self._collect()
        finally:
            for p, t in zip(self.policies, was_training):
                p.train(t)

    # ── Internals ─────────────────────────────────────────────────────────

    def _collect(self) -> list[list[Transition]]:
        N = self.n
        device = self.device

        # ── Per-agent state ──────────────────────────────────────────────
        latest_obs: list[Any] = [None] * N
        envs = []
        market_to_runner_maps: list[dict[str, dict[int, int]]] = []
        for i in range(N):
            obs_i, _info_i = self.shims[i].reset()
            latest_obs[i] = obs_i
            env_i = self.shims[i].env
            envs.append(env_i)
            mmap: dict[str, dict[int, int]] = {}
            for race_idx, race in enumerate(env_i.day.races):
                mmap[race.market_id] = env_i._runner_maps[race_idx]
            market_to_runner_maps.append(mmap)

        # Per-agent hidden state (separate, NOT batched on dim 1 —
        # vmap would batch but vmap is unavailable for nn.LSTM at
        # this PyTorch version, so each agent owns its own
        # ``(num_layers, 1, hidden)`` tuple).
        hidden_states: list[tuple[torch.Tensor, ...]] = []
        for p in self.policies:
            h0 = p.init_hidden(batch=1)
            hidden_states.append(tuple(t.to(device) for t in h0))

        # Per-agent reward-attribution snapshot. Bet objects are
        # owned by each agent's env — no cross-agent aliasing.
        prev_pnl_by_id: list[dict[int, float]] = [{} for _ in range(N)]

        # Pre-allocated per-agent transfer buffers. Mirror
        # ``RolloutCollector``'s ``obs_buffer`` / ``mask_buffer`` per
        # agent so we don't malloc once per tick × N.
        first_obs_dim = int(np.asarray(latest_obs[0]).shape[-1])
        action_n = int(self.action_space.n)
        obs_buffers = [
            torch.empty(
                (1, first_obs_dim), dtype=torch.float32, device=device,
            )
            for _ in range(N)
        ]
        mask_buffers = [
            torch.empty(
                (1, action_n), dtype=torch.bool, device=device,
            )
            for _ in range(N)
        ]

        # ── Per-agent sidecar buffers (Session 01 deferred-sync). ────────
        pending_log_prob_action: list[list[torch.Tensor]] = [[] for _ in range(N)]
        pending_log_prob_stake: list[list[torch.Tensor]] = [[] for _ in range(N)]
        pending_value_per_runner: list[list[torch.Tensor]] = [[] for _ in range(N)]

        # Per-tick CPU-side bookkeeping (per agent).
        per_tick_obs: list[list[np.ndarray]] = [[] for _ in range(N)]
        per_tick_hidden_in: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in range(N)]
        per_tick_mask: list[list[np.ndarray]] = [[] for _ in range(N)]
        per_tick_action_idx: list[list[int]] = [[] for _ in range(N)]
        per_tick_stake_unit: list[list[float]] = [[] for _ in range(N)]
        per_tick_per_runner_reward: list[list[np.ndarray]] = [[] for _ in range(N)]
        per_tick_done: list[list[bool]] = [[] for _ in range(N)]

        # ── Per-agent RNG state plumbing ─────────────────────────────────
        # Save/restore global RNG state around each agent's tick so
        # per-agent sampling is byte-identical to running each agent
        # solo. Cross-agent leakage is ruled out by construction: no
        # randomness is consumed outside the per-agent restore window.
        use_per_agent_rng = self.seeds is not None
        cuda_device = device if device.type == "cuda" else None

        per_agent_cpu_states: list[torch.Tensor] | None = None
        per_agent_cuda_states: list[torch.Tensor] | None = None
        if use_per_agent_rng:
            saved_cpu_state = torch.get_rng_state()
            saved_cuda_state = (
                torch.cuda.get_rng_state(cuda_device)
                if cuda_device is not None else None
            )
            per_agent_cpu_states = []
            per_agent_cuda_states = (
                [] if cuda_device is not None else None
            )
            for s in self.seeds or []:
                seed_int = int(s) & 0x7FFFFFFF
                torch.manual_seed(seed_int)
                if cuda_device is not None:
                    torch.cuda.manual_seed(seed_int)
                per_agent_cpu_states.append(torch.get_rng_state())
                if per_agent_cuda_states is not None:
                    per_agent_cuda_states.append(
                        torch.cuda.get_rng_state(cuda_device),
                    )
            # Restore the caller's RNG state — we re-enter per-agent
            # state only inside the per-agent tick block.
            torch.set_rng_state(saved_cpu_state)
            if saved_cuda_state is not None:
                torch.cuda.set_rng_state(saved_cuda_state, cuda_device)

        # ── Active-set bookkeeping ───────────────────────────────────────
        active: list[int] = list(range(N))
        n_steps_per_agent: list[int] = [0] * N
        terminated_info: list[dict] = [{} for _ in range(N)]

        # ── R1: batched-forward state ────────────────────────────────────
        # The policy forward runs ONCE per tick over ALL active agents on
        # ``self.device`` (GPU) via ``batched_forward_core`` (vmap over
        # stacked weights). Sampling + env.step stay PER-AGENT on CPU —
        # the env is CPU and per-agent CPU sampling keeps the RNG
        # byte-identical to the solo collector / golden. Params are
        # re-stacked only when the active set changes (weights are frozen
        # during rollout, so per-tick re-stacking was pure waste).
        stacked_active: list[int] | None = None
        stacked: tuple | None = None  # (base, params, buffers)

        with torch.no_grad():
            while active:
                # Snapshot active set for this tick — the env step
                # below may shrink the next tick's active list.
                tick_active = list(active)
                n_act = len(tick_active)

                # Re-stack params only on active-set change.
                if tick_active != stacked_active:
                    stacked = stack_policies(
                        [self.policies[i] for i in tick_active],
                    )
                    stacked_active = list(tick_active)
                base_p, params_p, buffers_p = stacked  # type: ignore[misc]

                # ── Pass A: gather + ONE batched GPU forward ─────────────
                masknp_by_i: dict[int, np.ndarray] = {}
                hidden_in_by_i: dict[int, tuple] = {}
                obs_rows = []
                mask_rows = []
                h_rows = []
                c_rows = []
                for i in tick_active:
                    obs_np = np.asarray(latest_obs[i], dtype=np.float32)
                    obs_rows.append(torch.from_numpy(obs_np))
                    mnp = np.asarray(self.shims[i].get_action_mask(), dtype=bool)
                    masknp_by_i[i] = mnp
                    mask_rows.append(torch.from_numpy(mnp))
                    # Hidden state going INTO the forward — snapshot on CPU
                    # for the Transition record (matches solo capture).
                    hidden_in_by_i[i] = tuple(
                        t.detach().cpu().clone() for t in hidden_states[i]
                    )
                    h_rows.append(hidden_states[i][0])
                    c_rows.append(hidden_states[i][1])
                obs_batch = torch.stack(obs_rows, 0).unsqueeze(1).to(device)
                mask_batch = torch.stack(mask_rows, 0).to(device)
                h_batch = torch.stack(h_rows, 0).to(device)
                c_batch = torch.stack(c_rows, 0).to(device)

                fout = batched_forward_core(
                    base_p, params_p, buffers_p,
                    obs_batch, h_batch, c_batch, mask_batch,
                )
                # One batched device→CPU transfer of everything sampling /
                # recording needs (replaces ~N per-agent syncs).
                ml_cpu = fout.masked_logits.squeeze(1).cpu()          # (n, A)
                sa_cpu = fout.stake_alpha.reshape(n_act, -1).cpu()    # (n, 1)
                sb_cpu = fout.stake_beta.reshape(n_act, -1).cpu()     # (n, 1)
                val_cpu = fout.value_per_runner.squeeze(1).cpu()      # (n, R)
                nh_cpu = fout.new_h.cpu()  # (n, num_layers, 1, H)
                nc_cpu = fout.new_c.cpu()

                # ── Pass B: per-agent sample + env.step (CPU) ────────────
                for k, i in enumerate(tick_active):
                    shim = self.shims[i]
                    obs = latest_obs[i]
                    mask_np = masknp_by_i[i]
                    hidden_in_t = hidden_in_by_i[i]
                    # Advance hidden state for next tick (back onto the
                    # policy/forward device).
                    hidden_states[i] = (
                        nh_cpu[k].to(device), nc_cpu[k].to(device),
                    )

                    logits_i = ml_cpu[k:k + 1]      # (1, A)
                    alpha_i = sa_cpu[k].reshape(1)  # (1,)
                    beta_i = sb_cpu[k].reshape(1)   # (1,)

                    # ── Per-agent RNG window (CPU sampling) ──────────────
                    # The forward (Pass A) consumes no RNG (deterministic),
                    # so wrapping only the sample preserves the solo RNG
                    # consumption order exactly. CPU-only — sampling moved
                    # to CPU, so no CUDA RNG juggling is needed.
                    saved_cpu_state = (
                        torch.get_rng_state() if use_per_agent_rng else None
                    )
                    if use_per_agent_rng and per_agent_cpu_states is not None:
                        torch.set_rng_state(per_agent_cpu_states[i])

                    action_dist = torch.distributions.Categorical(
                        logits=logits_i,
                    )
                    action = action_dist.sample()  # (1,)
                    action_idx = int(action.item())  # STRUCTURAL sync
                    pending_log_prob_action[i].append(
                        action_dist.log_prob(action).detach().squeeze()
                    )

                    stake_dist = torch.distributions.Beta(alpha_i, beta_i)
                    stake_unit_t = stake_dist.sample()  # (1,)
                    stake_unit = float(stake_unit_t.item())  # STRUCTURAL sync
                    if action_uses_stake(self.action_space, action_idx):
                        log_prob_stake_t = (
                            stake_dist.log_prob(stake_unit_t).detach().squeeze()
                        )
                    else:
                        log_prob_stake_t = torch.zeros(
                            (), dtype=stake_unit_t.dtype,
                        )
                    pending_log_prob_stake[i].append(log_prob_stake_t)
                    pending_value_per_runner[i].append(val_cpu[k].clone())

                    if use_per_agent_rng and per_agent_cpu_states is not None:
                        per_agent_cpu_states[i] = torch.get_rng_state()
                        torch.set_rng_state(saved_cpu_state)

                    # ── Env step (sequential, per agent) ─────────────────
                    env_i = envs[i]
                    bm = env_i.bet_manager
                    budget = bm.budget if bm is not None else 0.0
                    stake_pounds = max(stake_unit * budget, MIN_BET_STAKE)

                    next_obs, reward, terminated, truncated, info = shim.step(
                        action_idx,
                        stake=stake_pounds,
                        arb_spread=None,
                    )
                    done = bool(terminated or truncated)

                    per_runner_reward = self._attribute_step_reward(
                        env=env_i,
                        step_reward=float(reward),
                        prev_pnl_by_id=prev_pnl_by_id[i],
                        market_to_runner_map=market_to_runner_maps[i],
                    )

                    per_tick_obs[i].append(np.asarray(obs, dtype=np.float32))
                    per_tick_hidden_in[i].append(hidden_in_t)
                    per_tick_mask[i].append(mask_np)
                    per_tick_action_idx[i].append(action_idx)
                    per_tick_stake_unit[i].append(stake_unit)
                    per_tick_per_runner_reward[i].append(per_runner_reward)
                    per_tick_done[i].append(done)

                    latest_obs[i] = next_obs
                    n_steps_per_agent[i] += 1
                    if done:
                        terminated_info[i] = info or {}
                        active.remove(i)

        # ── End-of-episode batched materialisation, per agent ────────────
        out: list[list[Transition]] = []
        for i in range(N):
            n_steps = n_steps_per_agent[i]
            if n_steps > 0:
                log_prob_action_arr = (
                    torch.stack(pending_log_prob_action[i]).cpu().numpy()
                    .astype(np.float32)
                )
                log_prob_stake_arr = (
                    torch.stack(pending_log_prob_stake[i]).cpu().numpy()
                    .astype(np.float32)
                )
                value_per_runner_arr = (
                    torch.stack(pending_value_per_runner[i]).cpu().numpy()
                    .astype(np.float32)
                )
            else:
                log_prob_action_arr = np.zeros((0,), dtype=np.float32)
                log_prob_stake_arr = np.zeros((0,), dtype=np.float32)
                value_per_runner_arr = np.zeros(
                    (0, self.max_runners), dtype=np.float32,
                )

            transitions = [
                Transition(
                    obs=per_tick_obs[i][k],
                    hidden_state_in=per_tick_hidden_in[i][k],
                    mask=per_tick_mask[i][k],
                    action_idx=per_tick_action_idx[i][k],
                    stake_unit=per_tick_stake_unit[i][k],
                    log_prob_action=float(log_prob_action_arr[k]),
                    log_prob_stake=float(log_prob_stake_arr[k]),
                    value_per_runner=value_per_runner_arr[k],
                    per_runner_reward=per_tick_per_runner_reward[i][k],
                    done=per_tick_done[i][k],
                )
                for k in range(n_steps)
            ]
            out.append(transitions)
            self.last_infos[i] = terminated_info[i]

        logger.info(
            "BatchedRolloutCollector: collected N=%d agents, "
            "steps per agent=%s",
            N, n_steps_per_agent,
        )
        return out

    # ── Per-step reward attribution (per agent) ──────────────────────────

    def _attribute_step_reward(
        self,
        env,
        step_reward: float,
        prev_pnl_by_id: dict[int, float],
        market_to_runner_map: dict[str, dict[int, int]],
    ) -> np.ndarray:
        """Same logic as :meth:`RolloutCollector._attribute_step_reward`.

        Lifted here unchanged so cluster-1 N=1 reductions stay
        bit-identical to the single-agent path. The function is pure
        and operates on per-agent state; sharing it via inheritance
        would risk the env-internals coupling Session 01's collector
        already pays for.
        """
        per_runner = np.zeros(self.max_runners, dtype=np.float64)

        live_bets = (
            env.bet_manager.bets if env.bet_manager is not None else []
        )
        all_bets = list(env.all_settled_bets) + list(live_bets)

        attributed_total = 0.0
        for bet in all_bets:
            bet_id = id(bet)
            prev_pnl = prev_pnl_by_id.get(bet_id, 0.0)
            cur_pnl = float(bet.pnl)
            delta = cur_pnl - prev_pnl
            if delta == 0.0:
                prev_pnl_by_id[bet_id] = cur_pnl
                continue

            runner_map = market_to_runner_map.get(bet.market_id)
            if runner_map is not None:
                slot = runner_map.get(bet.selection_id)
                if slot is not None and slot < self.max_runners:
                    per_runner[slot] += delta
                    attributed_total += delta
            prev_pnl_by_id[bet_id] = cur_pnl

        residual = step_reward - attributed_total
        per_runner += residual / self.max_runners

        total = float(per_runner.sum())
        if not np.isclose(
            total, step_reward,
            rtol=0.0, atol=_ATTRIBUTION_TOLERANCE,
        ):
            raise AssertionError(
                f"per-runner reward attribution drift: sum={total!r} "
                f"vs scalar reward={step_reward!r} "
                f"(diff={total - step_reward!r}). "
                "A reward component in env._settle_current_race or "
                "the per-step path isn't being captured by the "
                "bet-pnl-delta + equal-residual split.",
            )

        return per_runner.astype(np.float32, copy=False)
