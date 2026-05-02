"""Rollout collector for the v2 discrete-PPO trainer.

Phase 2, Session 01 deliverable. Drives Phase 1's
:class:`agents_v2.env_shim.DiscreteActionShim` and a
:class:`agents_v2.discrete_policy.BaseDiscretePolicy` subclass through
one episode (one day) and returns a list of :class:`Transition`
objects ready for the (Session 02) PPO update.

What this does AND DOES NOT do:

- DOES: sample masked-categorical + Beta-stake actions, step the env,
  attribute per-step reward across runners via ``BetManager`` /
  ``env.all_settled_bets`` PnL deltas, capture ``hidden_state_in``
  BEFORE the forward pass, store the rollout-time mask alongside
  every transition.
- DOES NOT: compute a loss, take a gradient step, normalise
  advantages, run GAE. Those live in Session 02 / Session 03 of
  Phase 2 (``gae.py`` exists in this session for testing, but the
  collector doesn't call it).

Hard constraints (Phase 2 purpose §"Hard constraints", session
prompt §"Hard constraints"):

- No env edits.
- No re-import of v1 trainer / policy.
- No new shaped rewards. The collector only re-organises existing
  reward into per-runner buckets.
- Per-runner reward attribution must SUM to the env's scalar reward
  to floating-point tolerance (per-step assertion below).
- ``hidden_state_in`` captured BEFORE the forward pass, not after.

Phase 4 Session 03 (2026-05-02): module-level disable of
``torch.distributions.Distribution`` arg validation. The per-tick
``Beta(out.stake_alpha, out.stake_beta)`` construction and the
policy-side ``Categorical(logits=...)`` construction (in
``agents_v2/discrete_policy.py``) each pay a parameter-validation
+ broadcasting cost on every ``__init__``. At ~12 k ticks/episode
that compounds. The validations were never expected to fail (α/β
come from softplus heads → strictly > 0; logits come from a linear
head → finite); they are guards against malformed inputs, not
load-bearing correctness checks. Disabling them is bit-identical:
no RNG-consuming op runs inside the validation branch, only
``constraints.check`` calls that raise on violation. Verified by
``tests/test_v2_rollout_distributions.py`` — sample / log_prob
outputs are byte-equal across the toggle at fixed seed.

This toggle is process-wide (it sets a class attribute on
``torch.distributions.Distribution``). Importing this module is
the trigger; the trainer / batched_rollout paths inherit the
disabled state without further action.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributions

from agents_v2.discrete_policy import BaseDiscretePolicy
from agents_v2.env_shim import DiscreteActionShim
from env.bet_manager import MIN_BET_STAKE, BetOutcome
from training_v2.discrete_ppo.transition import Transition, action_uses_stake

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


__all__ = ["RolloutCollector"]


# Phase 4 Session 03: disable per-init parameter validation on every
# torch.distributions.Distribution subclass globally. See the module
# docstring for the bit-identity argument and the regression guard
# in tests/test_v2_rollout_distributions.py.
torch.distributions.Distribution.set_default_validate_args(False)


_ATTRIBUTION_TOLERANCE = 1e-4


class _AttributionState:
    """Per-episode bookkeeping for incremental per-runner attribution.

    Phase 4 Session 01 (2026-05-02): replaces the previous O(n²)-per-
    episode walk over ``all_settled_bets + bm.bets`` with an O(open-
    pending-bets) per-tick walk over a "pending pnl" set.

    A bet enters ``pending_bets`` the tick it first appears in either
    ``env._settled_bets`` (race-end extends) or the current
    ``bm.bets``, and leaves once ``bet.outcome != BetOutcome.UNSETTLED``
    (after which ``bet.pnl`` is immutable; verified by audit of
    ``env/bet_manager.py`` — only ``settle_race`` and ``void_race``
    write ``bet.pnl``, and both transition outcome out of UNSETTLED in
    the same call).

    Iteration order matches the legacy walk for bit-identity:
    ``_settled_bets`` is scanned before ``bm.bets`` each tick, so a bet
    that lands in ``_settled_bets`` at race-end (after being placed at
    the same tick the race settled) is added in the same position the
    legacy ``list(env.all_settled_bets) + list(bm.bets)`` order would
    have iterated it. Re-inserting an existing key into a Python dict
    does not move it, so a bet first added via ``bm.bets`` keeps its
    original placement-order slot.
    """

    __slots__ = (
        "pending_bets",
        "prev_pnl_by_id",
        "settled_count",
        "live_count",
        "bm_id",
        "iter_history",
    )

    def __init__(self) -> None:
        # Insertion-ordered. Keyed by id(bet) → bet.
        self.pending_bets: dict[int, object] = {}
        self.prev_pnl_by_id: dict[int, float] = {}
        # Suffix-scan watermarks: how much of each list we've already
        # ingested. Avoids re-walking the prefix every tick.
        self.settled_count: int = 0
        self.live_count: int = 0
        # Identity (not equality) of the BetManager whose bets list
        # ``live_count`` indexes into. The env replaces ``bet_manager``
        # at every race transition; the new instance starts with
        # ``bets == []`` so we reset ``live_count`` on identity change.
        self.bm_id: int | None = None
        # Per-tick iteration count over ``pending_bets``. Recorded for
        # the bounded-size and zero-scan-on-no-bet-tick regression
        # guards. Cheap (one int append per tick); off the hot path.
        self.iter_history: list[int] = []


class RolloutCollector:
    """Drive a (shim, policy) pair through one episode and collect transitions.

    Parameters
    ----------
    shim:
        A constructed :class:`DiscreteActionShim`. The collector calls
        ``shim.reset`` itself; callers should NOT pre-reset.
    policy:
        Any :class:`BaseDiscretePolicy` subclass (Phase 1's
        :class:`DiscreteLSTMPolicy` is the typical instance). The
        policy stays in eval mode for the duration of rollout —
        gradients are disabled via ``torch.no_grad``.
    device:
        Device for the policy's forward passes. Defaults to ``"cpu"``;
        callers in Phase 3 may pass ``"cuda"``.
    """

    def __init__(
        self,
        shim: DiscreteActionShim,
        policy: BaseDiscretePolicy,
        device: str = "cpu",
    ) -> None:
        self.shim = shim
        self.policy = policy
        self.device = torch.device(device)
        self.max_runners = shim.max_runners
        self.action_space = shim.action_space
        # Last terminal info dict from the most recent episode. Set by
        # _collect on the final step; consumers (e.g. trainer) read
        # day_pnl out of this for episode-level diagnostics.
        self.last_info: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def collect_episode(self) -> list[Transition]:
        """Run one full episode end-to-end.

        Returns the list of transitions in order. The final
        transition's ``done`` is ``True``; intermediate transitions'
        ``done`` is ``False``. The collector enforces (per step) that
        per-runner reward attribution sums to the env's scalar
        reward — failure raises immediately so attribution drift
        can't silently corrupt the PPO update.
        """
        was_training = self.policy.training
        self.policy.eval()
        try:
            return self._collect()
        finally:
            self.policy.train(was_training)

    # ── Internals ──────────────────────────────────────────────────────────

    def _collect(self) -> list[Transition]:
        shim = self.shim
        env = shim.env
        policy = self.policy

        obs, _info = shim.reset()

        # Pre-build market_id → race_idx → runner_map lookup so we
        # can attribute a bet's PnL to the runner slot in *its own*
        # race (not the currently-active one — settle steps may have
        # already advanced ``env._race_idx`` by the time we read
        # ``all_settled_bets``).
        market_to_runner_map: dict[str, dict[int, int]] = {}
        for race_idx, race in enumerate(env.day.races):
            market_to_runner_map[race.market_id] = env._runner_maps[race_idx]

        hidden_state = policy.init_hidden(batch=1)
        hidden_state = tuple(t.to(self.device) for t in hidden_state)

        # Phase 4 Session 01: incremental per-runner attribution.
        # Replaces the previous "walk every bet ever placed every
        # tick" loop with a pending-pnl set whose membership is
        # bounded by the open-position count (typically 0–50, vs
        # cumulative bets-this-episode in the hundreds-to-thousands
        # by tick 11k of a 12k-tick day). See ``_AttributionState``
        # docstring + ``_attribute_step_reward`` for ENTRY/EXIT rules.
        attribution_state = _AttributionState()

        # Phase 3 Session 01: pre-allocated per-step transfer buffers.
        # Mirrors v1's ``agents/ppo_trainer.py:1384-1390`` pattern.
        # On CUDA this avoids a per-tick CUDA malloc (small per-step
        # win, compounds across ~12k ticks/episode); on CPU the buffer
        # reuse is cheap enough to take the same path.
        obs_dim = int(np.asarray(obs).shape[-1])
        action_n = int(self.action_space.n)
        obs_buffer = torch.empty(
            (1, obs_dim), dtype=torch.float32, device=self.device,
        )
        mask_buffer = torch.empty(
            (1, action_n), dtype=torch.bool, device=self.device,
        )

        # Phase 4 Session 02: single-allocation obs / mask buffers.
        # Pre-Session-02 each tick did TWO np.asarray casts + TWO
        # buffer copies — once to fill the device buffer, once to
        # append into per_tick_obs / per_tick_mask CPU lists. At
        # ~12k ticks/episode that's 48k unnecessary allocations.
        # Now: one contiguous float32 / bool buffer per episode,
        # written once per tick at row n_steps. The device buffer
        # copy reads from the same row (a view, not a copy). The
        # Transition's obs / mask is a view into the row — the
        # PPO update consumer (``trainer._ppo_update``) does
        # ``np.stack([tr.obs ...])`` which copies into a new
        # contiguous array, so view aliasing is safe.
        n_steps_estimate = self._estimate_max_steps(env)
        obs_arr = np.empty((n_steps_estimate, obs_dim), dtype=np.float32)
        mask_arr = np.empty((n_steps_estimate, action_n), dtype=bool)

        # Phase 4 Session 04: pre-allocated hidden-state capture
        # buffers. Pre-Session-04 each tick did
        # ``tuple(t.detach().clone() for t in hidden_state)`` —
        # 2 × ``(num_layers, 1, hidden_size)`` tensor allocations
        # plus 2 ``.clone()`` memcopies per tick (LSTM family). At
        # ~12 k ticks/episode that's 24 k unnecessary allocations
        # churning the allocator.
        #
        # The clones are LOAD-BEARING — subsequent LSTM forwards
        # mutate the rolling hidden state, so we can't store a
        # view into ``hidden_state`` itself. But the per-tick
        # allocation IS unnecessary: pre-allocate one
        # ``(n_steps_estimate, *hidden_shape)`` buffer per element
        # of the hidden-state tuple at episode start, ``.copy_()``
        # each tick's snapshot into a slice view, and the captured
        # ``hidden_in_t`` is the slice view (not a fresh tensor).
        #
        # View-vs-copy semantics: ``nn.LSTM.forward`` returns a NEW
        # ``(h, c)`` tuple (a fresh allocation, not a mutation of
        # the input), and ``hidden_state = out.new_hidden_state``
        # rebinds the local variable to that new tuple. The
        # buffer slice we snapshotted on tick T is a view into
        # ``hidden_buffers[k][T]``; subsequent ticks write to
        # ``[T+1]``, which is a different memory region. So
        # earlier captures stay correct even after the rolling
        # ``hidden_state`` keeps advancing.
        #
        # Generic over the hidden_state tuple's shape — works for
        # the LSTM / TimeLSTM ``(h, c)`` shape AND the transformer's
        # ``(buffer, valid_count)`` shape. Each element of the
        # tuple gets its own buffer matching the element's
        # ``shape / dtype / device``.
        #
        # PPO update consumer: ``policy.pack_hidden_states`` does
        # ``torch.cat([s[k] for s in states], dim=...)`` which
        # always copies into a fresh tensor, so view aliasing
        # does NOT leak into the gradient path (same argument as
        # Session 02's obs / mask buffer).
        hidden_buffers: list[torch.Tensor] = [
            torch.empty(
                (n_steps_estimate, *t.shape),
                dtype=t.dtype,
                device=t.device,
            )
            for t in hidden_state
        ]

        # Throughput-fix Session 01: defer the three deferrable
        # CUDA→CPU sync points (log_prob_action, log_prob_stake,
        # value_per_runner) by stashing 0-d / 1-d device tensors in
        # sidecar buffers and doing one batched ``.cpu()`` per buffer
        # at end-of-episode. Cuts ~3 syncs per tick × ~12 k ticks ≈
        # 36 k syncs to 3 batched transfers.
        #
        # The two structural ``.item()`` calls (action_idx,
        # stake_unit) STAY — the CPU env consumes ``int`` and
        # ``float`` every tick.
        pending_log_prob_action: list[torch.Tensor] = []
        pending_log_prob_stake: list[torch.Tensor] = []
        pending_value_per_runner: list[torch.Tensor] = []

        # Per-tick CPU-side bookkeeping. Transition objects are built
        # at end-of-episode so the deferred device tensors can be
        # materialised in one batched transfer (Transition is a
        # frozen dataclass; we can't backfill in-place).
        # Phase 4 Session 02: obs / mask now live in the shared
        # contiguous buffers above — no per-tick list.
        per_tick_hidden_in: list[tuple[torch.Tensor, ...]] = []
        per_tick_action_idx: list[int] = []
        per_tick_stake_unit: list[float] = []
        per_tick_per_runner_reward: list[np.ndarray] = []
        per_tick_done: list[bool] = []

        done = False
        n_steps = 0
        last_info: dict = {}

        with torch.no_grad():
            while not done:
                # Phase 4 Session 02: single materialisation per tick.
                # Grow the contiguous buffers if the episode exceeds
                # the upper-bound estimate. The grow path is once-per-
                # episode at most in practice; the warning surfaces a
                # bad estimate so it can be tuned.
                if n_steps >= obs_arr.shape[0]:
                    obs_arr, mask_arr = self._grow_obs_mask_buffers(
                        obs_arr, mask_arr, n_steps,
                    )

                # Single write per tick into the contiguous row;
                # the device buffer copy reads from the same row.
                obs_arr[n_steps] = obs
                mask_np = shim.get_action_mask()
                mask_arr[n_steps] = mask_np
                obs_buffer.copy_(
                    torch.from_numpy(obs_arr[n_steps]).unsqueeze(0)
                )
                mask_buffer.copy_(
                    torch.from_numpy(mask_arr[n_steps]).unsqueeze(0)
                )
                obs_t = obs_buffer
                mask_t = mask_buffer

                # Capture the hidden state that's about to be passed
                # INTO the forward pass. The PPO update needs this
                # exact state to reproduce rollout-time log-probs.
                # Phase 3 Session 01b: keep the tensor device-resident
                # — moving it to CPU here forced ~24 k CUDA→CPU sync
                # barriers / episode (the dominant cost on the CUDA
                # path; see findings.md "Session 01"). ``.detach()``
                # peels off the autograd tape; the snapshot into the
                # pre-allocated buffer slice replaces the previous
                # per-tick ``.clone()`` (Phase 4 Session 04). The
                # snapshot is load-bearing because the rolling
                # ``hidden_state`` keeps mutating across ticks.
                if n_steps >= hidden_buffers[0].shape[0]:
                    hidden_buffers = self._grow_hidden_buffers(
                        hidden_buffers, n_steps,
                    )
                for buf, t in zip(hidden_buffers, hidden_state):
                    buf[n_steps].copy_(t.detach())
                hidden_in_t = tuple(buf[n_steps] for buf in hidden_buffers)

                out = policy(obs_t, hidden_state=hidden_state, mask=mask_t)
                hidden_state = out.new_hidden_state

                # Sample action and stake.
                action = out.action_dist.sample()              # (1,) long
                # STRUCTURAL sync: env.step needs an int.
                action_idx = int(action.item())

                # DEFERRED: stash the 0-d device tensor for the log
                # prob; it gets materialised at end-of-episode.
                log_prob_action_t = (
                    out.action_dist.log_prob(action).detach().squeeze()
                )
                pending_log_prob_action.append(log_prob_action_t)

                # Beta sample is in (0, 1). The shim takes a £ stake
                # so we re-scale outside the policy (per Phase 1
                # contract, env_shim.py::_encode_stake handles the
                # final clamp + budget-fraction encoding).
                stake_dist = torch.distributions.Beta(
                    out.stake_alpha, out.stake_beta,
                )
                stake_unit_t = stake_dist.sample()              # (1,)
                # STRUCTURAL sync: env.step's stake_pounds needs a
                # float (BetManager applies MIN_BET_STAKE clamp).
                stake_unit = float(stake_unit_t.item())

                bm = env.bet_manager
                budget = bm.budget if bm is not None else 0.0
                stake_pounds = max(stake_unit * budget, MIN_BET_STAKE)

                # DEFERRED: stake log-prob.  The mask
                # ``action_uses_stake`` decides whether this slot
                # carries a real value or a placeholder zero — the
                # PPO update masks the placeholder out before the
                # gradient flows, so any constant works. We keep the
                # device tensor either way so the batched cpu()
                # transfer at end-of-episode sees a uniform list.
                if action_uses_stake(self.action_space, action_idx):
                    log_prob_stake_t = (
                        stake_dist.log_prob(stake_unit_t).detach().squeeze()
                    )
                else:
                    log_prob_stake_t = torch.zeros(
                        (), dtype=stake_unit_t.dtype, device=self.device,
                    )
                pending_log_prob_stake.append(log_prob_stake_t)

                # DEFERRED: per-runner value head.  Keep on device;
                # we'll do one batched stack/cpu/numpy at episode end.
                value_per_runner_t = out.value_per_runner.detach().squeeze(0)
                pending_value_per_runner.append(value_per_runner_t)

                # Step the env. The shim's step returns the extended
                # (with scorer features) obs and forwards the env's
                # info untouched.
                next_obs, reward, terminated, truncated, info = shim.step(
                    action_idx,
                    stake=stake_pounds,
                    arb_spread=None,
                )
                done = bool(terminated or truncated)

                per_runner_reward = self._attribute_step_reward(
                    env=env,
                    step_reward=float(reward),
                    state=attribution_state,
                    market_to_runner_map=market_to_runner_map,
                )

                # Phase 4 Session 02: obs / mask already in the
                # contiguous buffers; the Transition reads from
                # ``obs_arr[i]`` / ``mask_arr[i]`` (a view) at end-
                # of-episode. No per-tick CPU-list copy.
                per_tick_hidden_in.append(hidden_in_t)
                per_tick_action_idx.append(action_idx)
                per_tick_stake_unit.append(stake_unit)
                per_tick_per_runner_reward.append(per_runner_reward)
                per_tick_done.append(done)

                obs = next_obs
                n_steps += 1
                if done:
                    last_info = info or {}

        # End-of-episode batched materialisation. Three CUDA→CPU
        # transfers total (was ~3 × n_steps). For a typical
        # ~12 k-tick episode this collapses ~36 k per-tick syncs.
        if n_steps > 0:
            log_prob_action_arr = (
                torch.stack(pending_log_prob_action).cpu().numpy()
                .astype(np.float32)
            )
            log_prob_stake_arr = (
                torch.stack(pending_log_prob_stake).cpu().numpy()
                .astype(np.float32)
            )
            value_per_runner_arr = (
                torch.stack(pending_value_per_runner).cpu().numpy()
                .astype(np.float32)
            )
        else:
            # Defensive: empty episode shouldn't happen but the env
            # technically allows it. Keep shapes consistent with the
            # n_steps>0 branch so downstream stacking doesn't break.
            log_prob_action_arr = np.zeros((0,), dtype=np.float32)
            log_prob_stake_arr = np.zeros((0,), dtype=np.float32)
            value_per_runner_arr = np.zeros(
                (0, self.max_runners), dtype=np.float32,
            )

        # Phase 4 Session 02: ``obs_arr[i]`` / ``mask_arr[i]`` are
        # views into the contiguous per-episode buffers. The PPO
        # update consumer (``trainer._ppo_update``) does
        # ``np.stack([tr.obs ...]).astype(np.float32)`` which copies
        # into a new contiguous array, so view aliasing does NOT
        # leak into the gradient path. Verified by audit of
        # ``training_v2/discrete_ppo/trainer.py`` at the start of
        # this session — no in-place mutation of ``Transition.obs``
        # or ``Transition.mask`` anywhere.
        transitions: list[Transition] = [
            Transition(
                obs=obs_arr[i],
                hidden_state_in=per_tick_hidden_in[i],
                mask=mask_arr[i],
                action_idx=per_tick_action_idx[i],
                stake_unit=per_tick_stake_unit[i],
                log_prob_action=float(log_prob_action_arr[i]),
                log_prob_stake=float(log_prob_stake_arr[i]),
                value_per_runner=value_per_runner_arr[i],
                per_runner_reward=per_tick_per_runner_reward[i],
                done=per_tick_done[i],
            )
            for i in range(n_steps)
        ]

        self.last_info = last_info
        # Stashed for tests / debug. The pending set should be empty
        # at end-of-episode (every bet has settled), but we keep the
        # whole state so ``test_pending_set_size_bounded_across_
        # episode`` can assert on the trajectory of the iteration
        # bound, not just the terminal state.
        self.last_attribution_state = attribution_state

        logger.info(
            "RolloutCollector: collected %d transitions (terminated)",
            n_steps,
        )
        return transitions

    # ── Per-episode obs / mask buffer helpers (Phase 4 Session 02) ─────────

    def _estimate_max_steps(self, env) -> int:
        """Upper-bound the number of env.step calls in this episode.

        Each ``env.step`` advances ``_tick_idx`` by one within the
        active race or transitions to the next race. The total step
        count is therefore bounded by ``sum_r len(race.ticks)`` plus
        a small per-race margin to absorb any transition / settle
        steps the env emits at race boundaries. Adding +1 per race
        plus a final +1 keeps the estimate exact-or-loose so the
        grow path is once-per-episode at most in practice (and
        typically never).

        If ``env.day.races`` is somehow unavailable (e.g. a test
        harness with a stub env), fall back to a generous default
        of 20 000 — matches the session prompt's recommendation.
        """
        races = getattr(getattr(env, "day", None), "races", None)
        if races is None:
            return 20_000
        return sum(len(r.ticks) for r in races) + len(races) + 1

    def _grow_obs_mask_buffers(
        self,
        obs_arr: np.ndarray,
        mask_arr: np.ndarray,
        n_filled: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Double the obs / mask buffer capacity, preserving filled rows.

        Returns the (obs, mask) pair of fresh np arrays. Rows
        ``[0, n_filled)`` are copied from the old buffers so any
        Transition built later that reads from the new buffer sees
        the same values for those rows. The OLD buffers are dropped
        — their refcounts go to zero unless the caller still holds
        references; in this collector they don't, so they're GCed.

        ``np.empty`` (not ``np.zeros``) is fine here: only
        ``[0, n_filled)`` is read after the copy and ``[n_filled,
        new_n)`` is overwritten by future ticks before any read.
        """
        obs_dim = obs_arr.shape[1]
        action_n = mask_arr.shape[1]
        new_n = obs_arr.shape[0] * 2
        new_obs = np.empty((new_n, obs_dim), dtype=np.float32)
        new_obs[:n_filled] = obs_arr[:n_filled]
        new_mask = np.empty((new_n, action_n), dtype=bool)
        new_mask[:n_filled] = mask_arr[:n_filled]
        logger.warning(
            "RolloutCollector: obs/mask buffer grow fired (was %d, "
            "now %d) — _estimate_max_steps undercounted; tune for "
            "this day shape if the warning recurs",
            obs_arr.shape[0], new_n,
        )
        return new_obs, new_mask

    def _grow_hidden_buffers(
        self,
        buffers: list[torch.Tensor],
        n_filled: int,
    ) -> list[torch.Tensor]:
        """Double the hidden-state capture buffers, preserving filled rows.

        Phase 4 Session 04: symmetric to ``_grow_obs_mask_buffers``
        but on the device-resident torch buffers. Doubles capacity
        along the leading time axis and copies the filled prefix
        ``[0, n_filled)`` from each old buffer into the new one.

        Why we copy the prefix even though existing
        ``per_tick_hidden_in`` views into the OLD buffers stay
        valid (their storage is held alive by the views): the
        copy keeps every captured snapshot in ONE buffer per
        tuple element, which makes the
        ``test_hidden_state_buffer_allocated_once_per_episode``
        regression guard a clean signal — under normal
        operation the buffer never grows, and even when it
        does, the post-grow buffer holds every snapshot. Without
        the copy, a future grow would split snapshots across
        old + new buffers, complicating the test for
        re-introduction of per-tick allocation.

        Returns the list of fresh torch buffers in the same
        order as ``buffers``. Mirrors the obs/mask pattern's
        once-per-episode (typically never) growth path.
        """
        new_buffers: list[torch.Tensor] = []
        old_n = buffers[0].shape[0]
        new_n = old_n * 2
        for old in buffers:
            new = torch.empty(
                (new_n, *old.shape[1:]),
                dtype=old.dtype,
                device=old.device,
            )
            new[:n_filled].copy_(old[:n_filled])
            new_buffers.append(new)
        logger.warning(
            "RolloutCollector: hidden-state buffer grow fired "
            "(was %d, now %d) — _estimate_max_steps undercounted; "
            "tune for this day shape if the warning recurs",
            old_n, new_n,
        )
        return new_buffers

    # ── Per-step reward attribution ────────────────────────────────────────

    def _attribute_step_reward(
        self,
        env,
        step_reward: float,
        state: _AttributionState,
        market_to_runner_map: dict[str, dict[int, int]],
    ) -> np.ndarray:
        """Split a scalar step reward across runner slots.

        Phase 4 Session 01 (2026-05-02): incremental tracking via a
        pending-pnl set. Same algebra as the pre-Phase-4 walk; same
        invariant assert; same numbers. Bit-identical on CPU at fixed
        seed (regression guard:
        ``tests/test_v2_rollout_per_runner_attribution.py``).

        Strategy:
        1. ENTRY — scan the suffix of ``env._settled_bets`` and
           ``env.bet_manager.bets`` for new bet objects since last
           tick. Add to ``state.pending_bets``. Reset the
           ``bm.bets`` watermark on race transition (BetManager
           identity change). Typical per-tick add count: 0–3.
        2. For each bet in ``state.pending_bets`` (insertion order
           — matches the legacy ``all_settled_bets + bm.bets``
           walk's bit-identity-relevant ordering): compute
           ``delta = bet.pnl - state.prev_pnl_by_id[id(bet)]`` and
           credit to the bet's runner slot.
        3. EXIT — once ``bet.outcome != BetOutcome.UNSETTLED``,
           ``bet.pnl`` is final (verified by audit of
           ``env/bet_manager.py`` — only ``settle_race`` and
           ``void_race`` write ``bet.pnl``, both transition outcome
           out of UNSETTLED in the same call). Mark for removal
           after the iteration; remove post-loop so dict mutation
           during iteration is avoided.
        4. Distribute the residual ``step_reward - attributed_
           total`` equally across all ``max_runners`` slots. The
           residual covers MTM shaping, terminal-bonus, and per-race
           shaped terms (efficiency_cost / precision / drawdown /
           matured-arb / open-cost) that aren't tied to a single
           runner.

        Returns ``per_runner_reward`` of shape ``(max_runners,)``.
        Asserts ``per_runner_reward.sum() ≈ step_reward`` — drift
        here would silently corrupt the PPO update.
        """
        pending_bets = state.pending_bets
        prev_pnl_by_id = state.prev_pnl_by_id

        # ENTRY: bets newly extended into ``_settled_bets`` (race-end
        # moves bm.bets → _settled_bets in one shot, so on a settle
        # tick this catches any bets placed at that very tick that
        # were never seen via the bm.bets scan path because bm has
        # since been replaced).
        settled_list = env._settled_bets
        new_settled_n = len(settled_list)
        if new_settled_n > state.settled_count:
            for i in range(state.settled_count, new_settled_n):
                bet = settled_list[i]
                bid = id(bet)
                # Re-inserting an existing key into a Python dict
                # preserves its original position, so a bet first
                # added via the bm.bets path keeps its placement-
                # order slot. We still skip the assignment to avoid
                # the dict's reference rebind cost.
                if bid not in pending_bets:
                    pending_bets[bid] = bet
            state.settled_count = new_settled_n

        # ENTRY: bets newly placed in the current race's BetManager.
        bm = env.bet_manager
        if bm is not None:
            bm_id = id(bm)
            if bm_id != state.bm_id:
                # New BetManager (race transition). The previous bm's
                # bets are already in ``_settled_bets`` (extended in
                # env.step before bm replacement) and so are already
                # captured by the suffix scan above. Reset the
                # watermark for the fresh empty list.
                state.bm_id = bm_id
                state.live_count = 0
            live_list = bm.bets
            new_live_n = len(live_list)
            if new_live_n > state.live_count:
                for i in range(state.live_count, new_live_n):
                    bet = live_list[i]
                    bid = id(bet)
                    if bid not in pending_bets:
                        pending_bets[bid] = bet
                state.live_count = new_live_n

        # Per-tick iteration count for the regression guards (off the
        # arithmetic path; cheap one-int append per tick).
        state.iter_history.append(len(pending_bets))

        per_runner = np.zeros(self.max_runners, dtype=np.float64)
        attributed_total = 0.0
        to_remove: list[int] = []

        for bid, bet in pending_bets.items():
            prev_pnl = prev_pnl_by_id.get(bid, 0.0)
            cur_pnl = float(bet.pnl)
            delta = cur_pnl - prev_pnl
            if delta != 0.0:
                runner_map = market_to_runner_map.get(bet.market_id)
                if runner_map is not None:
                    slot = runner_map.get(bet.selection_id)
                    if slot is not None and slot < self.max_runners:
                        per_runner[slot] += delta
                        attributed_total += delta
            prev_pnl_by_id[bid] = cur_pnl
            # EXIT: a finalised bet leaves the pending set after this
            # tick's delta is captured. ``bet.outcome != UNSETTLED``
            # implies ``bet.pnl`` is immutable (audit: only
            # settle_race and void_race write pnl, both in the same
            # call that transitions outcome). Subsequent ticks would
            # add zero — skipping them is the whole point.
            if bet.outcome is not BetOutcome.UNSETTLED:
                to_remove.append(bid)

        for bid in to_remove:
            del pending_bets[bid]

        residual = step_reward - attributed_total
        per_runner += residual / self.max_runners

        # Belt-and-braces invariant: sum across runners equals the
        # env's scalar reward to floating-point tolerance. Drift here
        # is the load-bearing failure mode the session prompt
        # flagged — stop and investigate, don't silence.
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
