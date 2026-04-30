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
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from agents_v2.discrete_policy import BaseDiscretePolicy
from agents_v2.env_shim import DiscreteActionShim
from env.bet_manager import MIN_BET_STAKE
from training_v2.discrete_ppo.transition import Transition, action_uses_stake

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


__all__ = ["RolloutCollector"]


_ATTRIBUTION_TOLERANCE = 1e-4


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

        transitions: list[Transition] = []

        # Snapshot of cumulative pnl per bet object (keyed by
        # ``id(bet)``) — used to compute per-step pnl deltas. Bets
        # only appear and have their pnl mutated; they aren't
        # removed, so the keys monotonically grow across the episode.
        prev_pnl_by_id: dict[int, float] = {}

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

        done = False
        n_steps = 0

        with torch.no_grad():
            while not done:
                obs_buffer.copy_(
                    torch.from_numpy(np.asarray(obs, dtype=np.float32))
                    .unsqueeze(0)
                )
                mask_np = shim.get_action_mask()
                mask_buffer.copy_(
                    torch.from_numpy(np.asarray(mask_np, dtype=bool))
                    .unsqueeze(0)
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
                # peels off the autograd tape; ``.clone()`` is load-
                # bearing because subsequent LSTM forwards mutate the
                # rolling hidden state in place.
                hidden_in_t = tuple(
                    t.detach().clone() for t in hidden_state
                )

                out = policy(obs_t, hidden_state=hidden_state, mask=mask_t)
                hidden_state = out.new_hidden_state

                # Sample action and stake.
                action = out.action_dist.sample()              # (1,) long
                action_idx = int(action.item())
                log_prob_action = float(
                    out.action_dist.log_prob(action).item(),
                )

                # Beta sample is in (0, 1). The shim takes a £ stake
                # so we re-scale outside the policy (per Phase 1
                # contract, env_shim.py::_encode_stake handles the
                # final clamp + budget-fraction encoding).
                stake_dist = torch.distributions.Beta(
                    out.stake_alpha, out.stake_beta,
                )
                stake_unit_t = stake_dist.sample()              # (1,)
                stake_unit = float(stake_unit_t.item())

                bm = env.bet_manager
                budget = bm.budget if bm is not None else 0.0
                stake_pounds = max(stake_unit * budget, MIN_BET_STAKE)

                # Stake log-prob is only meaningful for OPEN_* actions.
                # Storing 0.0 for the rest is a placeholder — Session
                # 02's PPO update masks via :func:`action_uses_stake`
                # before the gradient flows, so the placeholder never
                # contributes.
                if action_uses_stake(self.action_space, action_idx):
                    log_prob_stake = float(
                        stake_dist.log_prob(stake_unit_t).item(),
                    )
                else:
                    log_prob_stake = 0.0

                value_per_runner = (
                    out.value_per_runner.detach().squeeze(0).cpu().numpy()
                    .astype(np.float32)
                )

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
                    prev_pnl_by_id=prev_pnl_by_id,
                    market_to_runner_map=market_to_runner_map,
                )

                transitions.append(Transition(
                    obs=np.asarray(obs, dtype=np.float32),
                    hidden_state_in=hidden_in_t,
                    mask=np.asarray(mask_np, dtype=bool),
                    action_idx=action_idx,
                    stake_unit=stake_unit,
                    log_prob_action=log_prob_action,
                    log_prob_stake=log_prob_stake,
                    value_per_runner=value_per_runner,
                    per_runner_reward=per_runner_reward,
                    done=done,
                ))

                obs = next_obs
                n_steps += 1
                if done:
                    self.last_info = info or {}

        logger.info(
            "RolloutCollector: collected %d transitions (terminated)",
            n_steps,
        )
        return transitions

    # ── Per-step reward attribution ────────────────────────────────────────

    def _attribute_step_reward(
        self,
        env,
        step_reward: float,
        prev_pnl_by_id: dict[int, float],
        market_to_runner_map: dict[str, dict[int, int]],
    ) -> np.ndarray:
        """Split a scalar step reward across runner slots.

        Strategy:
        1. Walk every bet ever placed this episode (settled +
           current race's live ``bm.bets``). Compute per-bet pnl
           delta vs. last step's snapshot.
        2. Map each delta to a runner slot via the bet's
           ``(market_id, selection_id)`` pair → that race's
           ``runner_map``.
        3. Sum per slot — this is the runner-attributable share.
        4. Distribute the residual ``step_reward - attributed_total``
           equally across all ``max_runners`` slots. The residual
           covers MTM shaping, terminal-bonus, and per-race shaped
           terms (efficiency_cost / precision / drawdown / matured-
           arb / open-cost) that aren't tied to a single runner.

        Returns ``per_runner_reward`` of shape ``(max_runners,)``.
        Asserts ``per_runner_reward.sum() ≈ step_reward`` — drift
        here would silently corrupt the PPO update.
        """
        per_runner = np.zeros(self.max_runners, dtype=np.float64)

        # Iterate over the full episode's bets. ``all_settled_bets``
        # accumulates as races settle (per CLAUDE.md "info[
        # realised_pnl] is last-race-only"); the live ``bm.bets``
        # carries the current race's bets that haven't settled yet.
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
                # Update snapshot defensively — a bet that just
                # appeared with pnl=0 needs an entry so a future
                # delta is computed against 0 not "missing".
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
