"""Per-mini-batch advantage-normalisation stability tests.

plans/policy-startup-stability, Session 01 (2026-04-18).

Reproduces (as a synthetic, deterministic unit test) the first-update
policy-loss spike observed in production agent ``3e37822e-c9fa`` and
asserts that the per-mini-batch normalisation added in
``agents/ppo_trainer.py`` prevents the spike + the subsequent
action-head saturation.

Strategy
--------

The real ``PPOTrainer._ppo_update`` is a large method with auxiliary
losses (fill-prob BCE, risk NLL), an entropy-floor controller and
per-head entropy tracking, none of which are relevant to the
normalisation invariant. Rather than toggle the normalisation on/off
inside that method (which would require a test-only feature flag
polluting production code), this module reproduces the tiny slice of
the PPO surrogate loss that matters — the ratio, the clipped
surrogate, the policy-loss reduction, one optimiser step — as a
helper with a ``normalise: bool`` switch. Real obs/action/log-prob
tensors are produced by calling the trainer's own
``_collect_rollout`` on a synthetic ``Day``, so the policy network
sees correctly-shaped inputs; only the large-magnitude advantage
tensor is synthesised.

This is the load-bearing check referenced by
``plans/policy-startup-stability/hard_constraints.md`` §16.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from agents.architecture_registry import create_policy
from agents.ppo_trainer import PPOTrainer, Rollout, Transition
from data.episode_builder import Day, PriceSize, Race, RunnerMeta, RunnerSnap, Tick
from env.betfair_env import ACTIONS_PER_RUNNER


# ── Synthetic data helpers (mirror tests/test_ppo_trainer.py) ────────────────


def _make_runner_meta(selection_id: int, name: str = "Horse") -> RunnerMeta:
    return RunnerMeta(
        selection_id=selection_id, runner_name=name, sort_priority="1",
        handicap="0", sire_name="Sire", dam_name="Dam", damsire_name="DamSire",
        bred="GB", official_rating="85", adjusted_rating="85", age="4",
        sex_type="GELDING", colour_type="BAY", weight_value="140",
        weight_units="LB", jockey_name="J Smith", jockey_claim="0",
        trainer_name="T Jones", owner_name="Owner", stall_draw="3",
        cloth_number="1", form="1234", days_since_last_run="14", wearing="",
        forecastprice_numerator="3", forecastprice_denominator="1",
    )


def _make_runner_snap(
    selection_id: int, ltp: float = 4.0, back_price: float = 4.0,
    lay_price: float = 4.2, size: float = 100.0, status: str = "ACTIVE",
) -> RunnerSnap:
    return RunnerSnap(
        selection_id=selection_id, status=status, last_traded_price=ltp,
        total_matched=500.0, starting_price_near=0.0, starting_price_far=0.0,
        adjustment_factor=None, bsp=None, sort_priority=1, removal_date=None,
        available_to_back=[PriceSize(price=back_price, size=size)],
        available_to_lay=[PriceSize(price=lay_price, size=size)],
    )


def _make_tick(
    market_id: str, seq: int, runners: list[RunnerSnap],
    start_time: datetime | None = None, timestamp: datetime | None = None,
    in_play: bool = False, winner: int | None = None,
) -> Tick:
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    if timestamp is None:
        timestamp = start_time - timedelta(seconds=600 - seq * 5)
    return Tick(
        market_id=market_id, timestamp=timestamp, sequence_number=seq,
        venue="Newmarket", market_start_time=start_time,
        number_of_active_runners=len(runners), traded_volume=10000.0,
        in_play=in_play, winner_selection_id=winner, race_status=None,
        temperature=15.0, precipitation=0.0, wind_speed=5.0,
        wind_direction=180.0, humidity=60.0, weather_code=0, runners=runners,
    )


def _make_race(
    market_id: str = "1.200000001", start_time: datetime | None = None,
    n_ticks: int = 5, n_runners: int = 3, winner_sid: int = 1,
) -> Race:
    if start_time is None:
        start_time = datetime(2026, 3, 26, 14, 0, 0)
    runner_ids = list(range(1, n_runners + 1))
    runners = [_make_runner_snap(sid, ltp=3.0 + sid) for sid in runner_ids]
    ticks: list[Tick] = []
    for i in range(n_ticks):
        ts = start_time - timedelta(seconds=600 - i * 5)
        ticks.append(_make_tick(
            market_id, seq=i, runners=runners,
            start_time=start_time, timestamp=ts,
            in_play=False, winner=winner_sid,
        ))
    ticks.append(_make_tick(
        market_id, seq=n_ticks, runners=runners, start_time=start_time,
        timestamp=start_time + timedelta(seconds=5),
        in_play=True, winner=winner_sid,
    ))
    runner_meta = {sid: _make_runner_meta(sid, f"Horse{sid}") for sid in runner_ids}
    return Race(
        market_id=market_id, venue="Newmarket",
        market_start_time=start_time, winner_selection_id=winner_sid,
        ticks=ticks, runner_metadata=runner_meta,
    )


def _make_day(n_races: int = 1, n_ticks: int = 5, n_runners: int = 3) -> Day:
    races = []
    for i in range(n_races):
        start = datetime(2026, 3, 26, 14 + i, 0, 0)
        races.append(_make_race(
            market_id=f"1.{200000001 + i}", start_time=start,
            n_ticks=n_ticks, n_runners=n_runners, winner_sid=1,
        ))
    return Day(date="2026-03-26", races=races)


def _make_config() -> dict:
    return {
        "training": {
            "architecture": "ppo_lstm_v1",
            "starting_budget": 100.0,
            "max_runners": 14,
        },
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "coefficients": {
                "win_rate": 0.35, "sharpe": 0.30,
                "mean_daily_pnl": 0.15, "efficiency": 0.20,
            },
        },
        "paths": {
            "processed_data": "data/processed",
            "model_weights": "registry/weights",
            "logs": "logs",
            "registry_db": "registry/models.db",
        },
    }


def _make_policy(config: dict, seed: int = 0):
    """Freshly-initialised policy, seeded so the A/B comparison
    (normalised vs un-normalised) starts from the same weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_runners = config["training"]["max_runners"]
    from env.betfair_env import (
        AGENT_STATE_DIM, MARKET_DIM, POSITION_DIM, RUNNER_DIM, VELOCITY_DIM,
    )
    obs_dim = (
        MARKET_DIM + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )
    action_dim = max_runners * ACTIONS_PER_RUNNER
    return create_policy(
        "ppo_lstm_v1", obs_dim, action_dim, max_runners,
        hyperparams={"lstm_hidden_size": 64, "mlp_hidden_size": 32, "mlp_layers": 1},
    )


def _collect_real_batch(policy, n_min: int = 16):
    """Produce real, correctly-shaped obs tensors by letting the
    trainer's own rollout machinery drive the env. Truncate to
    ``n_min`` samples so the test is fast.

    For the surrogate step, we **re-derive** ``actions`` and
    ``old_log_probs`` from a single non-LSTM forward pass of the
    policy over the whole batch. The rollout's stored ``log_prob``
    values were computed with LSTM state carried forward per step,
    whereas the PPO update (and this test's surrogate step) treat
    each transition independently with hidden_state=None — so using
    the rollout's log_probs directly would give a non-unit ratio on
    the very first mini-batch, spuriously amplifying policy_loss in
    the normalised case too. Re-deriving keeps ``ratio == 1`` on
    update 0, which is exactly the first-update condition we want
    to test.

    Returns ``(obs_t, action_t, old_log_prob_t)``.
    """
    config = _make_config()
    trainer = PPOTrainer(
        policy, config,
        hyperparams={"ppo_epochs": 1, "mini_batch_size": 8},
    )
    day = _make_day(n_races=1, n_ticks=8, n_runners=3)
    rollout, _ = trainer._collect_rollout(day)
    transitions = rollout.transitions[:n_min]
    obs = torch.from_numpy(
        np.array([t.obs for t in transitions], dtype=np.float32)
    )

    with torch.no_grad():
        out = policy(obs)
        dist = Normal(out.action_mean, out.action_log_std.exp())
        # Seeded sample so both A/B runs use identical actions.
        gen = torch.Generator().manual_seed(12345)
        actions = (
            out.action_mean
            + out.action_log_std.exp()
            * torch.randn(
                out.action_mean.shape, generator=gen,
                dtype=out.action_mean.dtype,
            )
        )
        old_log_probs = dist.log_prob(actions).sum(dim=-1)

    return obs, actions, old_log_probs


def _make_large_advantages(n: int, seed: int = 0) -> torch.Tensor:
    """Synthesise large-magnitude advantages (±£2000 range) with a
    materially negative mean — matches the first-rollout scalping
    scale. The negative bias is deliberate: with ratio ≈ 1 on the
    first mini-batch, ``policy_loss = -advantages.mean()``, so a
    negative mean lands a positive spike (matching the production
    failure mode, where the un-normalised first update produced a
    large positive policy_loss)."""
    rng = np.random.default_rng(seed + 2000)
    adv_np = rng.uniform(-2000.0, 2000.0, size=n).astype(np.float32)
    adv_np -= 400.0
    return torch.from_numpy(adv_np)


def _run_one_surrogate_step(
    policy, obs: torch.Tensor, actions: torch.Tensor,
    old_log_probs: torch.Tensor, advantages: torch.Tensor,
    *, normalise: bool, clip_epsilon: float = 0.2, lr: float = 1e-4,
) -> tuple[float, float]:
    """Compute the PPO clipped-surrogate policy_loss on the given
    batch, take one vanilla SGD step, return
    ``(policy_loss_value, action_mean_shift)``.

    The normalisation branch exactly mirrors the production code in
    ``agents/ppo_trainer.py::_ppo_update`` (the per-mini-batch block
    directly above the ``ratio = ...`` line).

    SGD (rather than Adam) is used so the action-head shift is
    proportional to the raw gradient magnitude. Adam normalises the
    step size per-parameter, which would obscure the very effect the
    test is designed to demonstrate: that large-magnitude advantages
    produce large weight shifts. ``max_grad_norm`` is likewise NOT
    applied — piling gradient clipping on top would let the
    un-normalised case pass by virtue of the clamp rather than
    exposing the failure mode.
    """
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)

    with torch.no_grad():
        mean_before = float(policy(obs).action_mean.mean().item())

    adv = advantages.clone()
    if normalise and adv.numel() > 1:
        adv_mean = adv.mean()
        adv_std = adv.std() + 1e-8
        adv = (adv - adv_mean) / adv_std

    out = policy(obs)
    std = out.action_log_std.exp()
    dist = Normal(out.action_mean, std)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)

    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    with torch.no_grad():
        mean_after = float(policy(obs).action_mean.mean().item())

    return float(policy_loss.item()), mean_after - mean_before


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAdvantageNormalisationStability:
    """Per-batch advantage normalisation prevents the first-update
    policy-loss explosion that would otherwise saturate action heads.

    Reproduces the failure mode observed in production agent
    3e37822e-c9fa (2026-04-18 morning activation-A-baseline run): on
    a fresh policy with large-magnitude rewards, the first PPO update
    produced policy_loss = 3.35e+14 and saturated ``close_signal``,
    so the head never fired again across the agent's remaining 14
    episodes.
    """

    def _fresh_policy_and_rollout(self, with_normalisation: bool):
        """Build a freshly-initialised policy + a synthetic rollout
        batch with deliberately-large advantage magnitudes, run one
        PPO surrogate step, return
        ``(policy_loss, action_head_mean_shift)``.

        Seeded so the two (normalised / un-normalised) runs use the
        same starting policy and the same batch — the only difference
        is whether the normalisation branch is executed.
        """
        config = _make_config()
        policy = _make_policy(config, seed=0)
        obs, actions, old_log_probs = _collect_real_batch(policy, n_min=16)
        advantages = _make_large_advantages(len(obs), seed=0)
        return _run_one_surrogate_step(
            policy, obs, actions, old_log_probs, advantages,
            normalise=with_normalisation,
        )

    def test_unnormalised_first_update_spikes(self):
        """Without normalisation, large-reward rollouts produce a
        catastrophic policy_loss on the first update.

        On the very first mini-batch, ``ratio == 1`` for every sample
        (the policy has not moved since the rollout), so the policy
        loss degenerates to ``-advantages.mean()``. With advantages
        in the ±£2000 range and a non-zero mean, that is exactly the
        multi-hundred spike that destabilised the production run.
        """
        loss, _ = self._fresh_policy_and_rollout(with_normalisation=False)
        assert loss > 100, (
            f"expected spike >100 on un-normalised update; got {loss}"
        )

    def test_normalised_first_update_stays_bounded(self):
        """With normalisation, the same rollout produces a bounded
        policy_loss on the first update.

        After per-batch normalisation, ``advantages.mean() == 0`` so
        ``-advantages.mean()`` at ratio=1 is exactly zero (floating-
        point residual keeps the actual value very close to 0)."""
        loss, _ = self._fresh_policy_and_rollout(with_normalisation=True)
        assert abs(loss) < 5, (
            f"expected bounded |loss|<5 with normalisation; got {loss}"
        )

    def test_normalisation_dampens_action_head_shift(self):
        """The action_head's output mean must shift materially less
        in the normalised case — the principled check that the fix
        prevents head saturation, not just dampens the loss magnitude.

        §Acceptance of the session-01 prompt: this is the
        load-bearing invariant. Without normalisation, a single
        first-update gradient shifts the mean by a large amount; with
        it, the shift is bounded to at most a fifth of the
        un-normalised case.
        """
        _, shift_un = self._fresh_policy_and_rollout(with_normalisation=False)
        _, shift_norm = self._fresh_policy_and_rollout(with_normalisation=True)
        assert abs(shift_norm) < abs(shift_un) / 5, (
            f"expected normalised shift << un-normalised; "
            f"got norm={shift_norm}, un-norm={shift_un}"
        )


# ── Integration: the real _ppo_update path is bounded ────────────────────────


class TestRealTrainerUpdateBounded:
    """The production ``PPOTrainer._ppo_update`` path must produce a
    bounded policy_loss on a synthetic high-magnitude rollout. This
    is the integration-level equivalent of the unit tests above and
    directly exercises the edit in ``agents/ppo_trainer.py``.
    """

    def test_real_update_policy_loss_bounded(self, tmp_path):
        """Feed the real ``_ppo_update`` a rollout with forced
        large-magnitude advantages; assert the final policy loss is
        bounded. Rollout transitions are constructed so that the
        rollout's ``log_prob`` matches what the update loop's
        (no-LSTM-state) forward pass will produce, isolating the
        normalisation effect from the orthogonal LSTM-state
        mismatch. A monkey-patched ``_compute_advantages`` injects
        the high-magnitude advantage tensor that would otherwise
        produce a first-update spike.
        """
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config, seed=0)
        trainer = PPOTrainer(
            policy, config,
            hyperparams={"ppo_epochs": 1, "mini_batch_size": 8},
        )

        # Use _collect_rollout only to produce correctly-shaped obs;
        # synthesise (action, log_prob) via a single no-LSTM forward
        # pass so _ppo_update's ratio starts at 1 on update 0.
        day = _make_day(n_races=1, n_ticks=8, n_runners=3)
        src_rollout, _ = trainer._collect_rollout(day)
        n = len(src_rollout.transitions)
        obs_np = np.stack([t.obs for t in src_rollout.transitions], axis=0)
        with torch.no_grad():
            out = policy(torch.from_numpy(obs_np))
            std = out.action_log_std.exp()
            dist = Normal(out.action_mean, std)
            gen = torch.Generator().manual_seed(54321)
            actions = (
                out.action_mean
                + std * torch.randn(
                    out.action_mean.shape, generator=gen,
                    dtype=out.action_mean.dtype,
                )
            )
            lp = dist.log_prob(actions).sum(dim=-1).numpy()
        actions_np = actions.numpy()

        rollout = Rollout()
        for i in range(n):
            rollout.append(
                Transition(
                    obs=obs_np[i].astype(np.float32),
                    action=actions_np[i].astype(np.float32),
                    log_prob=float(lp[i]),
                    value=0.0,
                    reward=0.0,
                    done=(i == n - 1),
                    training_reward=0.0,
                )
            )

        adv = _make_large_advantages(n, seed=0)
        ret = torch.zeros(n, dtype=torch.float32)
        trainer._compute_advantages = lambda _r: (adv, ret)  # type: ignore[assignment]

        loss_info = trainer._ppo_update(rollout)

        assert np.isfinite(loss_info["policy_loss"]), (
            f"policy_loss not finite: {loss_info['policy_loss']}"
        )
        assert abs(loss_info["policy_loss"]) < 100, (
            "real _ppo_update should produce a bounded policy_loss "
            "on high-magnitude advantages thanks to per-mini-batch "
            f"normalisation; got {loss_info['policy_loss']}"
        )


# ── LR warmup (Session 01 stretch goal, shipped) ─────────────────────────────


class TestLRWarmup:
    """First-5-update linear LR warmup — the defence-in-depth that
    pairs with the per-mini-batch advantage normalisation.

    Shipped because the smoke test showed an episode-1 policy_loss
    of 1.48e+12 with normalisation alone — a residual spike driven
    by high first-rollout variance through a fresh LSTM. The warmup
    scales ``optimiser.lr`` linearly from ``base/5`` on update 0
    to the full ``base`` on update 4, and clamps at ``base`` from
    update 5 onward.
    """

    def _make_trainer(self, tmp_path, lr: float = 3e-4):
        config = _make_config()
        config["paths"]["logs"] = str(tmp_path / "logs")
        policy = _make_policy(config, seed=0)
        return PPOTrainer(
            policy, config,
            hyperparams={
                "learning_rate": lr,
                "ppo_epochs": 1,
                "mini_batch_size": 8,
            },
        )

    def _run_one_dummy_update(self, trainer):
        """Run one ``_ppo_update`` on a tiny synthetic rollout so the
        update-count counter advances. Uses a no-LSTM-state matched
        (action, log_prob) so the update completes cleanly without
        incidentally blowing up."""
        day = _make_day(n_races=1, n_ticks=4, n_runners=3)
        src, _ = trainer._collect_rollout(day)
        n = len(src.transitions)
        obs_np = np.stack([t.obs for t in src.transitions], axis=0)
        with torch.no_grad():
            out = trainer.policy(torch.from_numpy(obs_np))
            std = out.action_log_std.exp()
            dist = Normal(out.action_mean, std)
            gen = torch.Generator().manual_seed(999)
            actions = (
                out.action_mean
                + std * torch.randn(
                    out.action_mean.shape, generator=gen,
                    dtype=out.action_mean.dtype,
                )
            )
            lp = dist.log_prob(actions).sum(dim=-1).numpy()
        actions_np = actions.numpy()

        rollout = Rollout()
        for i in range(n):
            rollout.append(
                Transition(
                    obs=obs_np[i].astype(np.float32),
                    action=actions_np[i].astype(np.float32),
                    log_prob=float(lp[i]),
                    value=0.0,
                    reward=0.0,
                    done=(i == n - 1),
                    training_reward=0.0,
                )
            )
        trainer._ppo_update(rollout)

    def test_base_lr_captured_at_init(self, tmp_path):
        """``_base_learning_rate`` captures the configured lr at
        ``__init__`` — the warmup rescales off this immutable
        reference, so mid-run lr perturbations (the env doesn't do
        this, but defensively) can't drift the ramp."""
        trainer = self._make_trainer(tmp_path, lr=3e-4)
        assert trainer._base_learning_rate == pytest.approx(3e-4)
        assert trainer._update_count == 0
        assert trainer._lr_warmup_updates == 5

    def test_lr_ramps_linearly_over_five_updates(self, tmp_path):
        """After ``k`` updates, optimiser lr == base * (k / 5), up to
        update 5 where it stays pinned at ``base`` forever."""
        base = 3e-4
        trainer = self._make_trainer(tmp_path, lr=base)
        observed = []
        for _ in range(7):
            self._run_one_dummy_update(trainer)
            observed.append(trainer.optimiser.param_groups[0]["lr"])

        # After update k (counter is now k), lr used DURING update k
        # was ``base * min(1, (k_at_start+1)/5)`` = ``base * min(1, k/5)``
        # for k in 1..7. So observed[0] (= lr after update 0) = base * 1/5,
        # observed[1] = base * 2/5, ..., observed[4] = base * 5/5 = base,
        # observed[5] = base, observed[6] = base.
        expected = [base * min(1.0, (i + 1) / 5.0) for i in range(7)]
        for got, want in zip(observed, expected):
            assert got == pytest.approx(want), (
                f"expected {want}, got {got} (observed: {observed})"
            )

    def test_lr_stays_at_base_after_warmup(self, tmp_path):
        """After many updates, lr is pinned at base — the ramp never
        overshoots or drifts."""
        base = 1e-3
        trainer = self._make_trainer(tmp_path, lr=base)
        for _ in range(10):
            self._run_one_dummy_update(trainer)
        assert trainer.optimiser.param_groups[0]["lr"] == pytest.approx(base)

    def test_update_0_lr_is_reduced(self, tmp_path):
        """Invariant: the lr applied during update 0 is materially
        less than base — this is the part of the warmup that
        prevents the first-update spike."""
        base = 3e-4
        trainer = self._make_trainer(tmp_path, lr=base)
        self._run_one_dummy_update(trainer)
        # Update 0 used lr = base * 1/5; after update 0 the optimiser
        # still shows that value (the next update will rescale it).
        assert trainer.optimiser.param_groups[0]["lr"] < base / 2
