"""Phase 10 — deterministic action path in RolloutCollector + cohort wiring.

Tests 1–5 (S01): ``deterministic=True`` kwarg on RolloutCollector.collect_episode.
Tests 6–7 (S02): ``argmax_eval`` flag plumbed through run_cohort → scoreboard.

All tests run on stub policy / fixture rollouts — no real env data, no GPU
required.

Session prompts:
    plans/rewrite/phase-10-argmax-eval/session_prompts/
    01_add_deterministic_action_path.md
    02_wire_into_eval_codepaths.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import BaseDiscretePolicy, DiscretePolicyOutput
from training_v2.discrete_ppo.rollout import RolloutCollector
from training_v2.discrete_ppo.transition import action_uses_stake


# ── Test constants ─────────────────────────────────────────────────────────

_MAX_RUNNERS = 1   # action_space.n = 4: NOOP, OB_0, OL_0, CL_0
_OBS_DIM = 8
_N_STEPS = 10      # stub episode length


# ── Stub env + shim ────────────────────────────────────────────────────────


class _StubEnv:
    """Minimal env stub: no bets, no races, terminates after N steps."""

    def __init__(self, n_steps: int = _N_STEPS) -> None:
        self._n_steps = n_steps
        self._step_count = 0
        self._settled_bets: list = []
        self.bet_manager = None
        self._runner_maps: dict = {}

    @property
    def day(self):
        class _Day:
            races: list = []
        return _Day()


class _StubShim:
    """Minimal shim stub over ``_StubEnv``.

    Terminates after ``n_steps`` calls to ``step``.  Every obs is a
    zero vector; the action mask allows all actions.
    """

    def __init__(
        self,
        n_steps: int = _N_STEPS,
        max_runners: int = _MAX_RUNNERS,
    ) -> None:
        self.env = _StubEnv(n_steps=n_steps)
        self.max_runners = max_runners
        self.action_space = DiscreteActionSpace(max_runners)
        self._n_steps = n_steps
        self._step_count = 0
        self._obs = np.zeros(_OBS_DIM, dtype=np.float32)

    def reset(self):
        self._step_count = 0
        self.env._step_count = 0
        return self._obs.copy(), {}

    def get_action_mask(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=bool)

    def step(self, action_idx: int, stake=None, arb_spread=None):
        self._step_count += 1
        done = self._step_count >= self._n_steps
        return self._obs.copy(), 0.0, done, False, {}


# ── Stub policy ────────────────────────────────────────────────────────────


class _StubPolicy(BaseDiscretePolicy):
    """Policy that returns fixed logits / alpha / beta every forward call.

    Parameters
    ----------
    logits:
        Raw action logits (length ``action_space.n``).  Use ``-inf``
        to hard-mask an action as in test 5.
    stake_alpha, stake_beta:
        Beta-distribution parameters.  Both must be > 0.
    record_outputs:
        When ``True``, every ``DiscretePolicyOutput`` is appended to
        ``self.recorded_outputs`` so tests can inspect them post-run.
    """

    def __init__(
        self,
        action_space: DiscreteActionSpace,
        logits: list[float],
        stake_alpha: float = 2.0,
        stake_beta: float = 1.0,
        record_outputs: bool = False,
    ) -> None:
        super().__init__(
            obs_dim=_OBS_DIM,
            action_space=action_space,
            hidden_size=4,
        )
        self._logits = torch.tensor(logits, dtype=torch.float32)
        self._alpha = torch.tensor([stake_alpha], dtype=torch.float32)
        self._beta = torch.tensor([stake_beta], dtype=torch.float32)
        self._value = torch.zeros(1, action_space.max_runners, dtype=torch.float32)
        self.record_outputs = record_outputs
        self.recorded_outputs: list[DiscretePolicyOutput] = []

    def init_hidden(self, batch: int = 1) -> tuple[torch.Tensor, ...]:
        h = torch.zeros(1, batch, self.hidden_size)
        c = torch.zeros(1, batch, self.hidden_size)
        return (h, c)

    def forward(
        self,
        obs,
        hidden_state=None,
        mask=None,
    ) -> DiscretePolicyOutput:
        if hidden_state is None:
            hidden_state = self.init_hidden(batch=1)

        logits = self._logits.unsqueeze(0)   # (1, n_actions)

        if mask is not None:
            masked_logits = logits.masked_fill(~mask, float("-inf"))
        else:
            masked_logits = logits

        action_dist = Categorical(logits=masked_logits)

        out = DiscretePolicyOutput(
            logits=logits,
            masked_logits=masked_logits,
            action_dist=action_dist,
            stake_alpha=self._alpha,
            stake_beta=self._beta,
            value_per_runner=self._value,
            new_hidden_state=hidden_state,
            fill_prob_per_runner=torch.zeros(1, self.max_runners),
            mature_prob_per_runner=torch.zeros(1, self.max_runners),
            predicted_locked_pnl_per_runner=torch.zeros(1, self.max_runners),
            predicted_locked_log_var_per_runner=torch.zeros(1, self.max_runners),
            direction_back_prob_per_runner=torch.full(
                (1, self.max_runners), 0.5,
            ),
            direction_lay_prob_per_runner=torch.full(
                (1, self.max_runners), 0.5,
            ),
            direction_back_logits_per_runner=torch.zeros(1, self.max_runners),
            direction_lay_logits_per_runner=torch.zeros(1, self.max_runners),
        )

        if self.record_outputs:
            self.recorded_outputs.append(out)

        return out


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_collector(
    logits: list[float],
    stake_alpha: float = 2.0,
    stake_beta: float = 1.0,
    record_outputs: bool = False,
    n_steps: int = _N_STEPS,
    max_runners: int = _MAX_RUNNERS,
) -> RolloutCollector:
    shim = _StubShim(n_steps=n_steps, max_runners=max_runners)
    policy = _StubPolicy(
        action_space=shim.action_space,
        logits=logits,
        stake_alpha=stake_alpha,
        stake_beta=stake_beta,
        record_outputs=record_outputs,
    )
    return RolloutCollector(shim=shim, policy=policy, device="cpu")


# ── Tests ──────────────────────────────────────────────────────────────────


def test_collector_deterministic_action_is_argmax_of_logits():
    """Under ``deterministic=True`` every transition's action equals
    the argmax of the policy's logits.

    Logits ``[1.0, 5.0, 2.0, 0.5]`` with ``max_runners=1``
    (action_space.n=4) → argmax index = 1 (OPEN_BACK_0).
    """
    logits = [1.0, 5.0, 2.0, 0.5]
    expected_action = int(torch.tensor(logits).argmax().item())  # 1

    collector = _make_collector(logits=logits)
    batch = collector.collect_episode(deterministic=True)

    assert batch.n_steps == _N_STEPS
    assert all(
        int(a) == expected_action for a in batch.action_idx
    ), (
        f"Expected every action_idx == {expected_action}, "
        f"got {batch.action_idx.tolist()}"
    )


def test_collector_deterministic_stake_is_beta_mean():
    """Under ``deterministic=True`` every stake-using transition's
    ``stake_unit`` equals the Beta mean ``α / (α + β)``.

    ``stake_alpha=2.0``, ``stake_beta=1.0`` → mean = 2/3 ≈ 0.6667.
    """
    alpha, beta = 2.0, 1.0
    expected_mean = alpha / (alpha + beta)

    # Logits heavily favour OPEN_BACK_0 (index 1) so every step uses stake.
    logits = [0.0, 10.0, 0.0, 0.0]
    collector = _make_collector(logits=logits, stake_alpha=alpha, stake_beta=beta)
    batch = collector.collect_episode(deterministic=True)

    space = collector.shim.action_space
    for i in range(batch.n_steps):
        if action_uses_stake(space, int(batch.action_idx[i])):
            got = float(batch.stake_unit[i])
            assert abs(got - expected_mean) < 1e-5, (
                f"stake_unit at step {i}: expected {expected_mean:.6f}, "
                f"got {got:.6f}"
            )


def test_collector_default_is_stochastic_byte_identical():
    """Two runs at the same RNG seed with the default ``deterministic=False``
    produce bit-identical action, stake, log_prob_action, log_prob_stake
    sequences.

    This is the regression guard for the ``False`` default: confirms the
    stochastic path is untouched by the new kwarg.
    """
    logits = [1.0, 2.0, 3.0, 0.5]

    def _run(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        collector = _make_collector(logits=logits)
        return collector.collect_episode()   # no kwarg → default False

    batch_a = _run(seed=7)
    batch_b = _run(seed=7)

    assert batch_a.n_steps == batch_b.n_steps == _N_STEPS
    np.testing.assert_array_equal(
        batch_a.action_idx, batch_b.action_idx,
        err_msg="action_idx differed across same-seed stochastic runs",
    )
    np.testing.assert_array_equal(
        batch_a.stake_unit, batch_b.stake_unit,
        err_msg="stake_unit differed across same-seed stochastic runs",
    )
    np.testing.assert_array_equal(
        batch_a.log_prob_action, batch_b.log_prob_action,
        err_msg="log_prob_action differed across same-seed stochastic runs",
    )
    np.testing.assert_array_equal(
        batch_a.log_prob_stake, batch_b.log_prob_stake,
        err_msg="log_prob_stake differed across same-seed stochastic runs",
    )


def test_collector_log_prob_invariant_holds_under_deterministic():
    """Under ``deterministic=True`` the stored ``log_prob_action`` at
    each transition equals the action_dist's log-prob evaluated at the
    chosen (argmax) action.

    This is the load-bearing PPO-buffer invariant: ``log_prob_action``
    must always be ``action_dist.log_prob(chosen_action)``, regardless
    of how the action was selected.
    """
    logits = [1.0, 5.0, 2.0, 0.5]
    collector = _make_collector(logits=logits, record_outputs=True)
    policy: _StubPolicy = collector.policy  # type: ignore[assignment]

    batch = collector.collect_episode(deterministic=True)

    assert len(policy.recorded_outputs) == _N_STEPS

    for i, out in enumerate(policy.recorded_outputs):
        action_t = torch.tensor([int(batch.action_idx[i])], dtype=torch.long)
        expected_lp = float(
            out.action_dist.log_prob(action_t).detach().item()
        )
        stored_lp = float(batch.log_prob_action[i])
        assert abs(stored_lp - expected_lp) < 1e-5, (
            f"Step {i}: log_prob_invariant violated — "
            f"stored={stored_lp:.6f}, recomputed={expected_lp:.6f}"
        )


def test_collector_action_mask_respected_under_deterministic():
    """Under ``deterministic=True`` the argmax respects ``-inf`` logits.

    Policy logits ``[1.0, -inf, 5.0, -inf]`` with ``max_runners=1``
    (NOOP=0, OB_0=1, OL_0=2, CL_0=3) — actions 1 and 3 are hard-
    masked by the policy.  The highest unmasked logit is index 2
    (OL_0, value 5.0) so every transition must have action_idx == 2.
    """
    logits = [1.0, float("-inf"), 5.0, float("-inf")]
    expected_action = 2  # OL_0

    collector = _make_collector(logits=logits)
    batch = collector.collect_episode(deterministic=True)

    assert batch.n_steps == _N_STEPS
    assert all(
        int(a) == expected_action for a in batch.action_idx
    ), (
        f"Expected every action_idx == {expected_action}, "
        f"got {batch.action_idx.tolist()}"
    )


# ── S02: cohort-runner wiring ──────────────────────────────────────────────
#
# These tests exercise the argmax_eval flag flowing from run_cohort all the
# way through train_one_agent_fn and back into the scoreboard.  They use a
# minimal in-process stub and do NOT spin up a real env / GPU.


def _make_stub_result(agent_id: str, genes, days_to_train: list, eval_days: list, model_store):
    """Return a minimal AgentResult; write to model_store if provided."""
    from training_v2.cohort.worker import (
        AgentResult,
        EvalSummary,
        TrainSummary,
        arch_name_for_genes,
    )
    first_eval = eval_days[0] if eval_days else ""
    model_id = str(agent_id)
    weights_path = ""
    run_id = ""
    if model_store is not None:
        import torch as _torch
        from registry.model_store import EvaluationDayRecord
        model_id = model_store.create_model(
            generation=0,
            architecture_name=arch_name_for_genes(genes),
            architecture_description="stub",
            hyperparameters=genes.to_dict(),
            parent_a_id=None,
            parent_b_id=None,
            model_id=str(agent_id),
        )
        weights_path = model_store.save_weights(
            model_id=model_id,
            state_dict={"dummy": _torch.zeros(1)},
        )
        run_id = model_store.create_evaluation_run(
            model_id=model_id,
            train_cutoff_date=days_to_train[-1],
            test_days=[first_eval],
        )
        model_store.record_evaluation_day(EvaluationDayRecord(
            run_id=run_id, date=first_eval,
            day_pnl=0.0, bet_count=0, winning_bets=0,
            bet_precision=0.0, pnl_per_bet=0.0,
            early_picks=0, profitable=False,
        ))
        model_store.update_composite_score(model_id=model_id, score=0.0)
    return AgentResult(
        agent_id=str(agent_id),
        model_id=str(model_id),
        architecture_name=arch_name_for_genes(genes),
        genes=genes,
        train=TrainSummary(
            n_days=len(days_to_train), total_steps=10,
            total_reward=0.0, mean_reward=0.0, mean_pnl=0.0,
            mean_value_loss=0.0, mean_policy_loss=0.0,
            mean_approx_kl=0.0, wall_time_sec=0.01,
        ),
        eval=EvalSummary(
            eval_day=first_eval, total_reward=0.0, day_pnl=0.0,
            n_steps=10, bet_count=0, winning_bets=0, bet_precision=0.0,
            pnl_per_bet=0.0, early_picks=0, profitable=False,
            action_histogram={},
        ),
        weights_path=str(weights_path),
        run_id=str(run_id),
    )


def test_train_one_agent_argmax_eval_flag_reaches_fn(tmp_path):
    """run_cohort(argmax_eval=True/False) passes the matching bool to every
    train_one_agent_fn call via the argmax_eval kwarg.
    """
    from training_v2.cohort import runner as runner_mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for d in ["2026-04-21", "2026-04-22", "2026-04-23"]:
        (data_dir / f"{d}.parquet").write_bytes(b"")

    for argmax_flag in (False, True):
        captured = []

        def _capturing_stub(
            *, agent_id, genes, days_to_train, eval_days=None,
            eval_day=None, data_dir, device, seed, model_store,
            generation=0, parent_a_id=None, parent_b_id=None,
            **kwargs,
        ):
            captured.append(kwargs.get("argmax_eval", "NOT_SET"))
            _ev = eval_days or ([eval_day] if eval_day else [])
            return _make_stub_result(agent_id, genes, days_to_train, _ev, model_store)

        runner_mod.run_cohort(
            n_agents=2,
            n_generations=1,
            days=3,
            data_dir=data_dir,
            device="cpu",
            seed=42,
            output_dir=tmp_path / f"out_flag_{argmax_flag}",
            train_one_agent_fn=_capturing_stub,
            argmax_eval=argmax_flag,
        )

        assert len(captured) == 2, "Expected 2 agent calls"
        assert all(v is argmax_flag for v in captured), (
            f"Expected argmax_eval={argmax_flag!r} on all calls, got {captured}"
        )


def test_run_cohort_argmax_eval_scoreboard_eval_mode(tmp_path):
    """Scoreboard rows carry eval_mode='stochastic' or 'argmax' matching
    the argmax_eval kwarg passed to run_cohort.
    """
    from training_v2.cohort import runner as runner_mod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for d in ["2026-04-21", "2026-04-22", "2026-04-23"]:
        (data_dir / f"{d}.parquet").write_bytes(b"")

    def _simple_stub(
        *, agent_id, genes, days_to_train, eval_days=None,
        eval_day=None, data_dir, device, seed, model_store,
        generation=0, parent_a_id=None, parent_b_id=None,
        **kwargs,
    ):
        _ev = eval_days or ([eval_day] if eval_day else [])
        return _make_stub_result(agent_id, genes, days_to_train, _ev, model_store)

    for argmax_flag, expected_mode in [(False, "stochastic"), (True, "argmax")]:
        out_dir = tmp_path / f"out_mode_{argmax_flag}"
        runner_mod.run_cohort(
            n_agents=2,
            n_generations=1,
            days=3,
            data_dir=data_dir,
            device="cpu",
            seed=42,
            output_dir=out_dir,
            train_one_agent_fn=_simple_stub,
            argmax_eval=argmax_flag,
        )
        lines = (out_dir / "scoreboard.jsonl").read_text().splitlines()
        rows = [json.loads(ln) for ln in lines if ln.strip()]
        assert len(rows) == 2
        for row in rows:
            assert row.get("eval_mode") == expected_mode, (
                f"argmax_flag={argmax_flag}: expected eval_mode={expected_mode!r}, "
                f"got {row.get('eval_mode')!r}"
            )
